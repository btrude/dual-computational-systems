import os
import json
import uuid
from collections import defaultdict
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as t
from matplotlib.lines import Line2D
from pyfonts import load_font
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

from dual_computational_systems.data import prepare_dataset
from dual_computational_systems.models import prepare_model
from dual_computational_systems.models import print_model
from dual_computational_systems.train import train
from dual_computational_systems.util import set_seed
from dual_computational_systems.util.constants import CACHE_DIR
from dual_computational_systems.util.constants import OUTPUT_DIR


hn = load_font("assets/fonts/HelveticaNeue.ttf")


def split_units(x, y):
    base = x // y
    remainder = x % y
    return [base + 1 if i < remainder else base for i in range(y)]


class EvolutionaryFCNetwork(nn.Module):
    def __init__(
        self,
        channels=1,
        base_channels=250,
        border=100,
        dist_sparsity=0.975,
        device="cpu",
    ):
        super().__init__()

        per_model_channels_local = split_units((base_channels - border), 3)
        per_model_channels_dist = split_units((base_channels - (base_channels - border)), 2)
        self.per_model_channels_local = per_model_channels_local
        self.per_model_channels_dist = per_model_channels_dist

        vision_channels, audition_channels, somatosensation_channels = per_model_channels_local
        olfaction_channels, hippocampal_channels = per_model_channels_dist

        self.vision = prepare_model(
            "locally_connected_network",
            device=device,
            do_print_model=False,
            **{"base_channels": vision_channels, "channels": channels},
        )
        self.audition = prepare_model(
            "locally_connected_network",
            device=device,
            do_print_model=False,
            **{"base_channels": audition_channels, "channels": channels, "categories_out": 8},
        )
        self.somatosensation = prepare_model(
            "locally_connected_network",
            device=device,
            do_print_model=False,
            **{"base_channels": somatosensation_channels, "channels": channels, "categories_out": 25},
        )
        self.olfaction = prepare_model(
            "distributed_network",
            device=device,
            do_print_model=False,
            **{
                "base_channels": olfaction_channels,
                "channels": channels,
                "sparsity": dist_sparsity,
            },
        )
        self.hippocampus = prepare_model(
            "distributed_network",
            device=device,
            do_print_model=False,
            **{
                "base_channels": hippocampal_channels,
                "channels": channels,
                "categories_out": 1024,
                "sparsity": dist_sparsity,
            },
        )

    def forward(self, x):
        x_v, x_a, x_s, x_o, x_h = t.chunk(x, 5, dim=1)
        x_o = x_o.view(x_o.shape[0], -1)
        x_h = x_h.view(x_h.shape[0], -1)

        return (
            self.vision(x_v),
            self.audition(x_a),
            self.somatosensation(x_s),
            self.olfaction(x_o),
            self.hippocampus(x_h),
        )


class MultimodalDataset(Dataset):
    def __init__(self, train=True):
        self.dataset_name = "periphery"
        self.vision = prepare_dataset("cifar10", test=False, test_only=not train, disable_transform=True)
        self.olfaction = prepare_dataset("olfaction", test=False, test_only=not train, disable_transform=True)
        self.audition = prepare_dataset("audition", test=False, test_only=not train, disable_transform=True)
        self.somatosensation = prepare_dataset("somatosensation", test=False, test_only=not train, disable_transform=True)
        self.hippocampus = prepare_dataset("hippocampus", test=False, test_only=not train, disable_transform=True)

        self.vision_subtransform = transforms.Compose([
            transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.audition)

    # V -> A -> S -> O -> H
    def __getitem__(self, index):
        x_v, y_v = self.vision[index]
        x_a, y_a = self.audition[index]
        x_s, y_s = self.somatosensation[index]
        x_o, y_o = self.olfaction[index]
        x_h, y_h = self.hippocampus[index]

        x_v = self.vision_subtransform(x_v)
        x_s = t.tensor(x_s).squeeze().unsqueeze(0)
        x_h = t.tensor(x_h).squeeze().unsqueeze(0)

        x = t.cat((x_v, x_a, x_s, x_o, x_h))
        y = t.stack((t.tensor(y_v), t.tensor(y_a), t.tensor(y_s), y_o))

        return x, (y, t.tensor(y_h))


def fitness(s, v, o):
    return (s * v) + ((1 - s) * o)


def load_mutations_cache(mutations_cache_path):
    with open(mutations_cache_path) as cache_file:
        mutation_results = json.load(cache_file)

    modalities = ["v", "a", "s", "o", "h"]
    mutations_cache = t.zeros((len(modalities), 2500, 2500))
    for mutation, result in mutation_results.items():
        for modality, values in result.items():
            for i, value in enumerate(values):
                mutations_cache[modalities.index(modality), int(mutation), i] = value

    return mutations_cache


def get_mutation_pcts(
    model_base_channels,
    mutations,
    mutation_pcts_cache_path=None,
    dist_sparsity=0.4,
):
    if mutation_pcts_cache_path is not None:
        with open(mutation_pcts_cache_path, "r") as json_file:
            mutation_pcts = json.load(json_file)

        max_v, max_o = -1, -1
        for mut_info in mutation_pcts:
            if isinstance(mut_info, int):
                continue

            units_v, _, _, units_o, _, channels_v, *_ = mut_info
            max_v = max(units_v, max_v)
            max_o = max(channels_v, max_o)

        first = None
        for mutation, mut_info in enumerate(mutation_pcts):
            if isinstance(mut_info, int):
                continue

            (
                units_v, units_a, units_s, units_o, units_h,
                channels_v, channels_a, channels_s, channels_o, channels_h,
            ) = mut_info
            sparsity = 1 - dist_sparsity

            if first is None:
                first = units_v

            total = sum([units_v, units_a, units_s, units_o * sparsity, units_h * sparsity])
            mutation_pcts[mutation] = (
                units_v / total, units_a / total, units_s / total, (units_o * sparsity) / total, (units_h * sparsity) / total,
                channels_v, channels_a, channels_s, channels_o, channels_h,
            )

        return mutation_pcts

    mutation_pcts = [0 for _ in range(model_base_channels)]

    for mutation in tqdm(mutations):
        test_model = EvolutionaryFCNetwork(
            base_channels=model_base_channels,
            border=int(mutation),
        )
        test_model.train()

        channels_v, channels_a, channels_s = test_model.per_model_channels_local
        channels_o, channels_h = test_model.per_model_channels_dist
        units_v = sum([p.numel() for p in test_model.vision.parameters() if p.requires_grad])
        units_a = sum([p.numel() for p in test_model.somatosensation.parameters() if p.requires_grad])
        units_s = sum([p.numel() for p in test_model.audition.parameters() if p.requires_grad])
        units_o = sum([p.numel() for p in test_model.olfaction.parameters() if p.requires_grad])
        units_h = sum([p.numel() for p in test_model.hippocampus.parameters() if p.requires_grad])
        mutation_pcts[mutation] = (
            units_v, units_a, units_s, units_o, units_h,
            channels_v, channels_a, channels_s, channels_o, channels_h,
        )

    save_path = Path(CACHE_DIR) / f"mutation_pcts_{str(uuid.uuid4())[:8]}.json"
    print(f"Mutation percents saved at: {save_path}")

    with open(save_path, "w") as json_file:
        json.dump(mutation_pcts, json_file, indent=4, default=str)

    return get_mutation_pcts(model_base_channels, mutations, mutation_pcts_cache_path=save_path)


def random_choice_exclude_mask(values, size, exclude=None):
    values = np.array(values)

    if exclude is not None:
        mask = np.ones(len(values), dtype=bool)
        for ex in exclude:
            mask &= values != ex
        choices = values[mask]
    else:
        choices = values

    if len(choices) < size:
        size = len(choices)

    return np.random.choice(choices, size=size, replace=False)


def main(
    n_epochs=50,
    model_base_channels=315,
    n_generations=25,
    population_size=55,
    k_fit=15,
    mutation_sample_width=8,
    mutations_sparsity=1,
    mutations_cache_path=None,
    mutation_pcts_cache_path=None,
    max_generations_from_cache=50,
    max_survivors_scatter=float("inf"),
    allowed_mutation_drift=25,
    lr=0.01,
    dist_network_sparsity=0.4,
    device="cpu",
    verbose=False,
):
    set_seed(12345)
    S = np.arange(0.025, 1, 0.025)
    mutations = np.arange(2, model_base_channels - 2, mutations_sparsity)
    mutation_results = defaultdict(dict)

    if verbose:
        for mutation in mutations:
            per_model_channels_local = split_units((model_base_channels - mutation), 3)
            per_model_channels_dist = split_units((model_base_channels - (model_base_channels - mutation)), 2)
            print(f"mutation = {mutation}, local = {per_model_channels_local}, dist = {per_model_channels_dist}")

    if mutations_cache_path is None:
        train_dataset = MultimodalDataset()
        test_dataset = MultimodalDataset(train=False)

        for mutation in tqdm(mutations):
            model = EvolutionaryFCNetwork(
                base_channels=model_base_channels,
                border=int(mutation),
                dist_sparsity=dist_network_sparsity,
                device=device,
            )
            print_model(model.vision, write_func=tqdm.write)
            print_model(model.audition, write_func=tqdm.write)
            print_model(model.somatosensation, write_func=tqdm.write)
            print_model(model.olfaction, write_func=tqdm.write)
            print_model(model.hippocampus, write_func=tqdm.write)

            acc_v, acc_a, acc_s, acc_o, acc_h = train(
                model_override=model,
                optimizer_name="sgd",
                n_epochs=n_epochs,
                test_every=1,
                lr=lr,
                min_lr=0.,
                dataset_name="periphery",
                dataset_override=(train_dataset, test_dataset),
                return_accuracy=True,
                return_final_accuracy_only=False,
                skip_train_accuracy=True,
                log_metrics_inline=False,
                device=device,
                num_workers=8,
            )

            mutation_results[int(mutation)]["v"] = acc_v
            mutation_results[int(mutation)]["a"] = acc_a
            mutation_results[int(mutation)]["s"] = acc_s
            mutation_results[int(mutation)]["o"] = acc_o
            mutation_results[int(mutation)]["h"] = acc_h

            if verbose:
                tqdm.write(json.dumps(mutation_results[int(mutation)], indent=4, default=str))

        mutations_cache_out_path = Path(CACHE_DIR) / f"mutations_cache_{str(uuid.uuid4())[:8]}.json"
        print(f"Mutations cache saved at: {mutations_cache_out_path}")

        with open(mutations_cache_out_path, "w") as json_file:
            json.dump(mutation_results, json_file, indent=4, default=str)

    if mutations_cache_path is None:
        exit()

    S_plot_line = []
    mutations_v_line = []
    mutations_o_line = []

    S_plot_scatter = []
    mutations_v_scatter = []
    mutations_h_scatter = []
    mutations_o_scatter = []

    N = len(mutations)
    mutations_cache_path = mutations_cache_path.split(",")

    for mutation_cache_location in mutations_cache_path:
        mutation_pcts = get_mutation_pcts(
            model_base_channels,
            mutations,
            mutation_pcts_cache_path=mutation_pcts_cache_path,
        )
        mutations_cache = load_mutations_cache(mutation_cache_location)

        center = np.where(mutations == int(model_base_channels * .4))[0].item()
        window = (
            center - int(N // mutation_sample_width),
            center + int(N // mutation_sample_width),
        )
        window_v = window
        window_o = window
        window_orig = window

        for s in tqdm(S, desc="s ∈ S"):
            iter_mutations_v = np.random.choice(
                list(range(*window)),
                size=population_size,
            )
            iter_mutations_v = np.array(list(set(iter_mutations_v)))
            iter_mutations_o = iter_mutations_v.copy()

            for generation in range(n_generations):
                if generation > 0:
                    center_v = int(np.median(iter_mutations_v))
                    window_v = (
                        max(center_v - int(N // mutation_sample_width), 0),
                        min(center_v + int(N // mutation_sample_width), N - 1),
                    )

                    center_o = int(np.median(iter_mutations_o))
                    window_o = (
                        max(center_o - int(N // mutation_sample_width), 0),
                        min(center_o + int(N // mutation_sample_width), N - 1),
                    )

                    window_start_v, window_end_v = window_v
                    window_start_o, window_end_o = window_o

                    new_mutations_v = random_choice_exclude_mask(
                        list(range(window_start_v, window_end_v)),
                        size=population_size - k_fit,
                        exclude=iter_mutations_v,
                    )
                    new_mutations_o = random_choice_exclude_mask(
                        list(range(window_start_o, window_end_o)),
                        size=len(new_mutations_v),
                        exclude=iter_mutations_o,
                    )
                    new_mutations_v = new_mutations_v[:len(new_mutations_o)]
                    new_mutations_v = np.array(list(set(new_mutations_v)))
                    new_mutations_o = np.array(list(set(new_mutations_o)))

                    if not isinstance(iter_mutations_v, np.ndarray):
                        iter_mutations_v = np.array([iter_mutations_v])

                    if not isinstance(iter_mutations_o, np.ndarray):
                        iter_mutations_o = np.array([iter_mutations_o])

                    iter_mutations_v = np.concatenate((iter_mutations_v, new_mutations_v))
                    iter_mutations_o = np.concatenate((iter_mutations_o, new_mutations_o))

                last_avail_epoch = max_generations_from_cache - 1
                gen_i = generation if generation <= last_avail_epoch else last_avail_epoch
                cohort_v = mutations_cache[:1, mutations[iter_mutations_v], last_avail_epoch]
                cohort_o = mutations_cache[3:4, mutations[iter_mutations_o], last_avail_epoch]

                last_fit = fitness(s, cohort_v, cohort_o)
                _, survivors = t.topk(last_fit, k_fit)
                iter_mutations_v = iter_mutations_v[survivors.squeeze()]
                iter_mutations_o = iter_mutations_o[survivors.squeeze()]

                try:
                    iter(iter_mutations_v)  # sometimes iter_mutations_v is a single mutation
                    should_align = True
                except TypeError:
                    should_align = False

                if should_align:
                    for i, (mut_v, mut_o) in enumerate(list(zip(iter_mutations_v, iter_mutations_o))):
                        if np.abs(mut_v - mut_o) > allowed_mutation_drift:
                            iter_mutations_o[i] = iter_mutations_v[i]

            for i, (survivor_v, survivor_o) in enumerate(list(zip(
                mutations[iter_mutations_v],
                mutations[iter_mutations_o],
            ))):
                (
                    mutation_vision,
                    mutation_somatosensation,
                    mutation_audition,
                    _,
                    mutation_hippocampus,
                    *_,
                ) = mutation_pcts[survivor_v]
                (
                    _,
                    _,
                    _,
                    mutation_olfaction,
                    _,
                    *_,
                ) = mutation_pcts[survivor_o]

                mutations_v_line.append(mutation_vision)
                mutations_o_line.append(mutation_olfaction)
                S_plot_line.append(r)

                if i < max_survivors_scatter:
                    mutations_v_scatter.append(mutation_vision)
                    mutations_h_scatter.append(mutation_hippocampus)
                    mutations_o_scatter.append(mutation_olfaction)
                    S_plot_scatter.append(r)

        window = window_orig

    df = pd.DataFrame({
        "S": S_plot_scatter,
        "Vision": mutations_v_scatter,
        "Olfaction": mutations_o_scatter,
    })

    num_categories = 5
    bins = np.linspace(0, 1, num_categories + 1)
    labels = [f"{bins[i]:.1f} - {bins[i+1]:.1f}" for i in range(len(bins) - 1)]
    df["S_cat"] = pd.cut(df["S"], bins=bins, labels=labels, include_lowest=True)

    markers_list = ["o", "s", "D", "^", "H"]
    markers_dict = dict(zip(labels, markers_list))

    colors_list = ["red", "navy", "green", "deepskyblue", "yellow"]
    colors_dict = dict(zip(labels, colors_list))

    plt.figure(figsize=(10, 8))

    legend_handles = []
    for category in labels:
        mask = df["S_cat"] == category
        plt.scatter(
            df.loc[mask, "Olfaction"],
            df.loc[mask, "Vision"],
            marker=markers_dict[category],
            color=colors_dict[category],
            s=200,
            alpha=0.85,
            edgecolors="b",
            linewidths=0.5,
        )

        legend_handle = Line2D(
            [], [],
            marker=markers_dict[category],
            color="w",
            markerfacecolor=colors_dict[category],
            markeredgecolor="w",
            markersize=24,
            linewidth=0
        )
        legend_handles.append(legend_handle)

    xlabel = "Olfactory Units / Total"
    ylabel = "Visual Units / Total"

    plt.margins(x=0.1, y=0.1)

    plt.xlabel(xlabel, font=hn, fontsize=40)
    plt.ylabel(ylabel, font=hn, fontsize=40)

    ax = plt.gca()
    ax.xaxis.get_offset_text().set_fontproperties(hn)
    ax.xaxis.get_offset_text().set_fontsize(30)
    ax.yaxis.get_offset_text().set_fontproperties(hn)
    ax.yaxis.get_offset_text().set_fontsize(30)

    hn.set_size(30)
    plt.legend(
        handles=legend_handles,
        labels=labels,
        scatterpoints=1,
        prop=hn,
        fontsize=30,
        loc="best"
    )
    plt.xticks(font=hn, fontsize=30)
    plt.yticks(font=hn, fontsize=30)
    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(Path(OUTPUT_DIR) / "evo_vision.png", bbox_inches="tight", dpi=300)
    plt.close()

    df = pd.DataFrame({
        "S": S_plot_scatter,
        "Hippocampus": mutations_h_scatter,
        "Olfaction": mutations_o_scatter,
    })

    df["S_cat"] = pd.cut(df["S"], bins=bins, labels=labels, include_lowest=True)

    plt.figure(figsize=(10, 8))

    legend_handles = []
    for category in labels:
        mask = df["S_cat"] == category
        plt.scatter(
            df.loc[mask, "Olfaction"],
            df.loc[mask, "Hippocampus"],
            marker=markers_dict[category],
            color=colors_dict[category],
            s=250,
            alpha=0.85,
            edgecolors="b",
            linewidths=0.5,
        )

        legend_handle = Line2D(
            [], [],
            marker=markers_dict[category],
            color="w",
            markerfacecolor=colors_dict[category],
            markeredgecolor="w",
            markersize=24,
            linewidth=0
        )
        legend_handles.append(legend_handle)

    ylabel = "Hippocampal Units"
    xlabel = f"Olfactory Units / Total"
    ylabel = f"Hippocampal Units / Total"

    plt.margins(x=0.1, y=0.1)
    plt.xlabel(xlabel, font=hn, fontsize=40)
    plt.ylabel(ylabel, font=hn, fontsize=40)

    ax = plt.gca()
    ax.xaxis.get_offset_text().set_fontproperties(hn)
    ax.xaxis.get_offset_text().set_fontsize(30)
    ax.yaxis.get_offset_text().set_fontproperties(hn)
    ax.yaxis.get_offset_text().set_fontsize(30)

    hn.set_size(30)
    plt.legend(
        handles=legend_handles,
        labels=labels,
        scatterpoints=1,
        prop=hn,
        fontsize=30,
        loc="best"
    )
    plt.xticks(font=hn, fontsize=30)
    plt.yticks(font=hn, fontsize=30)
    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(Path(OUTPUT_DIR) / "evo_hippo.png", bbox_inches="tight", dpi=300)
    plt.close()


if __name__ == "__main__":
    fire.Fire(main)
