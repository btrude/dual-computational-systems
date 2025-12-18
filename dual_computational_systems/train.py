import os
from pathlib import Path

import evaluate
import fire
import torch as t
import torch.nn as nn
from torch.amp import autocast
from torch.nn.utils import clip_grad_norm_ as clip_grad_norm
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchmetrics.classification import BinaryAccuracy
from torch.utils.tensorboard import SummaryWriter

from dual_computational_systems.data import prepare_dataset
from dual_computational_systems.models import prepare_model
from dual_computational_systems.optim import prepare_opt
from dual_computational_systems.optim import prepare_scheduler
from dual_computational_systems.util.fsd50k import AUDITION_LOSS_WEIGHTS


def test_accuracy_periphery(model, test_loader, metric, device, modalities=2):
    hippocampus_accuracy = BinaryAccuracy()
    model.eval()
    (
        all_preds_v,
        all_preds_a,
        all_preds_s,
        all_preds_o,
        all_preds_h,
    ) = [], [], [], [], []
    (
        all_labels_v,
        all_labels_a,
        all_labels_s,
        all_labels_o,
        all_labels_h,
    ) = [], [], [], [], []

    with t.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)

            with autocast(device_type=device.type, dtype=t.bfloat16):
                outputs = model(inputs)

                (
                    outputs_v,
                    outputs_a,
                    outputs_s,
                    outputs_o,
                    outputs_h,
                ) = outputs
                labels, labels_h = labels
                labels_v = labels[:, 0]
                labels_a = labels[:, 1]
                labels_s = labels[:, 2]
                labels_o = labels[:, 3]
                labels_h = labels_h.view(labels_h.shape[0], -1)

                _, preds_v = t.max(outputs_v, dim=1)
                _, preds_a = t.max(outputs_a, dim=1)
                _, preds_s = t.max(outputs_s, dim=1)
                _, preds_o = t.max(outputs_o, dim=1)
                preds_h = outputs_h.float()

                all_preds_v.extend(preds_v.cpu().numpy())
                all_labels_v.extend(labels_v.int().numpy())

                all_preds_a.extend(preds_a.cpu().numpy())
                all_labels_a.extend(labels_a.int().numpy())

                all_preds_s.extend(preds_s.cpu().numpy())
                all_labels_s.extend(labels_s.int().numpy())

                all_preds_o.extend(preds_o.cpu().numpy())
                all_labels_o.extend(labels_o.int().numpy())

                all_preds_h.extend(preds_h.cpu())
                all_labels_h.extend(labels_h.int())

    model.train()

    if modalities == 2:
        results_v = metric.compute(predictions=all_preds_v, references=all_labels_v)
        results_o = metric.compute(predictions=all_preds_o, references=all_labels_o)
        return results_v["accuracy"], results_o["accuracy"]

    if modalities == 5:
        results_v = metric.compute(predictions=all_preds_v, references=all_labels_v)
        results_a = metric.compute(predictions=all_preds_a, references=all_labels_a)
        results_s = metric.compute(predictions=all_preds_s, references=all_labels_s)
        results_o = metric.compute(predictions=all_preds_o, references=all_labels_o)
        results_h = hippocampus_accuracy(t.stack(all_preds_h).cpu(), t.stack(all_labels_h).cpu())
        return (
            results_v["accuracy"],
            results_a["accuracy"],
            results_s["accuracy"],
            results_o["accuracy"],
            results_h.item(),
        )


def test_accuracy(model, test_loader, metric, device, multi_target=False):
    model.eval()
    all_preds = []
    all_labels = []

    with t.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)

            if multi_target:
                labels = labels.view(inputs.shape[0], -1)

            with autocast(device_type=device.type, dtype=t.bfloat16):
                outputs = model(inputs)

            if multi_target:
                preds = outputs.float()
                all_preds.extend(preds.cpu())
                all_labels.extend(labels.int())
            else:
                _, preds = t.max(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.int().numpy())

    model.train()

    if multi_target:
        accuracy = BinaryAccuracy()
        all_preds = t.stack(all_preds).cpu()
        all_labels = t.stack(all_labels).cpu()
        return accuracy(all_preds, all_labels)
    else:
        results = metric.compute(predictions=all_preds, references=all_labels)
        return results["accuracy"]


def test_sparsity(model):
    first_layer_units = model.features[0].weight.flatten().shape[0]
    zero_units = model.features[0].weight[model.features[0].weight == 0].shape[0]
    if zero_units:
        return zero_units / first_layer_units
    return 0.0


def train(
    dataset_name="cifar10",
    dataset_override=None,
    model_name="fully_connected_network",
    model_override=None,
    optimizer_name="sgd",
    bs=512,
    n_epochs=25,
    lr=0.1,
    min_lr=0.001,
    max_grad_norm=1.0,
    should_clip_grad_norm=False,
    test_every=-1,
    num_workers=12,
    reporter=None,
    return_accuracy=False,
    return_final_accuracy_only=True,
    skip_train_accuracy=False,
    log_metrics_inline=True,
    output_dir=None,
    dataset_overrides={},
    model_overrides={},
    opt_overrides={},
    device="cuda:0",
):
    if test_every == -1:
        test_every = n_epochs

    if t.cuda.is_available() and "cuda" in device:
        device = t.device(device)
    else:
        device = t.device("cpu")

    if output_dir is not None:
        writer = SummaryWriter(output_dir)
    else:
        writer = None

    if dataset_override is None:
        train_dataset, test_dataset = prepare_dataset(dataset_name, test=True, **dataset_overrides)

    else:
        train_dataset, test_dataset = dataset_override

    assert dataset_name == train_dataset.dataset_name, (dataset_name, test_dataset.dataset_name)
    assert dataset_name == test_dataset.dataset_name, (dataset_name, test_dataset.dataset_name)

    if dataset_name == "periphery":
        train_accuracies_v = []
        train_accuracies_a = []
        train_accuracies_s = []
        train_accuracies_o = []
        train_accuracies_h = []
        test_accuracies_v = []
        test_accuracies_a = []
        test_accuracies_s = []
        test_accuracies_o = []
        test_accuracies_h = []

    else:
        train_accuracies, test_accuracies = [], []

    train_loader = DataLoader(
        train_dataset,
        batch_size=bs,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=bs,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
    )

    if model_override is None:
        model = prepare_model(model_name, device, **model_overrides)
        model = t.compile(model)
    else:
        model = model_override.to(device)
        model_name = "fully_connected_network"

    optimizer = prepare_opt(model, optimizer_name, lr, **opt_overrides)

    num_training_steps = (len(train_dataset) * n_epochs) // bs
    print(f"Total Training Steps: {num_training_steps}")

    scheduler = prepare_scheduler("cosine", optimizer, num_training_steps, min_lr)
    criterion = (
        nn.BCEWithLogitsLoss()
        if dataset_name == "hippocampus"
        else nn.CrossEntropyLoss(
            weight=AUDITION_LOSS_WEIGHTS.to(device) if dataset_name == "audition" else None,
        )
    )

    if dataset_name == "periphery":
        bce_loss = nn.BCEWithLogitsLoss()

    metric = evaluate.load("accuracy")

    current_lr = f"{scheduler.get_last_lr()[0]:.4f}"
    model.train()

    for epoch in range(n_epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{n_epochs}")

        for inputs, labels in pbar:
            if dataset_name != "periphery":
                inputs, labels = (
                    inputs.to(device, non_blocking=True),
                    labels.to(device, non_blocking=True),
                )
            else:
                labels, labels_h = labels
                inputs, labels, labels_h = (
                    inputs.to(device, non_blocking=True),
                    labels.to(device, non_blocking=True),
                    labels_h.to(device, non_blocking=True),
                )

            if dataset_name == "hippocampus":
                labels = labels.view(inputs.shape[0], -1)

            with autocast(device_type="cuda", dtype=t.bfloat16):
                outputs = model(inputs)

                if dataset_name == "periphery":
                    (
                        outputs_v,
                        outputs_a,
                        outputs_s,
                        outputs_o,
                        outputs_h,
                    ) = outputs
                    labels_v = labels[:, 0]
                    labels_a = labels[:, 1]
                    labels_s = labels[:, 2]
                    labels_o = labels[:, 3]
                    labels_h = labels_h.view(labels_h.shape[0], -1)
                    loss_v = criterion(outputs_v, labels_v)
                    loss_a = criterion(outputs_a, labels_a)
                    loss_s = criterion(outputs_s, labels_s)
                    loss_o = criterion(outputs_o, labels_o)
                    loss_h = bce_loss(outputs_h, labels_h)
                    loss = loss_v + loss_a + loss_s + loss_o + loss_h

                else:
                    loss = criterion(outputs, labels)

                loss.backward()

            if should_clip_grad_norm:
                gn = clip_grad_norm(model.parameters(), max_grad_norm)
            else:
                gn = t.cat([
                    param.grad.detach().flatten()
                    for param in model.parameters()
                    if param.grad is not None
                ]).norm()

            optimizer.step()
            scheduler.step()
            current_lr = f"{scheduler.get_last_lr()[0]:.4f}"
            optimizer.zero_grad()

            pbar.set_postfix({"loss": loss.item(), "gn": gn.item(), "lr": current_lr})

            if writer is not None:
                writer.add_scalar("train/loss", loss.item(), global_step=scheduler.last_epoch)
                writer.add_scalar("train/gradient_norm", gn.item(), global_step=scheduler.last_epoch)
                writer.add_scalar("train/learning_rate", float(current_lr), global_step=scheduler.last_epoch)

        if (epoch + 1) % test_every == 0:
            if dataset_name != "periphery":
                if not skip_train_accuracy:
                    train_accuracy = test_accuracy(
                        model, train_loader, metric, device, multi_target=dataset_name == "hippocampus",
                    )
                    train_accuracies.append(train_accuracy)
                else:
                    train_accuracy = "N/A"

                accuracy = test_accuracy(model, test_loader, metric, device, multi_target=dataset_name == "hippocampus")
                test_accuracies.append(accuracy)

                sparsity = test_sparsity(model)

                if log_metrics_inline:
                    pbar.write(f"train accuracy: {train_accuracy}, test accuracy: {accuracy}, sparsity: {sparsity}")

                if reporter is not None:
                    reporter.report({"accuracy": accuracy})

                if writer is not None:
                    if not skip_train_accuracy:
                        writer.add_scalar("train/accuracy", train_accuracy, global_step=scheduler.last_epoch)

                    writer.add_scalar("test/accuracy", accuracy, global_step=scheduler.last_epoch)
                    writer.add_scalar("test/sparsity", sparsity, global_step=scheduler.last_epoch)

            else:
                n_modalities = 5

                if dataset_name == "periphery":
                    if not skip_train_accuracy:
                        (
                            train_accuracy_v,
                            train_accuracy_a,
                            train_accuracy_s,
                            train_accuracy_o,
                            train_accuracy_h,
                        ) = test_accuracy_periphery(
                            model, train_loader, metric, device, n_modalities,
                        )
                        train_accuracies_v.append(train_accuracy_v)
                        train_accuracies_a.append(train_accuracy_a)
                        train_accuracies_s.append(train_accuracy_s)
                        train_accuracies_o.append(train_accuracy_o)
                        train_accuracies_h.append(train_accuracy_h)

                    (
                        accuracy_v,
                        accuracy_a,
                        accuracy_s,
                        accuracy_o,
                        accuracy_h,
                    ) = test_accuracy_periphery(
                        model, test_loader, metric, device, n_modalities,
                    )
                    test_accuracies_v.append(accuracy_v)
                    test_accuracies_a.append(accuracy_a)
                    test_accuracies_s.append(accuracy_s)
                    test_accuracies_o.append(accuracy_o)
                    test_accuracies_h.append(accuracy_h)

                else:
                    raise RuntimeError(f"{dataset_name} is unrecognized")

                if log_metrics_inline:
                    if not skip_train_accuracy:
                        if dataset_name == "periphery":
                            pbar_msg_train = (
                                f"train accuracy (vision): {train_accuracy_v}\n"
                                f"train accuracy (audition): {train_accuracy_a}\n"
                                f"train accuracy (somatosensation): {train_accuracy_s}\n"
                                f"train accuracy (olfaction): {train_accuracy_o}\n"
                                f"train accuracy (hippocampus): {train_accuracy_h}\n"
                            )

                        pbar.write(pbar_msg_train)

                    if dataset_name == "periphery":
                        pbar_msg_test = (
                            f"test accuracy (vision): {accuracy_v}\n"
                            f"test accuracy (audition): {accuracy_a}\n"
                            f"test accuracy (somatosensation): {accuracy_s}\n"
                            f"test accuracy (olfaction): {accuracy_o}\n"
                            f"test accuracy (hippocampus): {accuracy_h}\n"
                        )

                        if writer is not None:
                            if not skip_train_accuracy:
                                writer.add_scalar("train/accuracy_vision", train_accuracy_v, global_step=scheduler.last_epoch)
                                writer.add_scalar("train/accuracy_audition", train_accuracy_a, global_step=scheduler.last_epoch)
                                writer.add_scalar("train/accuracy_somatosensation", train_accuracy_a, global_step=scheduler.last_epoch)
                                writer.add_scalar("train/accuracy_olfaction", train_accuracy_o, global_step=scheduler.last_epoch)
                                writer.add_scalar("train/accuracy_hippocampal", train_accuracy_h, global_step=scheduler.last_epoch)

                            writer.add_scalar("test/accuracy_vision", accuracy_v, global_step=scheduler.last_epoch)
                            writer.add_scalar("test/accuracy_audition", accuracy_a, global_step=scheduler.last_epoch)
                            writer.add_scalar("test/accuracy_somatosensation", accuracy_s, global_step=scheduler.last_epoch)
                            writer.add_scalar("test/accuracy_olfaction", accuracy_o, global_step=scheduler.last_epoch)
                            writer.add_scalar("test/accuracy_hippocampal", accuracy_h, global_step=scheduler.last_epoch)

                    pbar.write(pbar_msg_test)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        output_dir = Path(output_dir)
        output_path = output_dir / "checkpoint_latest.pth"
        t.save(
            {
                "model": model.state_dict(),
                "opt": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            output_path,
        )
        print(f"Checkpoint saved at {output_path}...")

    if return_accuracy:
        if return_final_accuracy_only:
            if dataset_name == "periphery":
                if not skip_train_accuracy:
                    train_accuracies_v = train_accuracies_v[-1]
                    train_accuracies_a = train_accuracies_a[-1]
                    train_accuracies_s = train_accuracies_s[-1]
                    train_accuracies_o = train_accuracies_o[-1]
                    train_accuracies_h = train_accuracies_h[-1]

                test_accuracies_v = train_accuracies_v[-1]
                test_accuracies_a = train_accuracies_a[-1]
                test_accuracies_s = train_accuracies_s[-1]
                test_accuracies_o = train_accuracies_o[-1]
                test_accuracies_h = train_accuracies_h[-1]

            else:
                if not skip_train_accuracy:
                    train_accuracies = train_accuracies[-1]

                test_accuracies = test_accuracies[-1]

        if dataset_name == "periphery":
            if skip_train_accuracy:
                return (
                    test_accuracies_v,
                    test_accuracies_a,
                    test_accuracies_s,
                    test_accuracies_o,
                    test_accuracies_h,
                )

            return (
                (train_accuracies_v, test_accuracies_v),
                (train_accuracies_a, test_accuracies_a),
                (train_accuracies_s, test_accuracies_s),
                (train_accuracies_o, test_accuracies_o),
                (train_accuracies_h, test_accuracies_h),
            )

        else:
            if skip_train_accuracy:
                return test_accuracies

            return (train_accuracies, test_accuracies)


if __name__ == "__main__":
    fire.Fire(train)
