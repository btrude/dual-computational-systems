import os
from pathlib import Path
from random import choice

import chcochleagram
import fire
import torch
import torchaudio
from tqdm import tqdm

from dual_computational_systems.util import set_seed
from dual_computational_systems.util.constants import AUDITION_DATASET_PATH


class Cochleagram:
    def __init__(self, sr=44100, signal_size=88200, device="cpu"):
        self.signal_size = signal_size
        pad_factor = 1.0
        use_rfft = True

        half_cos_filter_kwargs = {
            "n": 203,
            "low_lim": 30,
            "high_lim": 7860,
            "sample_factor": 4,
            "full_filter": False,
        }
        coch_filter_kwargs = {
            "use_rfft": use_rfft,
            "pad_factor": pad_factor,
            "filter_kwargs": half_cos_filter_kwargs,
        }
        filters = chcochleagram.cochlear_filters.ERBCosFilters(
            self.signal_size,
            sr,
            **coch_filter_kwargs,
        )
        envelope_extraction = chcochleagram.envelope_extraction.HilbertEnvelopeExtraction(
            self.signal_size,
            sr,
            use_rfft,
            pad_factor,
        )

        downsampling_kwargs = {"window_size": 1001}
        downsampling_op = chcochleagram.downsampling.SincWithKaiserWindow(
            sr,
            150,
            **downsampling_kwargs,
        )
        compression_kwargs = {
            "power": 0.3,
            "offset": 1e-8,
            "scale": 1,
            "clip_value": 100,
        }
        compression = chcochleagram.compression.ClippedGradPowerCompression(**compression_kwargs)

        self.transform = chcochleagram.cochleagram.Cochleagram(
            filters,
            envelope_extraction,
            downsampling_op,
            compression=compression,
        ).to(device)
        self.device = device

    def __call__(self, signal: torch.Tensor):
        if signal.shape[-1] < self.signal_size:
            signal = torch.nn.functional.pad(signal, (0, self.signal_size - signal.shape[-1]))
        else:
            signal = signal[..., :self.signal_size]

        if signal.device != self.device:
            signal = signal.to(self.device)

        return self.transform(signal).flip(dims=(1,))


def convert(
    audio_file,
    transform,
    sr_target=44100,
    signal_size=88200,
    save_tensor=True,
    out_size=(32, 32),
    device="cpu",
):
    if isinstance(audio_file, torch.Tensor):
        wav = audio_file
        sr = sr_target
    else:
        wav, sr = torchaudio.load(audio_file)
        assert sr_target == sr, (sr_target, sr)

    if wav.shape[-1] <= signal_size:
        inputs = [wav]
    else:
        inputs = list(wav.split(signal_size, dim=-1))

    if inputs[-1].shape[-1] <= signal_size:
        inputs[-1] = torch.nn.functional.pad(inputs[-1], (1, signal_size - inputs[-1].shape[-1] - 1,))

    x = choice(inputs)
    assert x.shape[-1] == signal_size, x.shape[-1]
    y = transform(x.to(device))
    y = torch.nn.functional.adaptive_max_pool2d(y.unsqueeze(0), out_size).squeeze(0)

    if save_tensor:
        output_filename = Path(str(audio_file).replace("fsd50k", "fsd50k_cochleagram").replace(".wav", ".pt"))
        os.makedirs(output_filename.parent, exist_ok=True)
        torch.save(y, output_filename)

    return y


def main(sr_target=44100, signal_size=88200, device="cpu"):
    torch.set_grad_enabled(False)
    dataset_dir = Path(AUDITION_DATASET_PATH.replace("_cochleagram", ""))
    transform = Cochleagram(signal_size=signal_size, sr=sr_target, device=device)

    for subset in ["train", "eval"]:
        audio_files = [p for p in Path(f"{dataset_dir / subset}").rglob("*.wav")]
        pbar = tqdm(audio_files, desc=subset)

        for audio_file in pbar:
            convert(audio_file, transform, sr_target=sr_target, signal_size=signal_size, device=device)


if __name__ == "__main__":
    set_seed(0)
    fire.Fire(main)
