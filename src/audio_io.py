from __future__ import annotations

import torchaudio


def load_audio(path: str, sample_rate: int):
    waveform, original_sample_rate = torchaudio.load(path)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if original_sample_rate != sample_rate:
        waveform = torchaudio.functional.resample(
            waveform, orig_freq=original_sample_rate, new_freq=sample_rate
        )
    return waveform.squeeze(0).cpu().numpy()
