"""HuggingFace inference wrapper for MOSS-Audio."""

from __future__ import annotations

import os
from typing import Optional

import torch

from src.audio_io import load_audio
from src.modeling_moss_audio import MossAudioModel
from src.processing_moss_audio import MossAudioProcessor

DEFAULT_MODEL_ID = "OpenMOSS-Team/MOSS-Audio"


def read_env_model_id() -> str:
    return os.environ.get("MOSS_AUDIO_MODEL_ID", DEFAULT_MODEL_ID)


def resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class MossAudioHFInference:
    """Thin wrapper that loads model + processor and exposes a single
    ``generate`` method for both audio-grounded and text-only queries."""

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_MODEL_ID,
        device: str = "cuda:0",
        dtype: str = "auto",
        enable_time_marker: bool = True,
    ):
        self.device = device
        self.model = MossAudioModel.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            dtype=dtype,
            device_map=device,
        )
        self.model.eval()
        self.processor = MossAudioProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            enable_time_marker=enable_time_marker,
        )

    @torch.no_grad()
    def generate(
        self,
        question: str,
        audio_path: Optional[str] = None,
        max_new_tokens: int = 1024,
        num_beams: int = 1,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
    ) -> str:
        if audio_path is not None:
            raw_audio = load_audio(audio_path, sample_rate=self.processor.config.mel_sr)
            inputs = self.processor(text=question, audios=[raw_audio], return_tensors="pt")
        else:
            inputs = self.processor(text=question, return_tensors="pt")

        inputs = inputs.to(self.model.device)
        if inputs.get("audio_data") is not None:
            inputs["audio_data"] = inputs["audio_data"].to(self.model.dtype)

        audio_input_mask = inputs["input_ids"] == self.processor.audio_token_id
        inputs["audio_input_mask"] = audio_input_mask

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            use_cache=True,
        )
        if do_sample:
            gen_kwargs.update(
                do_sample=True, temperature=temperature, top_p=top_p, top_k=top_k
            )
        else:
            gen_kwargs["do_sample"] = False

        generated_ids = self.model.generate(**inputs, **gen_kwargs)

        input_len = inputs["input_ids"].shape[1]
        return self.processor.decode(
            generated_ids[0, input_len:], skip_special_tokens=True
        )


__all__ = ["MossAudioHFInference", "read_env_model_id", "resolve_device"]
