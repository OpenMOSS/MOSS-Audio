import importlib.util
import os
import re
import sys
import types
from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
import torchaudio
from transformers import AutoTokenizer, BatchEncoding


@dataclass
class MelConfig:
    mel_sr: int = 16000
    mel_dim: int = 128
    mel_n_fft: int = 400
    mel_hop_length: int = 160
    mel_dtype: torch.dtype = torch.bfloat16
    use_whisper_feature_extractor: bool = True


def load_chat_template(template_path: str, mossflux_path: str = None) -> List:
    if mossflux_path is None:
        template_dir = os.path.dirname(os.path.abspath(template_path))
        current = template_dir
        while current and os.path.basename(current) != "mossLite":
            parent = os.path.dirname(current)
            if parent == current:
                break
            current = parent
        if os.path.basename(current) == "mossLite":
            mossflux_path = os.path.join(current, "mossflux")

    if mossflux_path and mossflux_path not in sys.path:
        sys.path.insert(0, mossflux_path)

    spec = importlib.util.spec_from_file_location("chat_template_module", template_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["chat_template_module"] = module
    spec.loader.exec_module(module)
    return module.chat_template


class MossAudioProcessor:
    _AUDIO_SPAN_RE = re.compile(r"<\|audio_bos\|>(?:<\|AUDIO\|>)+<\|audio_eos\|>")
    _auto_class = None

    @classmethod
    def register_for_auto_class(cls, auto_class="AutoProcessor"):
        if not isinstance(auto_class, str):
            auto_class = auto_class.__name__
        cls._auto_class = auto_class

    def __init__(
        self,
        tokenizer,
        *,
        mel_config: Optional[MelConfig] = None,
        template_path: Optional[str] = None,
        enable_time_marker: bool = True,
        audio_token_id: int = 151654,
        audio_start_id: int = 151669,
        audio_end_id: int = 151670,
    ):
        self._base_tokenizer = tokenizer
        self.tokenizer = tokenizer
        self.audio_token_id = int(audio_token_id)
        self.audio_start_id = int(audio_start_id)
        self.audio_end_id = int(audio_end_id)
        self.chat_template = (
            None if template_path is None else load_chat_template(template_path)
        )
        self.custom_texts = {}
        self.enable_time_marker = bool(enable_time_marker)
        self.config = mel_config or MelConfig()
        self._whisper_feature_extractor = None

        alias_map = {
            "<|AUDIO|>": self.audio_token_id,
            "<|audio_bos|>": self.audio_start_id,
            "<|audio_eos|>": self.audio_end_id,
        }
        orig_convert_tokens_to_ids = self.tokenizer.convert_tokens_to_ids

        def _patched_convert_tokens_to_ids(tokenizer_self, tokens):
            if isinstance(tokens, (list, tuple)):
                converted = [
                    _patched_convert_tokens_to_ids(tokenizer_self, token)
                    for token in tokens
                ]
                return converted if isinstance(tokens, list) else tuple(converted)
            if isinstance(tokens, str) and tokens in alias_map:
                return alias_map[tokens]
            return orig_convert_tokens_to_ids(tokens)

        self.tokenizer.convert_tokens_to_ids = types.MethodType(
            _patched_convert_tokens_to_ids, self.tokenizer
        )

        self._digit_token_ids = {
            "0": 15,
            "1": 16,
            "2": 17,
            "3": 18,
            "4": 19,
            "5": 20,
            "6": 21,
            "7": 22,
            "8": 23,
            "9": 24,
        }
        self.audio_tokens_per_second = 12.5
        self.time_marker_every_seconds = 2
        self.time_marker_every_audio_tokens = int(
            self.audio_tokens_per_second * self.time_marker_every_seconds
        )
        self.model_input_names = [
            "input_ids",
            "attention_mask",
            "audio_data",
            "audio_data_seqlens",
        ]

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        tokenizer_kwargs = {}
        for key in ["cache_dir", "revision", "token", "local_files_only"]:
            if key in kwargs:
                tokenizer_kwargs[key] = kwargs[key]

        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            use_fast=False,
            **tokenizer_kwargs,
        )

        mel_config = kwargs.pop("mel_config", None)
        template_path = kwargs.pop("template_path", None)
        enable_time_marker = kwargs.pop("enable_time_marker", False)
        audio_token_id = kwargs.pop("audio_token_id", 151654)
        audio_start_id = kwargs.pop("audio_start_id", 151669)
        audio_end_id = kwargs.pop("audio_end_id", 151670)

        return cls(
            tokenizer,
            mel_config=mel_config,
            template_path=template_path,
            enable_time_marker=enable_time_marker,
            audio_token_id=audio_token_id,
            audio_start_id=audio_start_id,
            audio_end_id=audio_end_id,
        )

    def load_template(self, template_path: str):
        self.chat_template = load_chat_template(template_path)
        return self

    def set_custom_text(self, key: str, text: str):
        self.custom_texts[key] = text
        return self

    def clear_custom_text(self, key: Optional[str] = None):
        if key is None:
            self.custom_texts.clear()
        else:
            self.custom_texts.pop(key, None)
        return self

    def _template_requires_audio(self) -> bool:
        if self.chat_template is None:
            return False
        for segment in self.chat_template:
            if segment.type in {"audio_contiguous", "audio_token"}:
                return True
        return False

    @staticmethod
    def _conv3_downsample_len(raw_mel_len: int) -> int:
        def conv_out_len(length: int) -> int:
            return (length - 1) // 2 + 1

        length1 = conv_out_len(int(raw_mel_len))
        length2 = conv_out_len(length1)
        length3 = conv_out_len(length2)
        return int(length3)

    def _get_whisper_feature_extractor(self):
        if self._whisper_feature_extractor is not None:
            return self._whisper_feature_extractor

        from transformers.models.whisper.feature_extraction_whisper import (
            WhisperFeatureExtractor,
        )

        self._whisper_feature_extractor = WhisperFeatureExtractor(
            feature_size=int(self.config.mel_dim),
            sampling_rate=int(self.config.mel_sr),
            hop_length=int(self.config.mel_hop_length),
            n_fft=int(self.config.mel_n_fft),
        )
        return self._whisper_feature_extractor

    def _extract_mel(self, audio: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(audio, np.ndarray):
            wav = torch.from_numpy(audio)
        else:
            wav = audio
        wav = wav.to(dtype=torch.float32)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)

        if bool(getattr(self.config, "use_whisper_feature_extractor", False)):
            fe = self._get_whisper_feature_extractor()
            wav_np = wav.detach().to("cpu", torch.float32).contiguous().numpy()
            if wav_np.ndim == 2:
                wav_np = wav_np[0]
            feats = fe._np_extract_fbank_features(wav_np[None, ...], device="cpu")
            mel = torch.from_numpy(feats[0])

        return mel.to(dtype=self.config.mel_dtype)

    def _get_time_marker_token_ids(self, second: int) -> List[int]:
        return [self._digit_token_ids[digit] for digit in str(second)]

    def _build_audio_tokens_with_time_markers(self, audio_seq_len: int) -> List[int]:
        total_duration_seconds = audio_seq_len / self.audio_tokens_per_second
        num_full_seconds = int(total_duration_seconds)

        token_ids: List[int] = []
        audio_tokens_consumed = 0
        for second in range(
            self.time_marker_every_seconds,
            num_full_seconds + 1,
            self.time_marker_every_seconds,
        ):
            marker_pos = (
                second // self.time_marker_every_seconds
            ) * self.time_marker_every_audio_tokens
            audio_segment_len = marker_pos - audio_tokens_consumed
            if audio_segment_len > 0:
                token_ids.extend([self.audio_token_id] * audio_segment_len)
                audio_tokens_consumed += audio_segment_len
            token_ids.extend(self._get_time_marker_token_ids(second))

        remaining = audio_seq_len - audio_tokens_consumed
        if remaining > 0:
            token_ids.extend([self.audio_token_id] * remaining)
        return token_ids

    def _build_audio_placeholder_ids(self, num_audio_tokens: int) -> List[int]:
        if self.enable_time_marker:
            return self._build_audio_tokens_with_time_markers(num_audio_tokens)
        return [self.audio_token_id] * num_audio_tokens

    def _build_input_from_template(
        self, num_audio_tokens: int, include_answer: bool = False
    ) -> List[int]:
        if self.chat_template is None:
            raise ValueError("Chat template not loaded.")

        input_ids: List[int] = []
        for segment in self.chat_template:
            seg_type = segment.type
            if seg_type == "constant_text_token":
                input_ids.extend(segment.text_ids.tolist())
            elif seg_type in {"audio_contiguous", "audio_token"}:
                input_ids.extend(self._build_audio_placeholder_ids(num_audio_tokens))
            elif seg_type == "text_token":
                text_token_key = segment.text_token_key
                if "answer" in text_token_key.lower() and not include_answer:
                    break
                if text_token_key not in self.custom_texts:
                    break
                text_ids = self._base_tokenizer.encode(
                    self.custom_texts[text_token_key], add_special_tokens=False
                )
                input_ids.extend(text_ids)

        return input_ids

    def _build_default_prompt(self, text: str, has_audio: bool) -> str:
        if has_audio:
            return (
                "<|im_start|>system\n"
                "You are a helpful assistant.<|im_end|>\n"
                "<|im_start|>user\n"
                "<|audio_bos|><|AUDIO|><|audio_eos|>\n"
                f"{text}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
        return (
            "<|im_start|>system\n"
            "You are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n"
            f"{text}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def _build_input_from_prompt(self, prompt: str, token_lens: List[int]) -> List[int]:
        spans = list(self._AUDIO_SPAN_RE.finditer(prompt))
        if len(spans) != len(token_lens):
            raise ValueError(
                f"Audio placeholder count mismatch: found {len(spans)} spans in text, "
                f"but got {len(token_lens)} audio inputs."
            )

        input_ids: List[int] = []
        cursor = 0
        for index, match in enumerate(spans):
            prefix = prompt[cursor : match.start()]
            if prefix:
                input_ids.extend(
                    self._base_tokenizer.encode(prefix, add_special_tokens=False)
                )

            input_ids.append(self.audio_start_id)
            input_ids.extend(self._build_audio_placeholder_ids(int(token_lens[index])))
            input_ids.append(self.audio_end_id)
            cursor = match.end()

        suffix = prompt[cursor:]
        if suffix:
            input_ids.extend(
                self._base_tokenizer.encode(suffix, add_special_tokens=False)
            )
        return input_ids

    def __call__(
        self,
        *,
        text: Union[str, Sequence[str], None] = None,
        audios: Optional[Sequence[Union[np.ndarray, torch.Tensor]]] = None,
        audio: Optional[Sequence[Union[np.ndarray, torch.Tensor]]] = None,
        return_tensors: str = "pt",
        **kwargs,
    ):
        if isinstance(text, (list, tuple)):
            if len(text) != 1:
                raise ValueError(f"Expected text batch size 1, got {len(text)}")
            prompt_text = text[0]
        else:
            prompt_text = text

        audio_list = audios if audios is not None else audio
        audio_list = [] if audio_list is None else list(audio_list)

        mels: List[torch.Tensor] = []
        raw_lengths: List[int] = []
        token_lens: List[int] = []
        for one_audio in audio_list:
            mel = self._extract_mel(one_audio)
            raw_len = int(mel.shape[-1])
            mels.append(mel)
            raw_lengths.append(raw_len)
            token_lens.append(self._conv3_downsample_len(raw_len))

        if mels:
            max_length = max(raw_lengths)
            audio_batch = torch.zeros(
                (len(mels), self.config.mel_dim, max_length),
                dtype=self.config.mel_dtype,
            )
            for index, mel in enumerate(mels):
                audio_batch[index, :, : mel.shape[-1]] = mel
            seqlens_tensor = torch.tensor(raw_lengths, dtype=torch.long)
        else:
            audio_batch = None
            seqlens_tensor = None

        if prompt_text is not None:
            if self._AUDIO_SPAN_RE.search(prompt_text) is None and audio_list:
                prompt_text = self._build_default_prompt(prompt_text, has_audio=True)
            elif self._AUDIO_SPAN_RE.search(prompt_text) is None and not audio_list:
                prompt_text = self._build_default_prompt(prompt_text, has_audio=False)
            input_ids_list = self._build_input_from_prompt(prompt_text, token_lens)
        elif self.chat_template is not None:
            input_ids_list = self._build_input_from_template(
                token_lens[0] if token_lens else 0
            )
        else:
            raise ValueError(
                "Either provide text or load a chat_template before calling the processor."
            )

        input_ids_tensor = torch.tensor([input_ids_list], dtype=torch.long)
        attention_mask_tensor = torch.ones_like(input_ids_tensor)

        data = {
            "input_ids": input_ids_tensor,
            "attention_mask": attention_mask_tensor,
        }
        if audio_batch is not None:
            data["audio_data"] = audio_batch
            data["audio_data_seqlens"] = seqlens_tensor
        return BatchEncoding(data=data, tensor_type=return_tensors)

    def batch_decode(self, *args, **kwargs):
        return self._base_tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self._base_tokenizer.decode(*args, **kwargs)


__all__ = ["MelConfig", "MossAudioProcessor"]
