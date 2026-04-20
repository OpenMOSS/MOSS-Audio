"""
MOSS-Audio SFT fine-tuning script.

Minimal single-file trainer supporting LoRA and full-parameter fine-tuning.
Usage:
    # LoRA
    accelerate launch finetune.py \
        --model_dir ./weights/moss-audio \
        --data_path train.jsonl \
        --output_dir ./output \
        --use_lora

    # Full-parameter
    accelerate launch finetune.py \
        --model_dir ./weights/moss-audio \
        --data_path train.jsonl \
        --output_dir ./output

Data format (JSONL, one per line):
    {"conversation": [
        {"role": "user", "message_type": "audio", "content": "/path/to/audio.wav"},
        {"role": "user", "message_type": "text",  "content": "Transcribe the audio."},
        {"role": "assistant", "message_type": "text", "content": "Hello world."}
    ]}
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import librosa
import numpy as np
import torch
import transformers
from transformers import Trainer
from transformers.models.whisper.feature_extraction_whisper import (
    WhisperFeatureExtractor,
)

from src.configuration_moss_audio import MossAudioConfig
from src.modeling_moss_audio import MossAudioModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Qwen3 special token IDs
AUDIO_TOKEN_ID = 151654       # <|AUDIO|>
AUDIO_START_ID = 151669       # <|audio_bos|>
AUDIO_END_ID = 151670         # <|audio_eos|>
IM_START_ID = 151644          # <|im_start|>
IM_END_ID = 151645            # <|im_end|>

# Qwen3 chat format token sequences (pre-tokenized for speed)
SYSTEM_IDS = [IM_START_ID, 8948, 198, 2610, 525, 264, 10950, 17847, 13, IM_END_ID, 198]
USER_AUDIO_START_IDS = [IM_START_ID, 872, 198, AUDIO_START_ID]
AUDIO_END_NL_IDS = [AUDIO_END_ID, 198]
TURN_BOUNDARY_IDS = [IM_END_ID, 198, IM_START_ID, 77091, 198]  # <|im_end|>\n<|im_start|>assistant\n


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

@dataclass
class ModelArguments:
    model_dir: str = field(metadata={"help": "Path to MOSS-Audio model directory."})
    attn_implementation: str = field(default="flash_attention_2")


@dataclass
class DataArguments:
    data_path: str = field(metadata={"help": "Path to training JSONL file."})
    eval_data_path: Optional[str] = field(default=None)
    max_len: int = field(default=8192)
    prompt_default: str = field(default="")


@dataclass
class FinetuneArguments(transformers.TrainingArguments):
    use_lora: bool = field(default=False)
    lora_rank: int = field(default=64)
    lora_alpha: int = field(default=2)
    lora_on_audio_encoder: bool = field(default=False)
    optim: str = field(default="adamw_torch_fused")


# ---------------------------------------------------------------------------
# Mel extraction
# ---------------------------------------------------------------------------

_whisper_fe: Optional[WhisperFeatureExtractor] = None


def extract_mel(audio_path: str, sr: int = 16000) -> torch.Tensor:
    global _whisper_fe
    if _whisper_fe is None:
        _whisper_fe = WhisperFeatureExtractor(
            feature_size=128, sampling_rate=sr, hop_length=160, n_fft=400,
        )
    wav, _ = librosa.load(audio_path, sr=sr)
    feats = _whisper_fe._np_extract_fbank_features(wav[None, ...], device="cpu")
    return torch.from_numpy(feats[0]).to(torch.bfloat16)


def _compute_audio_tokens(mel_len: int) -> int:
    """Three stride-2 convolutions → downsampled token count."""
    for _ in range(3):
        mel_len = (mel_len - 1) // 2 + 1
    return mel_len


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MossAudioDataset(torch.utils.data.Dataset):
    def __init__(self, data: List[dict], tokenizer, max_len: int, prompt_default: str = ""):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.prompt_default = prompt_default

    def __len__(self):
        return len(self.data)

    def _parse(self, conversation):
        audio_path = prompt = answer = None
        prompt_parts, answer_parts = [], []
        for msg in conversation:
            mt = msg.get("message_type")
            if mt == "audio" and audio_path is None:
                audio_path = msg["content"]
            elif mt == "text":
                if msg["role"] == "user":
                    prompt_parts.append(msg.get("content", ""))
                elif msg["role"] == "assistant":
                    answer_parts.append(msg.get("content", ""))
        prompt = "\n".join(prompt_parts).strip() or self.prompt_default
        answer = "\n".join(answer_parts).strip()
        return audio_path, prompt, answer

    def __getitem__(self, idx):
        obj = self.data[idx]
        audio_path, prompt, answer = self._parse(obj["conversation"])
        if audio_path is None:
            raise ValueError(f"No audio in sample {idx}")

        mel = extract_mel(audio_path)
        n_tokens = _compute_audio_tokens(mel.shape[-1])

        # Build Qwen3 chat-format sequence
        audio_ids = [AUDIO_TOKEN_ID] * n_tokens
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False) if prompt else []
        answer_ids = self.tokenizer.encode(answer, add_special_tokens=False)

        input_ids = (
            SYSTEM_IDS
            + USER_AUDIO_START_IDS
            + audio_ids
            + AUDIO_END_NL_IDS
            + prompt_ids
            + TURN_BOUNDARY_IDS
            + answer_ids
            + [IM_END_ID]
        )
        labels = (
            [-100] * (len(SYSTEM_IDS) + len(USER_AUDIO_START_IDS) + len(audio_ids)
                      + len(AUDIO_END_NL_IDS) + len(prompt_ids) + len(TURN_BOUNDARY_IDS))
            + answer_ids + [IM_END_ID]
        )
        audio_mask = (
            [False] * len(SYSTEM_IDS)
            + [False] * len(USER_AUDIO_START_IDS)
            + [tid == AUDIO_TOKEN_ID for tid in audio_ids]
            + [False] * (len(AUDIO_END_NL_IDS) + len(prompt_ids) + len(TURN_BOUNDARY_IDS)
                         + len(answer_ids) + 1)
        )

        # Truncate & pad to max_len
        input_ids = input_ids[: self.max_len]
        labels = labels[: self.max_len]
        audio_mask = audio_mask[: self.max_len]

        # If audio tokens were truncated, trim mel to match
        actual_audio_tokens = sum(1 for m in audio_mask if m)
        if actual_audio_tokens < n_tokens:
            keep_frames = actual_audio_tokens * 8  # 3 conv layers stride 2 → ×8
            mel = mel[:, :keep_frames]

        seq_len = len(input_ids)
        pad_len = self.max_len - seq_len

        return {
            "input_ids": torch.tensor(input_ids + [self.tokenizer.pad_token_id] * pad_len, dtype=torch.long),
            "labels": torch.tensor(labels + [-100] * pad_len, dtype=torch.long),
            "attention_mask": torch.tensor([1] * seq_len + [0] * pad_len, dtype=torch.long),
            "audio_data": mel,                                    # [n_mels, T]
            "audio_data_seqlens": torch.tensor(mel.shape[-1], dtype=torch.long),
            "audio_input_mask": torch.tensor(audio_mask + [False] * pad_len, dtype=torch.bool),
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, FinetuneArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # ---- Model ----
    config = MossAudioConfig.from_pretrained(model_args.model_dir)
    for cfg in [config, getattr(config, "language_config", None)]:
        if cfg is not None:
            cfg._attn_implementation = model_args.attn_implementation

    model = MossAudioModel.from_pretrained(
        model_args.model_dir, config=config, dtype=torch.bfloat16,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_dir, trust_remote_code=True)

    # ---- LoRA ----
    if training_args.use_lora:
        from peft import LoraConfig, get_peft_model

        target = (
            r"language_model\.layers\.\d+\.(self_attn|mlp)\."
            r"(q_proj|k_proj|v_proj|o_proj|up_proj|gate_proj|down_proj)"
        )
        if training_args.lora_on_audio_encoder:
            target += r"|audio_encoder\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj)"

        model = get_peft_model(
            model,
            LoraConfig(
                r=training_args.lora_rank,
                lora_alpha=training_args.lora_alpha,
                target_modules=target,
                lora_dropout=0.0,
                task_type="CAUSAL_LM",
            ),
        )
        model.print_trainable_parameters()

    # ---- Data ----
    def load_jsonl(path):
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]

    train_data = load_jsonl(data_args.data_path)
    train_dataset = MossAudioDataset(train_data, tokenizer, data_args.max_len, data_args.prompt_default)
    logger.info("Train samples: %d", len(train_dataset))

    eval_dataset = None
    if data_args.eval_data_path:
        eval_dataset = MossAudioDataset(
            load_jsonl(data_args.eval_data_path), tokenizer, data_args.max_len, data_args.prompt_default,
        )
        logger.info("Eval samples: %d", len(eval_dataset))

    # ---- Train ----
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()
    trainer.save_state()
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    train()
