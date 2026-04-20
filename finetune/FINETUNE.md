# MOSS-Audio Fine-tuning Guide

## Quick Start

```bash
# LoRA fine-tuning (recommended)
accelerate launch finetune.py \
    --model_dir ./weights/moss-audio \
    --data_path train.jsonl \
    --output_dir ./output/lora \
    --use_lora \
    --bf16

# Full-parameter fine-tuning
accelerate launch finetune.py \
    --model_dir ./weights/moss-audio \
    --data_path train.jsonl \
    --output_dir ./output/full \
    --bf16
```

## Data Format

JSONL file, one sample per line. Each sample contains a `conversation` list with three messages:

```json
{"conversation": [
    {"role": "user",      "message_type": "audio", "content": "/path/to/audio.wav"},
    {"role": "user",      "message_type": "text",  "content": "Transcribe the audio."},
    {"role": "assistant", "message_type": "text",  "content": "Hello world."}
]}
```

- Audio: WAV format, automatically resampled to 16kHz.
- Text prompt (user text message) is optional. If omitted, `--prompt_default` is used.
- Only the assistant response is supervised (contributes to loss). All other parts (system, user, audio) are masked with `label=-100`.

## Arguments

### Model

| Argument | Default | Description |
|---|---|---|
| `--model_dir` | (required) | Path to MOSS-Audio checkpoint directory |
| `--attn_implementation` | `flash_attention_2` | Attention backend. Options: `flash_attention_2`, `eager` |

### Data

| Argument | Default | Description |
|---|---|---|
| `--data_path` | (required) | Path to training JSONL |
| `--eval_data_path` | `None` | Path to evaluation JSONL |
| `--max_len` | `8192` | Maximum sequence length (tokens). Sequences are padded/truncated to this length |
| `--prompt_default` | `""` | Default text prompt when none is provided in the data |

### Training

All [HuggingFace TrainingArguments](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments) are supported, plus:

| Argument | Default | Description |
|---|---|---|
| `--use_lora` | `False` | Enable LoRA fine-tuning |
| `--lora_rank` | `64` | LoRA rank |
| `--lora_alpha` | `2` | LoRA alpha |
| `--lora_on_audio_encoder` | `False` | Also apply LoRA to audio encoder (q/k/v projections) |

## Examples

### Single-GPU LoRA

```bash
python finetune.py \
    --model_dir ./weights/moss-audio \
    --data_path train.jsonl \
    --output_dir ./output/lora \
    --use_lora \
    --lora_rank 32 \
    --bf16 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 3 \
    --learning_rate 2e-4 \
    --logging_steps 10 \
    --save_steps 500
```

### Multi-GPU with DeepSpeed

```bash
accelerate launch --num_processes 8 finetune.py \
    --model_dir ./weights/moss-audio \
    --data_path train.jsonl \
    --output_dir ./output/full \
    --bf16 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 3 \
    --learning_rate 5e-5 \
    --deepspeed ds_config.json
```

### LoRA with Audio Encoder

```bash
accelerate launch finetune.py \
    --model_dir ./weights/moss-audio \
    --data_path train.jsonl \
    --output_dir ./output/lora-aut \
    --use_lora \
    --lora_on_audio_encoder \
    --bf16
```

## Input Sequence Format

The script constructs the following Qwen3 chat-format sequence:

```
<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n
<|im_start|>user\n<|audio_bos|>[audio tokens]<|audio_eos|>\n
[text prompt]<|im_end|>\n
<|im_start|>assistant\n
[answer tokens]<|im_end|>
```

Loss is only computed on `[answer tokens]<|im_end|>`. Everything else is masked.

## Notes

- `per_device_train_batch_size` must be **1** when using the default PyTorch data collator, since audio mel spectrograms have variable lengths across samples.
- For long audio (>30s), ensure `max_len` is large enough. The script automatically truncates both the token sequence and mel spectrogram if they exceed `max_len`.
- On GPUs without Flash Attention support (e.g., V100), use `--attn_implementation eager`.

## Dependencies

```
torch
transformers
accelerate
librosa
soundfile
peft  # only needed for LoRA
```
