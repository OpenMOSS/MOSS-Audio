import torch

from src.audio_io import load_audio
from src.modeling_moss_audio import MossAudioModel
from src.processing_moss_audio import MossAudioProcessor

MODEL_PATH = "/inspire/qb-ilm/project/embodied-multimodality/public/yangchen/MOSS-Audio/weights/MOSS-Audio-4B-Thinking"
AUDIO_PATH = "test_kr.mp3"
TEMPERATURE = 1.0
TOP_P = 1.0
TOP_K = 50


def main():
    model = MossAudioModel.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="cuda:0",
    )
    model.eval()

    processor = MossAudioProcessor.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        enable_time_marker=True,
    )

    raw_audio = load_audio(AUDIO_PATH, sample_rate=processor.config.mel_sr)

    prompt = "Describe this audio."
    inputs = processor(text=prompt, audios=[raw_audio], return_tensors="pt")

    inputs = inputs.to(model.device)
    if inputs.get("audio_data") is not None:
        inputs["audio_data"] = inputs["audio_data"].to(model.dtype)

    audio_input_mask = inputs["input_ids"] == processor.audio_token_id
    inputs["audio_input_mask"] = audio_input_mask

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=True,
            num_beams=1,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            use_cache=True,
        )

    input_len = inputs["input_ids"].shape[1]
    transcription = processor.decode(
        generated_ids[0, input_len:], skip_special_tokens=True
    )
    print(transcription)


if __name__ == "__main__":
    main()
