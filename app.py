from __future__ import annotations

import os
import subprocess
import tempfile
import time
from functools import lru_cache
from pathlib import Path

import gradio as gr

from src.hf_inference import MossAudioHFInference, read_env_model_id, resolve_device

TITLE = "MOSS-Audio Demo"

DEFAULT_QUESTION = "Describe this audio."
DEFAULT_MAX_NEW_TOKENS = 1024
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 1.0
DEFAULT_TOP_K = 50
VIDEO_EXTENSIONS = {".mp4"}


@lru_cache(maxsize=2)
def get_inference(model_name_or_path: str, device: str) -> MossAudioHFInference:
    return MossAudioHFInference(
        model_name_or_path=model_name_or_path,
        device=device,
        dtype="auto",
        enable_time_marker=True,
    )


def format_status(model_name_or_path: str, device: str, elapsed_seconds: float) -> str:
    return (
        f"Model: `{model_name_or_path}`  \n"
        f"Device: `{device}`  \n"
        f"Elapsed: `{elapsed_seconds:.2f}s`"
    )


def convert_media_to_mp3(media_path: str, output_path: str) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-i",
        media_path,
        "-vn",
        "-acodec",
        "libmp3lame",
        output_path,
    ]
    try:
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise gr.Error(
            f"Failed to extract audio from the uploaded media. Please make sure the mp4 file is valid and decodable.\n{exc.stderr}"
        ) from exc


def resolve_media_path(audio_path: str | None, video_path: str | None) -> str | None:
    if video_path:
        return video_path
    return audio_path


def run_inference(
    audio_path: str | None,
    video_path: str | None,
    question: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
):
    prompt = (question or "").strip() or DEFAULT_QUESTION
    model_name_or_path = read_env_model_id()
    device = resolve_device()

    try:
        inference = get_inference(model_name_or_path, device)
    except Exception as exc:  # pragma: no cover - runtime environment dependent
        raise gr.Error(
            f"Failed to load the model. Please check the weights path or Hugging Face download status.\n{exc}"
        ) from exc

    media_path = resolve_media_path(audio_path, video_path)

    try:
        started_at = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="moss-audio-") as temp_dir:
            prepared_audio_path = media_path
            if media_path and Path(media_path).suffix.lower() in VIDEO_EXTENSIONS:
                prepared_audio_path = os.path.join(temp_dir, "input.mp3")
                convert_media_to_mp3(media_path, prepared_audio_path)

            answer = inference.generate(
                question=prompt,
                audio_path=prepared_audio_path,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
        elapsed_seconds = time.perf_counter() - started_at
    except Exception as exc:  # pragma: no cover - runtime environment dependent
        raise gr.Error(
            f"Inference failed. Please make sure the uploaded file is readable and the format is supported.\n{exc}"
        ) from exc

    return answer, format_status(model_name_or_path, device, elapsed_seconds)


with gr.Blocks(title=TITLE) as demo:
    gr.Markdown(f"# {TITLE}")

    with gr.Row():
        with gr.Column(scale=5):
            audio_input = gr.Audio(
                label="Audio",
                sources=["upload", "microphone"],
                type="filepath",
            )
            with gr.Accordion("Optional Video Input (.mp4)", open=False):
                gr.Markdown(
                    "Upload an mp4 only when needed. If a video is provided, its audio track will be extracted and used for inference."
                )
                video_input = gr.File(
                    label="Video File",
                    file_types=[".mp4"],
                    type="filepath",
                )
            question_input = gr.Textbox(
                label="Prompt",
                lines=4,
                value=DEFAULT_QUESTION,
                placeholder="For example: Please transcribe this audio. Describe the sounds in this clip. What emotion does the speaker convey?",
            )

            with gr.Accordion("Advanced Settings", open=False):
                max_new_tokens_input = gr.Slider(
                    minimum=64,
                    maximum=2048,
                    value=DEFAULT_MAX_NEW_TOKENS,
                    step=32,
                    label="Max New Tokens",
                )
                temperature_input = gr.Slider(
                    minimum=0.0,
                    maximum=1.5,
                    value=DEFAULT_TEMPERATURE,
                    step=0.1,
                    label="Temperature",
                )
                top_p_input = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=DEFAULT_TOP_P,
                    step=0.05,
                    label="Top-p",
                )
                top_k_input = gr.Slider(
                    minimum=1,
                    maximum=100,
                    value=DEFAULT_TOP_K,
                    step=1,
                    label="Top-k",
                )

            with gr.Row():
                submit_btn = gr.Button("Generate", variant="primary")
                gr.ClearButton(
                    [audio_input, video_input, question_input, max_new_tokens_input, temperature_input, top_p_input, top_k_input],
                    value="Clear",
                )

        with gr.Column(scale=5):
            output_text = gr.Textbox(label="Output", lines=16)
            status_text = gr.Markdown("Waiting for input.")

    gr.Examples(
        examples=[
            ["Describe this audio."],
            ["Please transcribe this audio."],
            ["What is happening in this audio clip?"],
            ["Describe the speaker's voice characteristics in detail."],
            ["What emotion does the speaker convey?"],
        ],
        inputs=[question_input],
        label="Prompt Examples",
    )

    submit_btn.click(
        fn=run_inference,
        inputs=[
            audio_input,
            video_input,
            question_input,
            max_new_tokens_input,
            temperature_input,
            top_p_input,
            top_k_input,
        ],
        outputs=[output_text, status_text],
    )


if __name__ == "__main__":
    server_name = os.environ.get("MOSS_AUDIO_SERVER_NAME", "127.0.0.1")
    server_port = int(os.environ.get("MOSS_AUDIO_SERVER_PORT", "7860"))
    demo.queue(max_size=8).launch(
        server_name=server_name,
        server_port=server_port,
    )
