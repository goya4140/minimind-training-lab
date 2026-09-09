#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, MimiModel, SiglipImageProcessor, SiglipVisionModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from minimind_lab.omni import MiniMindOmni, OmniConfig
from minimind_lab.omni.external import load_sensevoice
from minimind_lab.training import resolve_device, seed_everything
from minimind_lab.training.utils import environment_info, write_json


def audio_features(path: Path, processor):
    import librosa

    waveform, _ = librosa.load(path, sr=16_000, mono=True)
    inputs = processor(waveform, sampling_rate=16_000, return_tensors="pt", return_attention_mask=True)
    return inputs.input_features, inputs.attention_mask.sum(dim=1)


@torch.inference_mode()
def run_case(model, tokenizer, case, eval_dir, audio_bundle, vision_bundle, device, max_new_tokens):
    encoded_audio = encoded_audio_lengths = encoded_images = None
    content = case.get("prompt", "")
    if case["kind"] == "audio":
        encoder, processor = audio_bundle
        features, lengths = audio_features(eval_dir / case["file"], processor)
        encoded_audio, encoded_audio_lengths = encoder(features.to(device), lengths.to(device))
        content = "<|audio_pad|>" * int(lengths.item())
    elif case["kind"] == "image":
        from PIL import Image

        encoder, processor = vision_bundle
        pixels = processor(images=Image.open(eval_dir / case["file"]).convert("RGB"), return_tensors="pt")
        encoded_images = encoder(pixel_values=pixels["pixel_values"].to(device)).last_hidden_state
        content = f"{'<|image_pad|>' * model.config.image_token_length}\n{content}"

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer(prompt).input_ids
    started = time.perf_counter()
    generated = model.generate_multimodal(
        torch.tensor([prompt_ids], device=device),
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=max_new_tokens,
        text_temperature=0.0,
        audio_temperature=0.2,
        encoded_audio=encoded_audio,
        encoded_audio_lengths=encoded_audio_lengths,
        encoded_images=encoded_images,
    )
    elapsed = time.perf_counter() - started
    return {
        **case,
        "completion": tokenizer.decode(generated["text_ids"][0].tolist(), skip_special_tokens=True),
        "audio_codes": generated["audio_codes"].cpu(),
        "seconds": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--audio-encoder", default="assets/models/SenseVoiceSmall")
    parser.add_argument("--vision-encoder", default="assets/models/siglip2-base-p32-256-ve")
    parser.add_argument("--codec", default="assets/models/mimi")
    parser.add_argument("--output", default="artifacts/eval/omni.json")
    args = parser.parse_args()
    seed_everything(47)
    device = resolve_device(args.device)
    state = torch.load(ROOT / args.checkpoint, map_location="cpu", weights_only=False)
    model = MiniMindOmni(OmniConfig(**state["model_config"])).to(device)
    model.load_state_dict(state["model"])
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "assets/tokenizer", local_files_only=True)

    audio_bundle = load_sensevoice(ROOT / args.audio_encoder, device)
    vision_path = ROOT / args.vision_encoder
    vision = SiglipVisionModel.from_pretrained(vision_path, local_files_only=True).eval().to(device)
    vision_processor = SiglipImageProcessor.from_pretrained(vision_path, local_files_only=True)
    codec = MimiModel.from_pretrained(ROOT / args.codec, local_files_only=True).eval()

    eval_dir = ROOT / "data/eval/omni"
    cases = json.loads((eval_dir / "cases.json").read_text(encoding="utf-8"))
    output_dir = (ROOT / args.output).parent / "omni-audio"
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    import soundfile as sf

    for case in cases:
        result = run_case(
            model,
            tokenizer,
            case,
            eval_dir,
            audio_bundle,
            (vision, vision_processor),
            device,
            args.max_new_tokens,
        )
        codes = result.pop("audio_codes")
        result["audio_frames"] = codes.size(-1)
        if codes.size(-1):
            audio = codec.decode(codes).audio_values.squeeze().float().cpu().numpy()
            audio_path = output_dir / f"{case['id']}.wav"
            sf.write(audio_path, audio, 24_000)
            result["audio_path"] = str(audio_path.relative_to(ROOT))
        results.append(result)

    report = {
        "checkpoint": args.checkpoint,
        "environment": environment_info(device),
        "cases": results,
    }
    write_json(ROOT / args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
