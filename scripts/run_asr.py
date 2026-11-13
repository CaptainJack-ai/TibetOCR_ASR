"""Run Tibetan ASR checkpoints over NICT-Tib1 test or the OpenPecha benchmark.

Examples:
  python scripts/run_asr.py --model whisper-small --dataset nict --limit 20
  python scripts/run_asr.py --model wav2vec2-base --dataset openpecha
"""
from __future__ import annotations

import argparse
import json
import resource
import time
from datetime import datetime
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch

from common import ASR_DIR, OUT_DIR, dump_json, get_logger, load_json

log = get_logger("run_asr")

MODELS = {
    "whisper-tiny": "billingsmoore/tibetan-asr-nict-tib1-whisper-tiny",
    "whisper-base": "billingsmoore/tibetan-asr-nict-tib1-whisper-base",
    "wav2vec2-base": "billingsmoore/tibetan-asr-nict-tib1-wav2vec2-base",
    "whisper-small": "billingsmoore/tibetan-asr-nict-tib1-whisper-small",
    "tenzin-small": "tenzin/whisper-small-tibetan",
    "openpecha-general": "openpecha/general_stt_base_model",
}

DATASETS = {
    "nict": ASR_DIR / "nict_test",
    "openpecha": ASR_DIR / "openpecha_bench",
}


def load_rows(dataset: str, limit: int) -> list[dict]:
    meta = pd.read_csv(DATASETS[dataset] / "metadata.csv", dtype={"id": str})
    if dataset == "nict" and limit:
        ids = set(load_json(DATASETS[dataset] / "sample_ids.json")["sample"])
        meta = meta[meta.id.isin(ids)]
    if limit:
        meta = meta.head(limit)
    return meta.to_dict("records")


def resolve_model(name: str) -> str:
    return MODELS.get(name, name)


def peak_memory_mb() -> float:
    if torch.backends.mps.is_available() and torch.mps.is_available():
        try:
            return round(torch.mps.driver_allocated_memory() / 1e6, 1)
        except Exception:  # noqa: BLE001
            pass
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 1)


class Transcriber:
    def __init__(self, model_id: str, device: str, dtype: str, beams: int, language: str | None,
                 max_new_tokens: int = 440):
        from transformers import AutoConfig, AutoModelForCTC, AutoModelForSpeechSeq2Seq, AutoProcessor

        self.model_id = model_id
        self.device = device
        self.beams = beams
        self.language = language
        self.max_new_tokens = max_new_tokens
        cfg = AutoConfig.from_pretrained(model_id)
        self.kind = "whisper" if cfg.model_type in ("whisper",) else "ctc"
        self.dtype = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}[dtype]
        log.info("loading %s (kind=%s device=%s dtype=%s)", model_id, self.kind, device, dtype)

        try:
            self.processor = AutoProcessor.from_pretrained(model_id)
        except (OSError, ValueError):
            # Several Tibetan checkpoints ship weights only; rebuild the processor
            # from the upstream base model / default feature extractor.
            if self.kind == "whisper":
                from transformers import WhisperFeatureExtractor, WhisperProcessor, WhisperTokenizer

                base = {384: "openai/whisper-tiny", 512: "openai/whisper-base",
                        768: "openai/whisper-small", 1024: "openai/whisper-medium",
                        1280: "openai/whisper-large-v3"}.get(cfg.d_model, "openai/whisper-small")
                fe = WhisperFeatureExtractor(feature_size=cfg.num_mel_bins)
                tok = WhisperTokenizer.from_pretrained(base)
                self.processor = WhisperProcessor(feature_extractor=fe, tokenizer=tok)
                log.info("processor rebuilt from %s", base)
            else:
                from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor

                tok = Wav2Vec2CTCTokenizer.from_pretrained(model_id)
                fe = Wav2Vec2FeatureExtractor(
                    sampling_rate=16000, do_normalize=True, return_attention_mask=False
                )
                self.processor = Wav2Vec2Processor(feature_extractor=fe, tokenizer=tok)
                log.info("CTC processor rebuilt from repo tokenizer")

        cls = AutoModelForSpeechSeq2Seq if self.kind == "whisper" else AutoModelForCTC
        try:
            self.model = cls.from_pretrained(model_id, dtype=self.dtype, attn_implementation="sdpa")
        except TypeError:
            self.model = cls.from_pretrained(model_id, dtype=self.dtype)
        self.model.to(device).eval()

    def __call__(self, wav_path: str) -> str:
        audio, _ = librosa.load(ASR_DIR.parent.parent / wav_path if not Path(wav_path).is_absolute() else wav_path, sr=16000, mono=True)
        return self.decode_array(audio)

    def decode_array(self, audio: "np.ndarray") -> str:
        """Decode an in-memory 16 kHz mono waveform."""
        if self.kind == "whisper":
            feats = self.processor(audio, sampling_rate=16000, return_tensors="pt").input_features
            feats = feats.to(self.device, dtype=self.dtype)
            kwargs = {"num_beams": self.beams, "max_new_tokens": self.max_new_tokens}
            if self.language:
                kwargs["language"] = self.language
                kwargs["task"] = "transcribe"
            with torch.inference_mode():
                ids = self.model.generate(feats, **kwargs)
            return self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        feats = self.processor(audio, sampling_rate=16000, return_tensors="pt").input_values
        feats = feats.to(self.device, dtype=self.dtype)
        with torch.inference_mode():
            logits = self.model(feats).logits
        ids = torch.argmax(logits, dim=-1)
        return self.processor.batch_decode(ids)[0].strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="alias (see MODELS) or full HF repo id")
    ap.add_argument("--dataset", choices=list(DATASETS), default="nict")
    ap.add_argument("--limit", type=int, default=0, help="0 = dataset default sample (300 NICT / 160 OpenPecha)")
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"])
    ap.add_argument("--dtype", default="auto", choices=["auto", "fp16", "fp32", "bf16"])
    ap.add_argument("--beams", type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, default=440, help="Whisper decoder budget (max 444)")
    ap.add_argument("--language", default=None, help="force Whisper language token, e.g. bo")
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    device = args.device
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = args.dtype
    if dtype == "auto":
        dtype = "fp16" if device == "mps" else "fp32"

    rows = load_rows(args.dataset, args.limit)
    model_id = resolve_model(args.model)
    tag = args.model if args.model in MODELS else model_id.split("/")[-1]
    run_id = args.run_id or f"{tag}_{args.dataset}_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir = OUT_DIR / "asr" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    pipe = Transcriber(model_id, device, dtype, args.beams, args.language, args.max_new_tokens)

    out, t0 = [], time.time()
    for i, row in enumerate(rows, 1):
        start = time.time()
        status, hyp = "ok", ""
        try:
            hyp = pipe(row["audio_path"])
        except Exception as exc:  # noqa: BLE001
            status = f"error: {type(exc).__name__}: {exc}"[:300]
        elapsed = time.time() - start
        out.append(
            {
                "id": row["id"],
                "reference": row["transcription"],
                "transcription": hyp,
                "duration": row["duration"],
                "seconds": round(elapsed, 3),
                "status": status,
                "dept": row.get("dept", ""),
                "speaker": row.get("speaker", ""),
            }
        )
        if i % 10 == 0 or i == len(rows):
            log.info("[%d/%d] %s %.2fs (%d chars)", i, len(rows), status, elapsed, len(hyp))

    df = pd.DataFrame(out)
    df.to_csv(out_dir / "predictions.csv", index=False)
    dump_json(
        out_dir / "run.json",
        {
            "model": model_id,
            "kind": pipe.kind,
            "dataset": args.dataset,
            "n": len(rows),
            "device": device,
            "dtype": dtype,
            "beams": args.beams,
            "language": args.language,
            "total_seconds": round(time.time() - t0, 1),
            "audio_seconds": round(float(df.duration.sum()), 1),
            "rtf": round((time.time() - t0) / max(df.duration.sum(), 1e-9), 3),
            "peak_memory_mb": peak_memory_mb(),
            "errors": int((df.status != "ok").sum()),
        },
    )
    log.info("wrote %s (%.1f min audio, RTF %.2f)", out_dir, df.duration.sum() / 60, (time.time() - t0) / df.duration.sum())


if __name__ == "__main__":
    main()
