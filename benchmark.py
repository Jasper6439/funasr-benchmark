#!/usr/bin/env python3
"""
FunASR Model Benchmark — Compare speech recognition models on real audio.

Usage:
    python benchmark.py --audio samples/ --models all
    python benchmark.py --audio file.wav --models sensevoice nano --ground-truth ref.txt
"""

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def edit_distance(ref: list, hyp: list) -> int:
    """Levenshtein edit distance between two token lists."""
    n, m = len(ref), len(hyp)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            temp = dp[j]
            dp[j] = min(
                dp[j] + 1,       # deletion
                dp[j - 1] + 1,   # insertion
                prev + (0 if ref[i - 1] == hyp[j - 1] else 1),  # substitution
            )
            prev = temp
    return dp[m]


def wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate (for English / word-segmented languages)."""
    ref = reference.strip().split()
    hyp = hypothesis.strip().split()
    if not ref:
        return 0.0 if not hyp else 1.0
    return edit_distance(ref, hyp) / len(ref)


def cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate (for Chinese / character-level languages)."""
    ref = list(reference.strip().replace(" ", ""))
    hyp = list(hypothesis.strip().replace(" ", ""))
    if not ref:
        return 0.0 if not hyp else 1.0
    return edit_distance(ref, hyp) / len(ref)


def normalize_text(text: str) -> str:
    """Strip FunASR emotion/event tags, lowercase, remove punctuation."""
    text = re.sub(r"<\|[^|]+\|>", "", text)
    text = text.lower()
    text = re.sub(r"[，。！？、；：\u201c\u201d\u2018\u2019（）【】《》…—\-.!?;:\"\u0027()\[\]{}]", "", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Model wrappers
# ---------------------------------------------------------------------------

@dataclass
class ModelResult:
    model_name: str
    audio_file: str
    raw_text: str
    clean_text: str
    inference_time: float
    rtf: float = 0.0
    wer: Optional[float] = None
    cer: Optional[float] = None
    language: Optional[str] = None
    emotion: Optional[str] = None


class FunASRBenchmark:
    """Load and run FunASR models for benchmarking."""

    MODEL_CONFIGS = {
        "sensevoice": {
            "model_id": "iic/SenseVoiceSmall",
            "hub": "modelscope",
            "description": "SenseVoiceSmall — fast, emotion-aware",
        },
        "nano": {
            "model_id": "FunAudioLLM/Fun-ASR-Nano-2512",
            "hub": "huggingface",
            "description": "Fun-ASR-Nano — LLM-based, high accuracy",
        },
        "paraformer": {
            "model_id": "iic/paraformer-zh",
            "hub": "modelscope",
            "description": "Paraformer-zh — production Chinese",
        },
    }

    def __init__(self, device: str = "cpu", models: list[str] | None = None):
        self.device = device
        self.models_to_load = models or list(self.MODEL_CONFIGS.keys())
        self._loaded: dict = {}

    def _load_model(self, name: str):
        if name in self._loaded:
            return self._loaded[name]

        from funasr import AutoModel

        cfg = self.MODEL_CONFIGS[name]
        hub = cfg.get("hub", "modelscope")
        print(f"  Loading {name} ({cfg['model_id']}) from {hub}...", flush=True)

        kwargs = dict(
            model=cfg["model_id"],
            device=self.device,
            disable_update=True,
        )
        if hub == "huggingface":
            kwargs["hub"] = "huggingface"

        model = AutoModel(**kwargs)
        self._loaded[name] = model
        return model

    def transcribe(self, model_name: str, audio_path: str) -> ModelResult:
        """Run inference on a single audio file."""
        model = self._load_model(model_name)

        t0 = time.time()
        results = model.generate(input=audio_path)
        elapsed = time.time() - t0

        raw_text = results[0]["text"] if results else ""
        clean = normalize_text(raw_text)

        # Extract language/emotion from SenseVoice tags
        lang_match = re.search(r"<\|(zh|en|ja|ko|yue)\|>", raw_text)
        emo_match = re.search(r"<\|(NEUTRAL|HAPPY|SAD|ANGRY|FEAR|DISGUST|SURPRISE)\|>", raw_text)

        # Estimate audio duration for RTF
        try:
            import soundfile as sf
            info = sf.info(audio_path)
            audio_dur = info.duration
        except Exception:
            audio_dur = 0.0

        return ModelResult(
            model_name=model_name,
            audio_file=str(audio_path),
            raw_text=raw_text,
            clean_text=clean,
            inference_time=elapsed,
            rtf=elapsed / audio_dur if audio_dur > 0 else 0.0,
            language=lang_match.group(1) if lang_match else None,
            emotion=emo_match.group(1) if emo_match else None,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def find_audio_files(path: str) -> list[Path]:
    """Find audio files in a directory or return single file."""
    p = Path(path)
    if p.is_file():
        return [p]
    if p.is_dir():
        exts = {".wav", ".mp3", ".ogg", ".m4a", ".flac", ".opus"}
        return sorted(f for f in p.rglob("*") if f.suffix.lower() in exts)
    return []


def load_ground_truth(path: str) -> dict[str, str]:
    """Load ground truth transcripts. Supports:
    - Single .txt file (applied to all audio)
    - Directory with .txt files matching audio filenames
    - JSON file: {"filename": "transcript", ...}
    """
    p = Path(path)
    if p.is_file():
        if p.suffix == ".json":
            return json.loads(p.read_text())
        # Single text file — same transcript for all
        text = p.read_text().strip()
        return {"__default__": text}
    if p.is_dir():
        result = {}
        for f in p.glob("*.txt"):
            result[f.stem] = f.read_text().strip()
        return result
    return {}


def print_results(results: list[ModelResult], output_json: Optional[str] = None):
    """Print formatted results table."""
    if not results:
        print("No results.")
        return

    # Group by audio file
    by_file: dict[str, list[ModelResult]] = {}
    for r in results:
        by_file.setdefault(r.audio_file, []).append(r)

    for audio, file_results in by_file.items():
        print(f"\n{'=' * 60}")
        print(f"Audio: {Path(audio).name}")
        print(f"{'=' * 60}")

        for r in file_results:
            print(f"\n  Model: {r.model_name}")
            print(f"  Text:  {r.clean_text}")
            print(f"  Time:  {r.inference_time:.2f}s (RTF: {r.rtf:.3f})")
            if r.language:
                print(f"  Lang:  {r.language}")
            if r.emotion:
                print(f"  Emot:  {r.emotion}")
            if r.wer is not None:
                print(f"  WER:   {r.wer:.4f}")
            if r.cer is not None:
                print(f"  CER:   {r.cer:.4f}")

    # Summary table
    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    print(f"{'Model':<20} {'Avg Time':>10} {'Avg RTF':>10} {'Avg CER':>10} {'Avg WER':>10}")
    print("-" * 60)

    model_stats: dict[str, list[ModelResult]] = {}
    for r in results:
        model_stats.setdefault(r.model_name, []).append(r)

    for name, rs in model_stats.items():
        avg_time = sum(r.inference_time for r in rs) / len(rs)
        avg_rtf = sum(r.rtf for r in rs) / len(rs)
        cers = [r.cer for r in rs if r.cer is not None]
        wers = [r.wer for r in rs if r.wer is not None]
        avg_cer = f"{sum(cers)/len(cers):.4f}" if cers else "N/A"
        avg_wer = f"{sum(wers)/len(wers):.4f}" if wers else "N/A"
        print(f"{name:<20} {avg_time:>9.2f}s {avg_rtf:>10.4f} {avg_cer:>10} {avg_wer:>10}")

    # Save JSON if requested
    if output_json:
        data = [asdict(r) for r in results]
        Path(output_json).write_text(json.dumps(data, indent=2, ensure_ascii=False))
        print(f"\nResults saved to {output_json}")


def main():
    parser = argparse.ArgumentParser(description="FunASR Model Benchmark")
    parser.add_argument("--audio", required=True, help="Audio file or directory")
    parser.add_argument("--models", nargs="+", default=["all"],
                        help="Models to test: sensevoice, nano, paraformer, all")
    parser.add_argument("--device", default="cpu", help="Device: cpu or cuda")
    parser.add_argument("--ground-truth", "--gt", help="Ground truth file/directory")
    parser.add_argument("--output", "-o", help="Save results as JSON")
    parser.add_argument("--language", default="auto",
                        help="Language hint for WER/CER: zh, en, auto")
    args = parser.parse_args()

    # Resolve models
    benchmark = FunASRBenchmark(device=args.device)
    if "all" in args.models:
        models = list(benchmark.MODEL_CONFIGS.keys())
    else:
        models = args.models

    # Find audio files
    audio_files = find_audio_files(args.audio)
    if not audio_files:
        print(f"Error: No audio files found at {args.audio}", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(audio_files)} audio file(s)")

    # Load ground truth
    gt = load_ground_truth(args.ground_truth) if args.ground_truth else {}

    # Run benchmark
    all_results: list[ModelResult] = []
    for audio in audio_files:
        print(f"\n--- Processing: {audio.name} ---")
        for model_name in models:
            print(f"  [{model_name}]", end=" ", flush=True)
            result = benchmark.transcribe(model_name, str(audio))

            # Calculate error rates if ground truth available
            if gt:
                ref = gt.get(audio.stem) or gt.get("__default__", "")
                if ref:
                    # Auto-detect: use CER for Chinese, WER for others
                    lang_hint = args.language
                    if lang_hint == "auto":
                        # Check if reference contains CJK
                        has_cjk = any("\u4e00" <= c <= "\u9fff" for c in ref)
                        lang_hint = "zh" if has_cjk else "en"

                    if lang_hint == "zh":
                        result.cer = cer(ref, result.clean_text)
                        result.wer = wer(ref, result.clean_text)
                    else:
                        result.wer = wer(ref, result.clean_text)
                        result.cer = cer(ref, result.clean_text)

                    print(f"CER={result.cer:.4f} WER={result.wer:.4f}" if result.cer else f"WER={result.wer:.4f}")
                else:
                    print("done")
            else:
                print("done")

            all_results.append(result)

    # Print results
    print_results(all_results, output_json=args.output)


if __name__ == "__main__":
    main()
