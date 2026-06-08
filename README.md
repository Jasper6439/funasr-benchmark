# FunASR Model Benchmark

Compare FunASR speech recognition models side-by-side on real audio files.

## Models Supported

| Model | Type | Languages | Features |
|-------|------|-----------|----------|
| SenseVoiceSmall | CTC | zh/en/ja/ko/yue | Emotion, events, fast |
| Fun-ASR-Nano | LLM (Qwen3-0.6B) | 31 languages | High accuracy, punctuation |
| Paraformer-zh | CTC | zh/en | Production-grade |

## Quick Start

```bash
pip install funasr torch torchaudio soundfile jiwer

# Run benchmark on sample audio
python benchmark.py --audio samples/ --models all

# Compare specific models
python benchmark.py --audio samples/zh_podcast.wav --models sensevoice nano

# With ground truth for WER/CER calculation
python benchmark.py --audio samples/ --ground-truth transcripts/ --models all
```

## Output

```
=== FunASR Benchmark Results ===

Model              | CER(zh) | WER(en) | Time(s) | RTF
SenseVoiceSmall    | 0.0523  | 0.0812  | 1.23    | 0.14
Fun-ASR-Nano       | 0.0489  | 0.0756  | 3.45    | 0.39
Paraformer-zh      | 0.0534  | 0.0845  | 0.98    | 0.11
```

## Methodology

- Audio files are processed through each model with identical settings
- WER (Word Error Rate) for English, CER (Character Error Rate) for Chinese
- Ground truth transcripts required for accuracy metrics
- Timing includes model inference only (excludes loading)

## License

MIT
