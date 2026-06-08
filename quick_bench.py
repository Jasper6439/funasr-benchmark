#!/usr/bin/env python3
"""
Quick benchmark using bundled example audio from SenseVoice.
No extra downloads needed — runs immediately after pip install.
"""

import sys
from pathlib import Path

# Find example audio from cached models
CACHE = Path.home() / ".cache"
MODELS_DIR = CACHE / "modelscope/hub/models/iic/SenseVoiceSmall/example"
HF_DIR = CACHE / "huggingface/hub/models--FunAudioLLM--Fun-ASR-Nano-2512/snapshots"

def find_examples():
    """Find example audio from cached model downloads."""
    examples = {}
    
    # Try ModelScope cache
    if MODELS_DIR.exists():
        for f in MODELS_DIR.glob("*.mp3"):
            examples[f.stem] = str(f)
    
    # Try HuggingFace cache
    if HF_DIR.exists():
        for snap_dir in HF_DIR.iterdir():
            ex = snap_dir / "example"
            if ex.exists():
                for f in ex.glob("*.mp3"):
                    if f.stem not in examples:
                        examples[f.stem] = str(f)
    
    return examples

def main():
    examples = find_examples()
    if not examples:
        print("No example audio found. Run the models at least once to download examples:")
        print("  python -c \"from funasr import AutoModel; AutoModel(model='iic/SenseVoiceSmall', device='cpu')\"")
        sys.exit(1)
    
    print(f"Found {len(examples)} example(s): {', '.join(examples.keys())}")
    
    # Build audio path list
    sample_dir = Path("samples")
    sample_dir.mkdir(exist_ok=True)
    
    # Create symlinks to examples
    for name, path in examples.items():
        link = sample_dir / f"{name}.mp3"
        if not link.exists():
            link.symlink_to(path)
    
    # Run benchmark
    import subprocess
    cmd = [
        sys.executable, "benchmark.py",
        "--audio", "samples/",
        "--models", "sensevoice", "nano",
    ]
    print(f"\nRunning: {' '.join(cmd)}\n")
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
