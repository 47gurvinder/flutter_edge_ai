"""Step 2 of the Knowledge screen's sample voice memos (step 1 is
generate_knowledge_memos.dart, which has Matcha-TTS speak them on macOS):
resample each clip to 16 kHz mono PCM16 — the rate the STT models and the
EmbeddingGemma 2 audio encoder work at — into
example/assets/knowledge/memos/, with FFmpeg's bit-exact flags.

usage: python3 -I convert_knowledge_memos.py <directory step 1 wrote>
       (needs ffmpeg on PATH)
"""
import os
import subprocess
import sys

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "assets", "knowledge", "memos")

BITEXACT = ["-map_metadata", "-1", "-fflags", "+bitexact", "-flags:a", "+bitexact"]


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    source = sys.argv[1]
    os.makedirs(OUT, exist_ok=True)
    names = sorted(n for n in os.listdir(source) if n.endswith(".wav"))
    if not names:
        sys.exit(f"no .wav files in {source}")
    for name in names:
        subprocess.run(
            ["ffmpeg", "-y", "-i", os.path.join(source, name), "-ar", "16000", "-ac", "1",
             "-c:a", "pcm_s16le", *BITEXACT, os.path.join(OUT, name)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        print(f"{name} -> {os.path.join(OUT, name)}")


if __name__ == "__main__":
    main()
