"""Build a full-resolution, lossless animated WebP from the robot recording.

Optional dependencies: av (validated with 14.2.0) and Pillow with WebP support.
The original video supplies every pixel; no palette quantization or resizing.
"""
from pathlib import Path
import argparse

import av
from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "docs/media/robot_demo.webm")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/media/robot_demo.webp")
    args = parser.parse_args()
    frames = []
    stride = 4
    with av.open(str(args.source)) as source:
        rate = source.streams.video[0].average_rate
        for index, frame in enumerate(source.decode(video=0)):
            if index % stride == 0:
                frames.append(frame.to_image())
    if not frames:
        raise ValueError("Source contains no video frames")
    duration = round(1000 * stride / rate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(args.output, format="WEBP", save_all=True, append_images=frames[1:],
                   duration=duration, loop=0, lossless=True, quality=80, method=4)
    with Image.open(args.output) as preview:
        assert preview.size == frames[0].size
        assert preview.n_frames == len(frames)
        for index, original in enumerate(frames):
            preview.seek(index)
            assert ImageChops.difference(original, preview.convert("RGB")).getbbox() is None
    print(f"Verified {len(frames)} lossless frames at {frames[0].size}; {duration} ms per frame.")


if __name__ == "__main__":
    main()
