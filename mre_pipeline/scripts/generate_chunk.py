#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tqdm import tqdm
from src.dataset import make_training_pair, write_chunk


def main():
    parser = argparse.ArgumentParser(description="Generate one chunk of MRE training data")
    parser.add_argument("--task_id", type=int, required=True)
    parser.add_argument("--chunk_size", type=int, default=500)
    parser.add_argument("--grid_size", type=int, default=64)
    parser.add_argument("--output_dir", type=Path, default=Path("data/chunks"))
    args = parser.parse_args()

    import numpy as np

    rng = np.random.default_rng(args.task_id * 1000)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"chunk_{args.task_id:04d}.h5"

    samples = []
    for _ in tqdm(range(args.chunk_size), desc=f"Task {args.task_id}"):
        X, Y, meta = make_training_pair(N=args.grid_size, rng=rng)
        samples.append((X, Y, meta))

    write_chunk(samples, out_path, chunk_id=args.task_id, N=args.grid_size)
    print(f"Saved {args.chunk_size} samples to {out_path}")


if __name__ == "__main__":
    main()
