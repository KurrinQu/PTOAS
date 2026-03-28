#!/usr/bin/python3
# coding=utf-8

import numpy as np


ROWS = 32
COLS = 32
TILE_SIZE = ROWS * COLS
OUTPUT_COUNT = 32
SCALE = np.float32(2.0)
BIAS = np.float32(-0.125)
SLOPE = np.float32(0.3125)


def f32(x):
    return np.asarray(x, dtype=np.float32)


def make_inputs(seed=19):
    rng = np.random.default_rng(seed)
    a = f32(0.5 + rng.random((ROWS, COLS)) * 1.0)
    b = f32(0.25 + rng.random((ROWS, COLS)) * 1.5)
    c = f32(0.1 + rng.random((ROWS, COLS)) * 1.25)
    return a, b, c


def generate_golden(v1, v2, v3):
    a = v1.reshape(ROWS, COLS).astype(np.float32)
    b = v2.reshape(ROWS, COLS).astype(np.float32)
    c = v3.reshape(ROWS, COLS).astype(np.float32)

    outputs = [
        f32(a + b),
        f32(a - b),
        f32(a * b),
        f32(a / b),
        f32(np.maximum(a, b)),
        f32(np.minimum(a, b)),
        f32(np.fmod(a, b)),
        f32(np.where(a > 0.0, a, b * a)),
        f32(a + SCALE),
        f32(a - SCALE),
        f32(a * SCALE),
        f32(a / SCALE),
        f32(SCALE / a),
        f32(np.maximum(a, SCALE)),
        f32(np.minimum(a, SCALE)),
        f32(np.fmod(a, SCALE)),
        f32(a + b + c),
        f32(a - b + c),
        f32(a + BIAS + b),
        f32(a - BIAS + b),
        f32(np.abs(a)),
        f32(-a),
        f32(np.exp(a)),
        f32(np.log(a)),
        f32(np.sqrt(a)),
        f32(1.0 / np.sqrt(a)),
        f32(1.0 / a),
        f32(np.maximum(a, 0.0)),
        f32(np.where(a > 0.0, a, SLOPE * a)),
        # For a full 32x32 tile, partial arithmetic matches regular elementwise arithmetic.
        f32(a + b),
        f32(np.maximum(a, b)),
        f32(np.minimum(a, b)),
    ]

    assert len(outputs) == OUTPUT_COUNT
    return np.concatenate([out.reshape(-1) for out in outputs]).astype(np.float32)


def main():
    v1, v2, v3 = make_inputs()
    v1.reshape(-1).tofile("v1.bin")
    v2.reshape(-1).tofile("v2.bin")
    v3.reshape(-1).tofile("v3.bin")
    golden = generate_golden(v1.reshape(-1), v2.reshape(-1), v3.reshape(-1))
    golden.tofile("golden_v4.bin")


if __name__ == "__main__":
    main()
