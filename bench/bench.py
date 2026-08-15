"""Head-to-head benchmark: myers-batch against edlib on identical data.

Both sides run in one process so they see the same data, the same timer and the
same cache state. Single-threaded throughout; the multi-core number is reported
separately and is not part of the algorithmic claim.

Usage:  python bench/bench.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import myers_batch

ALPHABET = b"ACGT"
E2E_EXPECTED_SHA256 = "26b487d016a5afc7c4f35f42af23912125aaacbb93845a7b6dc4522922316ba0"


def make_reads(rng, n, read_len, query, hit_rate=0.7, max_err=3):
    """Reads shaped like a sequencer run: most contain the query, with errors."""
    import random

    out = []
    m = len(query)
    for _ in range(n):
        r = bytearray(rng.choice(ALPHABET) for _ in range(read_len))
        if rng.random() < hit_rate and read_len >= m:
            pos = rng.randint(0, read_len - m)
            r[pos : pos + m] = query
            for _ in range(rng.randint(0, max_err)):
                r[rng.randrange(read_len)] = rng.choice(ALPHABET)
        out.append(bytes(r))
    assert isinstance(rng, random.Random)
    return out


def timeit(fn, repeats=5):
    fn()  # warm up
    runs = []
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        fn()
        runs.append(time.perf_counter_ns() - t0)
    return min(runs), statistics.median(runs)


def case(rng, label, query, n_reads, read_len, k_values, threads=0):
    reads = make_reads(rng, n_reads, read_len, query)
    m = len(query)

    ref = [edlib.align(query, r, mode="HW", task="distance")["editDistance"] for r in reads]
    got = myers_batch.distances(query, reads)
    identical = got == ref

    print(f"\n=== {label}  (query {m}bp, {n_reads} x {read_len}bp reads) ===")
    print(f"  results identical to edlib on all {n_reads} reads: {identical}")
    if not identical:
        raise SystemExit("MISMATCH: refusing to report a benchmark for incorrect output")

    res = {}
    for k in k_values:

        def edlib_pass(k=k):
            for r in reads:
                edlib.align(query, r, mode="HW", task="distance", k=k)

        lo, _ = timeit(edlib_pass)
        res[f"edlib k={k}"] = lo
    lo, _ = timeit(lambda: myers_batch.distances_scalar(query, reads))
    res["myers-batch scalar"] = lo
    lo, _ = timeit(lambda: myers_batch.distances(query, reads))
    res["myers-batch NEON"] = lo

    for name, ns in res.items():
        print(f"  {name:<20} {ns / 1e6:8.2f} ms   {n_reads / (ns / 1e9) / 1e6:6.2f}M pairs/s")

    base = res["edlib k=-1"]
    fast = res["myers-batch NEON"]
    print(f"  speedup vs edlib unbounded : {base / fast:5.2f}x")
    for k in k_values:
        if k != -1:
            print(f"  speedup vs edlib k={k:<3}      : {res[f'edlib k={k}'] / fast:5.2f}x")
    print(
        f"  of which: batch API + single-word path {base / res['myers-batch scalar']:5.2f}x, "
        f"NEON vectorization {res['myers-batch scalar'] / fast:5.2f}x"
    )

    if threads:
        shards = [reads[i::threads] for i in range(threads)]
        with ThreadPoolExecutor(max_workers=threads) as ex:
            lo, _ = timeit(lambda: list(ex.map(lambda s: myers_batch.distances(query, s), shards)))
        print(
            f"  [{threads} threads, not part of the algorithmic claim] {lo / 1e6:8.2f} ms   "
            f"{n_reads / (lo / 1e9) / 1e6:6.2f}M pairs/s   {base / lo:5.2f}x vs single-thread edlib"
        )


def mixed_workload():
    """Deterministic ragged sequencer batch for end-to-end API measurement."""
    rng = random.Random(20260815)
    query = b"AGATCGGAAGAGCACACGTCTGAACTCCAGTCA"
    lengths = (75, 100, 125, 150, 180, 220, 260, 300)
    reads = []
    for _ in range(100_000):
        n = rng.choice(lengths)
        read = bytearray(rng.choice(ALPHABET) for _ in range(n))
        if rng.random() < 0.7:
            pos = rng.randrange(n - len(query) + 1)
            read[pos : pos + len(query)] = query
            for _ in range(rng.randrange(4)):
                read[rng.randrange(n)] = rng.choice(ALPHABET)
        reads.append(bytes(read))
    return query, reads, lengths


def measured_samples(fn, expected, warmups=3, repeats=11):
    for _ in range(warmups):
        if fn() != expected:
            raise SystemExit("MISMATCH: measured result changed during warmup")
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        result = fn()
        samples.append(time.perf_counter_ns() - start)
        if result != expected:
            raise SystemExit("MISMATCH: measured result changed between timed samples")
    return samples


def e2e_result():
    query, reads, lengths = mixed_workload()
    scalar = myers_batch.distances_scalar(query, reads)
    public = myers_batch.distances(query, reads)
    if public != scalar:
        raise SystemExit("MISMATCH: public dispatcher differs from scalar oracle")
    samples = measured_samples(lambda: myers_batch.distances(query, reads), scalar)
    digest = hashlib.sha256(
        b"".join(distance.to_bytes(4, "little", signed=True) for distance in public)
    ).hexdigest()
    if digest != E2E_EXPECTED_SHA256:
        raise SystemExit(f"MISMATCH: expected result digest {E2E_EXPECTED_SHA256}, got {digest}")
    scalar_samples = measured_samples(lambda: myers_batch.distances_scalar(query, reads), scalar)
    return {
        "workload": "distances(query, targets) mixed-length sequencer batch",
        "backend": myers_batch.simd_backend(),
        "query_length": len(query),
        "targets": len(reads),
        "target_lengths": list(lengths),
        "target_bytes": sum(map(len, reads)),
        "warmups": 3,
        "samples": len(samples),
        "sample_ms": [round(ns / 1e6, 6) for ns in samples],
        "median_ms": round(statistics.median(samples) / 1e6, 6),
        "min_ms": round(min(samples) / 1e6, 6),
        "max_ms": round(max(samples) / 1e6, 6),
        "scalar_equivalent": True,
        "scalar_median_ms": round(statistics.median(scalar_samples) / 1e6, 6),
        "sha256_int32_le": digest,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--e2e-json",
        action="store_true",
        help="run the deterministic 11-sample public-API benchmark and emit JSON",
    )
    args = parser.parse_args()
    if args.e2e_json:
        print(json.dumps(e2e_result(), sort_keys=True))
        return
    global edlib
    import edlib

    rng = random.Random(20260812)
    print(f"machine : {platform.platform()} / {platform.machine()}")
    print(f"python  : {platform.python_version()}")
    print(f"kernel  : NEON={myers_batch.have_neon()} lanes={myers_batch.lanes()}")

    case(
        rng,
        "TruSeq Read 1 adapter",
        b"AGATCGGAAGAGCACACGTCTGAACTCCAGTCA",
        200_000,
        150,
        [-1, 5, 2],
        threads=8,
    )
    case(rng, "16bp cell barcode", b"GTCACGTAGCTTACGA", 200_000, 150, [-1, 3, 1])
    case(
        rng,
        "64bp capture probe",
        bytes(rng.choice(ALPHABET) for _ in range(64)),
        100_000,
        300,
        [-1, 8],
    )


if __name__ == "__main__":
    main()
