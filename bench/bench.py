"""Head-to-head benchmark: myers-batch against edlib on identical data.

Both sides run in one process so they see the same data, the same timer and the
same cache state. Single-threaded throughout; the multi-core number is reported
separately and is not part of the algorithmic claim.

Usage:  python bench/bench.py
"""

from __future__ import annotations

import platform
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import edlib

import myers_batch

ALPHABET = b"ACGT"


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


def main() -> None:
    import random

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
