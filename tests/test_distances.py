"""Correctness of the batched infix Myers kernel.

Two independent oracles are used. A pure-Python dynamic-programming reference
makes the suite self-contained, and edlib provides a differential check against
the tool this kernel is meant to replace.
"""

from __future__ import annotations

import random

import pytest

import myers_batch

edlib = pytest.importorskip("edlib", reason="edlib is the differential oracle")

ALPHABET = b"ACGT"


def dp_infix(query: bytes, target: bytes) -> int:
    """O(nm) reference: min edit distance of query against any substring of target.

    Row 0 is all zeros, which makes the target prefix free; the answer is the
    minimum over the final row, which makes the target suffix free.
    """
    m = len(query)
    prev = list(range(m + 1))
    best = m
    for tc in target:
        cur = [0] + [0] * m
        for i in range(1, m + 1):
            cost = 0 if query[i - 1] == tc else 1
            cur[i] = min(prev[i] + 1, cur[i - 1] + 1, prev[i - 1] + cost)
        prev = cur
        best = min(best, prev[m])
    return best


def rand_seq(rng: random.Random, n: int) -> bytes:
    return bytes(rng.choice(ALPHABET) for _ in range(n))


# --------------------------------------------------------------- basic shape


def test_exact_match_is_zero():
    assert myers_batch.distances(b"ACGTACGT", [b"TTTACGTACGTTTT"]) == [0]


def test_no_match_costs_whole_query():
    assert myers_batch.distances(b"AAAA", [b"CCCCCCCC"]) == [4]


def test_empty_target_costs_whole_query():
    assert myers_batch.distances(b"ACGT", [b""]) == [4]


def test_empty_batch_returns_empty_list():
    assert myers_batch.distances(b"ACGT", []) == []


def test_target_shorter_than_query():
    assert myers_batch.distances(b"ACGTACGT", [b"ACGT"]) == [dp_infix(b"ACGTACGT", b"ACGT")]


def test_case_insensitive_bases():
    assert myers_batch.distances(b"acgt", [b"TTACGTTT"]) == [0]


def test_unknown_bytes_match_only_themselves():
    # N in the query cannot be satisfied by A, so one substitution is required.
    assert myers_batch.distances(b"ANA", [b"AAA"]) == [1]
    assert myers_batch.distances(b"ANA", [b"ANA"]) == [0]


# ------------------------------------------------------------------ argument


@pytest.mark.parametrize("bad", [b"", b"A" * 65, b"A" * 200])
def test_query_length_out_of_range(bad):
    with pytest.raises(ValueError, match="query length must be 1..64"):
        myers_batch.distances(bad, [b"ACGT"])


def test_query_at_max_length_is_accepted():
    q = b"ACGT" * 16
    assert len(q) == myers_batch.MAX_QUERY
    assert myers_batch.distances(q, [q]) == [0]


def test_targets_must_be_a_sequence():
    with pytest.raises(TypeError):
        myers_batch.distances(b"ACGT", 5)


def test_target_items_must_be_bytes_like():
    with pytest.raises(TypeError):
        myers_batch.distances(b"ACGT", ["ACGT"])


# ------------------------------------------------------- oracles, systematic


@pytest.mark.parametrize("m", [1, 2, 3, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64])
def test_matches_dp_reference_across_query_lengths(m):
    rng = random.Random(1000 + m)
    query = rand_seq(rng, m)
    targets = [rand_seq(rng, n) for n in (1, m, m + 1, 2 * m + 5, 90)]
    expected = [dp_infix(query, t) for t in targets]
    assert myers_batch.distances(query, targets) == expected


@pytest.mark.parametrize("batch", list(range(1, 18)) + [63, 64, 65])
def test_batch_sizes_around_lane_boundaries(batch):
    """Exercises full 8-wide blocks plus every possible scalar remainder."""
    rng = random.Random(7 * batch)
    query = rand_seq(rng, 21)
    targets = [rand_seq(rng, rng.randint(1, 60)) for _ in range(batch)]
    expected = [dp_infix(query, t) for t in targets]
    assert myers_batch.distances(query, targets) == expected


def test_ragged_lengths_within_one_block():
    """Lanes finish at different times, so the per-lane scalar tail must run."""
    rng = random.Random(99)
    query = rand_seq(rng, 19)
    targets = [rand_seq(rng, n) for n in (1, 2, 5, 13, 40, 41, 100, 250)]
    expected = [dp_infix(query, t) for t in targets]
    assert myers_batch.distances(query, targets) == expected


def test_scalar_and_widest_paths_agree():
    rng = random.Random(4242)
    query = rand_seq(rng, 33)
    targets = [rand_seq(rng, rng.randint(1, 200)) for _ in range(200)]
    assert myers_batch.distances(query, targets) == myers_batch.distances_scalar(query, targets)


# ------------------------------------------------------------- vs edlib


def edlib_infix(query: bytes, target: bytes) -> int:
    return edlib.align(query, target, mode="HW", task="distance")["editDistance"]


def test_differential_against_edlib_random():
    rng = random.Random(20260812)
    for _ in range(400):
        m = rng.randint(1, 64)
        query = rand_seq(rng, m)
        targets = []
        for _ in range(rng.randint(1, 11)):
            n = rng.randint(1, 150)
            t = bytearray(rand_seq(rng, n))
            if n >= m and rng.random() < 0.6:  # plant an approximate occurrence
                pos = rng.randint(0, n - m)
                t[pos : pos + m] = query
                for _ in range(rng.randint(0, 3)):
                    t[rng.randrange(n)] = rng.choice(ALPHABET)
            targets.append(bytes(t))
        expected = [edlib_infix(query, t) for t in targets]
        assert myers_batch.distances(query, targets) == expected


def test_differential_against_edlib_adapter():
    """The workload the library exists for: one adapter against many reads."""
    rng = random.Random(5)
    adapter = b"AGATCGGAAGAGCACACGTCTGAACTCCAGTCA"
    reads = []
    for _ in range(500):
        r = bytearray(rand_seq(rng, 150))
        if rng.random() < 0.7:
            pos = rng.randint(0, 150 - len(adapter))
            r[pos : pos + len(adapter)] = adapter
            for _ in range(rng.randint(0, 4)):
                r[rng.randrange(150)] = rng.choice(ALPHABET)
        reads.append(bytes(r))
    assert myers_batch.distances(adapter, reads) == [edlib_infix(adapter, r) for r in reads]


# ------------------------------------------------------------------ plumbing


def test_have_neon_matches_platform():
    import platform

    expected = platform.machine() in {"arm64", "aarch64"}
    assert myers_batch.have_neon() is expected
    assert myers_batch.lanes() == (8 if expected else 1)


def test_accepts_memoryview_and_bytearray():
    targets = [bytearray(b"TTACGTTT"), memoryview(b"ACGT")]
    assert myers_batch.distances(b"ACGT", targets) == [0, 0]


def test_releases_gil_so_threads_make_progress():
    from concurrent.futures import ThreadPoolExecutor

    rng = random.Random(11)
    query = rand_seq(rng, 25)
    shards = [[rand_seq(rng, 120) for _ in range(500)] for _ in range(4)]
    serial = [myers_batch.distances(query, s) for s in shards]
    with ThreadPoolExecutor(max_workers=4) as ex:
        threaded = list(ex.map(lambda s: myers_batch.distances(query, s), shards))
    assert threaded == serial
