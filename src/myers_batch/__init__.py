"""Batched infix edit distance, faster than edlib on the same data.

The public entry point is :func:`distances`, which computes, for one query and
many targets, the minimum edit distance between the full query and any substring
of each target. Those are the semantics of ``edlib.align(..., mode="HW",
task="distance")`` and of adapter, primer, barcode and probe search.

Example
-------
>>> import myers_batch
>>> myers_batch.distances(b"ACGTACGT", [b"TTACGTACGTTT", b"TTTTTTTT"])
[0, 6]
"""

from __future__ import annotations

from ._kernel import MAX_QUERY, distances, distances_scalar, have_neon

__all__ = ["MAX_QUERY", "distances", "distances_scalar", "have_neon", "lanes"]

__version__ = "0.1.0"


def lanes() -> int:
    """Number of alignments the compiled kernel advances per inner iteration."""
    return 8 if have_neon() else 1
