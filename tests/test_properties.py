"""Deterministic properties of the public batched-distance API."""

from hypothesis import given, settings
from hypothesis import strategies as st

from myers_batch import distances, distances_scalar

queries = st.binary(min_size=1, max_size=64)
target_batches = st.lists(st.binary(max_size=120), max_size=20)


@settings(max_examples=50)
@given(query=queries, targets=target_batches)
def test_simd_and_scalar_distances_agree(query: bytes, targets: list[bytes]) -> None:
    assert distances(query, targets) == distances_scalar(query, targets)


@given(query=queries)
def test_query_has_zero_distance_from_itself(query: bytes) -> None:
    assert distances(query, [query]) == [0]


@settings(max_examples=50)
@given(
    query=queries,
    prefix=st.binary(max_size=40),
    target=st.binary(max_size=120),
    suffix=st.binary(max_size=40),
)
def test_embedding_target_cannot_increase_infix_distance(
    query: bytes,
    prefix: bytes,
    target: bytes,
    suffix: bytes,
) -> None:
    embedded_distance = distances(query, [prefix + target + suffix])[0]
    original_distance = distances(query, [target])[0]

    assert embedded_distance <= original_distance
