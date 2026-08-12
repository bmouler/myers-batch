from collections.abc import Sequence
from typing import Literal

MAX_QUERY: int

def distances(
    query: bytes | bytearray | memoryview,
    targets: Sequence[bytes | bytearray | memoryview],
    /,
) -> list[int]: ...
def distances_scalar(
    query: bytes | bytearray | memoryview,
    targets: Sequence[bytes | bytearray | memoryview],
    /,
) -> list[int]: ...
def have_neon() -> bool: ...
def simd_backend() -> Literal["neon", "avx2", "scalar"]: ...
