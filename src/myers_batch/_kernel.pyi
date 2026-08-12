from collections.abc import Sequence

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
def simd_backend() -> str: ...
