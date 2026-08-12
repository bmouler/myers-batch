# Changelog

## [1.0.0] - 2026-08-12

First stable release.

- Batched infix edit distance using bit-parallel Myers on aarch64 NEON and x86-64 AVX2, 7.5–10× faster than edlib single-threaded on the reference machine.
- Added a runtime-dispatched AVX2 kernel alongside the NEON and portable scalar kernels.
- Added a deterministic Hypothesis property-based suite and differential checks against the scalar implementation and edlib.
- Adopted strict mypy checking for the typed Python API.
- Expanded CI across native aarch64 and x86-64 Linux and macOS, covering Python 3.9 and 3.11–3.13 and enforcing the active SIMD backend.
