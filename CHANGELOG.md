# Changelog

## [Unreleased]

## [1.0.1] - 2026-08-15

- Grouped ragged targets by length before SIMD execution and bypassed grouping for uniform and empty batches, preserving scalar-equivalent results.
- Added a deterministic mixed-length end-to-end benchmark and retained safe snapshots for mutable buffer exporters.


## [1.0.0] - 2026-08-12

First stable release.

- Batched infix edit distance using bit-parallel Myers on aarch64 NEON and x86-64 AVX2, 7.5–10× faster than edlib single-threaded on the reference machine.
- Added a runtime-dispatched AVX2 kernel alongside the NEON and portable scalar kernels.
- Added a deterministic Hypothesis property-based suite and differential checks against the scalar implementation and edlib.
- Adopted strict mypy checking for the typed Python API.
- Expanded CI across native aarch64 and x86-64 Linux and macOS, covering Python 3.9 and 3.11–3.13 and enforcing the active SIMD backend.
