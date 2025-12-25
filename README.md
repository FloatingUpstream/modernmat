# modernmat

This is the modernmat project.

# Development prerequisites

- OpenCV (components `core` and `imgproc`) must be available on your system; CMake fails fast if it cannot locate them.
- Install [pre-commit](https://pre-commit.com/) and run `pre-commit install` to activate the `clang-format` hook.
- Run a clang-tidy sweep with the CI profile: `cmake --preset=ci-ubuntu && cmake --build --preset=ci-ubuntu`.

## MatX Copy-on-Write Model

- Copies alias until first write: copy/move just share `m_owner` and propagate `m_detach_on_write`.
- Mutable gateways (`data()`, `ptr<T>()`, non-const `operator cv::Mat()`) call `detach()`; const accessors never detach.
- Constructing `matx` from a `cv::Mat` marks the view for detach-on-first-write, then normal CoW rules apply. `detach()` only clones when ownership is shared or the flag is set, and clears the flag afterwards.
- Const OpenCV views (`as_cv_const()`) still return writable headers—treat them as read-only.
- ROI slices (`slice(Rect)` / `operator()(Rect)`) share the parent owner/stride, rebase `m_data`,
  and detach on first write; they recompute `m_contiguous_stride` like any other MatX view.
- Buffer allocation uses standard `new`/`delete` by default; defining `MODERNMAT_USE_OPENCV_ALLOCATOR` swaps in OpenCV’s `fastMalloc`/`fastFree` for aligned storage.

### Interop Caveats

- APIs that require `cv::Mat&` (non-const) cannot bind to temporaries; create a named header (for example, `cv::Mat header = image;`) so it lives long enough. Use `auto header = image.as_cv_const();` when you only have a const `matx`.
- APIs that resize/allocate the destination must either call `prepare_same_shape_as()` before invocation, or run into a scratch `cv::Mat` and construct a `matx` from it once the OpenCV call completes.
- Use `resize_aware(matx&, Fn)` when calling resize-capable APIs; it updates the `matx` in place and returns `true` if OpenCV reallocates the header.
- See `example/reallocation.cpp` for a concrete demonstration of the reallocation edge case.

# Building and installing

See the [BUILDING](BUILDING.md) document.

# Release policy

This project does not publish formal releases. Consumers should pin to a
specific commit (or vendored snapshot) for reproducible builds.

# Licensing

This project is released under the Unlicense. See `UNLICENSE`.
