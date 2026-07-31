#ifndef MODERNMAT_MODERNMAT_HPP
#define MODERNMAT_MODERNMAT_HPP

#include <cstddef>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>

#include <opencv2/core.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

namespace modernmat
{
//------------------------------ MatX facade ------------------------------
// Header-only wrapper around cv::Mat semantics with copy-on-write guarantees.
//------------------------------------------------------------------------
// Toggle at compile time: define MODERNMAT_USE_OPENCV_ALLOCATOR to 1 to use
// cv::fastMalloc / cv::fastFree for MatX storage.
//------------------------------------------------------------------------
#ifndef MODERNMAT_USE_OPENCV_ALLOCATOR
#    define MODERNMAT_USE_OPENCV_ALLOCATOR 0  // NOLINT(cppcoreguidelines-macro-usage)
#endif

class matx
{
  public:
    using byte = unsigned char;

    template<typename Fn>
    friend auto resize_aware(matx& destination, Fn&& fn) -> bool;

    matx() noexcept = default;

    matx(int rows, int cols, int type)
        : matx(cv::Size {cols, rows}, type)
    {
    }

    explicit matx(cv::Size size, int type) { allocate(size, type); }

    explicit matx(const cv::Mat& source)
    {
        if (source.empty()) {
            return;
        }

        if (source.u == nullptr) {
            auto copy = source.clone();
            auto keeper = std::make_shared<cv::Mat>(std::move(copy));
            m_owner = std::shared_ptr<void>(keeper, keeper->data);
        } else {
            auto keeper = std::make_shared<cv::Mat>(source);
            m_owner = std::shared_ptr<void>(keeper, keeper->data);
        }

        m_data = static_cast<byte*>(m_owner.get());
        m_stride = static_cast<std::size_t>(source.step);
        m_type = source.type();
        m_rows = source.rows;
        m_cols = source.cols;
        m_contiguous_stride = checked_row_stride(m_cols, element_size());
        m_detach_on_write = true;
    }

    matx(const matx&) = default;
    matx(matx&&) noexcept = default;
    auto operator=(const matx&) -> matx& = default;
    auto operator=(matx&&) noexcept -> matx& = default;
    ~matx() = default;

    [[nodiscard]] auto rows() const noexcept -> int { return m_rows; }

    [[nodiscard]] auto cols() const noexcept -> int { return m_cols; }

    [[nodiscard]] auto size() const noexcept -> cv::Size { return {m_cols, m_rows}; }

    [[nodiscard]] auto type() const noexcept -> int { return m_type; }

    [[nodiscard]] auto channels() const noexcept -> int { return CV_MAT_CN(m_type); }

    [[nodiscard]] auto depth() const noexcept -> int { return CV_MAT_DEPTH(m_type); }

    [[nodiscard]] auto empty() const noexcept -> bool
    {
        return m_rows == 0 || m_cols == 0 || m_data == nullptr;
    }

    [[nodiscard]] auto step() const noexcept -> std::size_t { return m_stride; }

    [[nodiscard]] auto step1() const -> std::size_t
    {
        if (empty()) {
            return 0;
        }
        return m_stride / element_size1();
    }

    [[nodiscard]] auto is_continuous() const noexcept -> bool
    {
        if (empty()) {
            return true;
        }
        if (m_rows == 1) {
            return true;
        }
        return m_stride == contiguous_stride();
    }

    // Writable access must ensure unique ownership first.
    [[nodiscard]] auto data() -> byte*
    {
        detach();
        return m_data;
    }

    // Const access never detaches; treat the pointer as read-only.
    [[nodiscard]] auto data() const noexcept -> const byte* { return m_data; }

    // NOLINTNEXTLINE(google-explicit-constructor,hicpp-explicit-conversions)
    operator cv::Mat()
    {
        detach();
        // Returned header will not track reallocations; pre-size or use a
        // temporary cv::Mat and construct a matx after the call when APIs resize.
        return {m_rows, m_cols, m_type, m_data, m_stride};
    }

    // Contract-only read view for const MatX instances. cv::Mat cannot encode
    // immutable data, so callers must not mutate the returned header.
    [[nodiscard]] auto as_cv_const() const -> cv::Mat
    {
        return {m_rows, m_cols, m_type, const_cast<byte*>(m_data), m_stride};
    }

    // Const-safe OpenCV value for APIs that may mutate their input header/data.
    [[nodiscard]] auto to_cv_mat_copy() const -> cv::Mat { return as_cv_const().clone(); }

    template<typename T>
    [[nodiscard]] auto ptr(int row) -> T*
    {
        detach();
        ensure_row(row);
        auto const offset = static_cast<std::size_t>(row) * m_stride;
        return reinterpret_cast<T*>(m_data + offset);  // NOLINT
    }

    template<typename T>
    [[nodiscard]] auto ptr(int row) const -> const T*
    {
        ensure_row(row);
        auto const offset = static_cast<std::size_t>(row) * m_stride;
        return reinterpret_cast<const T*>(m_data + offset);  // NOLINT
    }

    [[nodiscard]] auto clone() const -> matx
    {
        matx copy {};
        if (empty()) {
            return copy;
        }

        copy.allocate(cv::Size {m_cols, m_rows}, m_type);
        auto source = cv::Mat(m_rows, m_cols, m_type, m_data, m_stride);
        auto destination =
            cv::Mat(copy.m_rows, copy.m_cols, copy.m_type, copy.m_data, copy.m_stride);
        source.copyTo(destination);
        copy.m_detach_on_write = false;

        return copy;
    }

    // slice() is an alias until a write occurs while the buffer is shared; then it detaches
    [[nodiscard]] auto slice(const cv::Rect& region) const -> matx
    {
        validate_roi(region);

        // CoW slice aliases storage and detaches on first write.
        auto view = matx {};
        view.m_owner = m_owner;
        const auto header = cv::Mat(m_rows, m_cols, m_type, m_data, m_stride);
        const auto roi_header = cv::Mat(header, region);
        view.m_data = roi_header.data;
        view.m_stride = roi_header.step;
        view.m_rows = roi_header.rows;
        view.m_cols = roi_header.cols;
        view.m_type = roi_header.type();
        view.m_contiguous_stride = static_cast<std::size_t>(roi_header.cols) * view.element_size();
        // Slices always detach on first write to avoid mutating the parent.
        view.m_detach_on_write = m_detach_on_write;

        return view;
    }

    [[nodiscard]] auto operator()(const cv::Rect& region) const -> matx { return slice(region); }

    void prepare_same_shape_as(const matx& other)
    {
        if (other.empty()) {
            m_owner.reset();
            m_data = nullptr;
            m_stride = 0;
            m_rows = 0;
            m_cols = 0;
            m_type = 0;
            m_contiguous_stride = 0;
            return;
        }

        if (m_rows == other.m_rows && m_cols == other.m_cols && m_type == other.m_type) {
            return;
        }

        allocate(cv::Size {other.m_cols, other.m_rows}, other.m_type);
    }

    void prepare(int rows, int cols, int type)
    {
        if (m_rows == rows && m_cols == cols && m_type == type && m_data != nullptr) {
            return;
        }
        allocate(cv::Size {cols, rows}, type);
    }

  private:
    std::shared_ptr<void> m_owner;
    byte* m_data = nullptr;
    std::size_t m_stride = 0;
    int m_rows = 0;
    int m_cols = 0;
    int m_type = 0;
    std::size_t m_contiguous_stride = 0;
    bool m_detach_on_write = false;

    [[nodiscard]] static auto checked_row_stride(int cols, std::size_t element_bytes) -> std::size_t
    {
        auto const width = static_cast<std::size_t>(cols);
        if (element_bytes != 0 && width > std::numeric_limits<std::size_t>::max() / element_bytes) {
            throw std::overflow_error("row byte count overflows size_t");
        }
        return width * element_bytes;
    }

    [[nodiscard]] static auto checked_total_bytes(std::size_t stride, int rows) -> std::size_t
    {
        auto const height = static_cast<std::size_t>(rows);
        if (stride != 0 && height > std::numeric_limits<std::size_t>::max() / stride) {
            throw std::overflow_error("image byte count overflows size_t");
        }
        return stride * height;
    }

    [[nodiscard]] auto contiguous_stride() const noexcept -> std::size_t
    {
        return m_contiguous_stride;
    }

    [[nodiscard]] auto element_size() const noexcept -> std::size_t
    {
        return static_cast<std::size_t>(CV_ELEM_SIZE(m_type));
    }

    [[nodiscard]] auto element_size1() const noexcept -> std::size_t
    {
        return static_cast<std::size_t>(CV_ELEM_SIZE1(m_type));
    }

    void ensure_row(int row) const
    {
        if (row < 0 || row >= m_rows) {
            throw std::out_of_range("Row index out of bounds");
        }
    }

    void validate_roi(const cv::Rect& region) const
    {
        if (empty()) {
            throw std::out_of_range("ROI requested on empty image");
        }
        if (region.x < 0 || region.y < 0 || region.width <= 0 || region.height <= 0
            || region.width > m_cols - region.x || region.height > m_rows - region.y)
        {
            throw std::out_of_range("ROI out of bounds");
        }
    }

    void allocate(cv::Size size, int type)
    {
        m_owner.reset();
        m_data = nullptr;
        m_stride = 0;
        m_rows = 0;
        m_cols = 0;
        m_type = 0;
        m_contiguous_stride = 0;
        m_detach_on_write = false;

        if (size.width <= 0 || size.height <= 0) {
            throw std::invalid_argument("rows and cols must be positive");
        }
        if (type < 0) {
            throw std::invalid_argument("type must be a valid OpenCV type");
        }

        m_type = type;
        m_rows = size.height;
        m_cols = size.width;
        m_contiguous_stride = checked_row_stride(m_cols, element_size());
        m_stride = m_contiguous_stride;
        auto const total_bytes = checked_total_bytes(m_contiguous_stride, m_rows);

        // NOLINTBEGIN
        if constexpr (MODERNMAT_USE_OPENCV_ALLOCATOR != 0) {
            auto* storage = cv::fastMalloc(total_bytes);
            if (storage == nullptr) {
                throw std::bad_alloc();
            }
            m_owner = std::shared_ptr<void>(
                storage, [](void* pointer) noexcept -> void { cv::fastFree(pointer); });
        } else {
            auto deleter = [](void* pointer) noexcept -> void { ::operator delete(pointer); };
            m_owner = std::shared_ptr<void>(::operator new(total_bytes), deleter);
        }
        // NOLINTEND
        m_data = static_cast<byte*>(m_owner.get());
        m_detach_on_write = false;
    }

    //----------------------------------------
    // CoW detach helper
    //----------------------------------------
    void detach()
    {
        if (!m_owner) {
            return;
        }

        auto const requires_unique = m_detach_on_write || m_owner.use_count() > 1;
        if (!requires_unique) {
            return;
        }

        auto unique_copy = clone();
        unique_copy.m_detach_on_write = false;
        *this = std::move(unique_copy);
    }
};

// Resize-aware helper for OpenCV APIs that may reallocate destination headers.
// Returns true when the destination buffer changes and the matx is updated.
template<typename Fn>
auto resize_aware(matx& destination, Fn&& fn) -> bool
{
    // Build the header first so any CoW detach happens before we snapshot metadata.
    cv::Mat header = destination;
    auto const* before_data = header.data;
    auto const* before_start = header.datastart;
    auto const* before_end = header.dataend;
    auto const before_rows = header.rows;
    auto const before_cols = header.cols;
    auto const before_type = header.type();
    auto const before_step = static_cast<std::size_t>(header.step);

    std::forward<Fn>(fn)(header);

    auto const data_changed = header.data != before_data;
    auto const meta_changed = header.rows != before_rows || header.cols != before_cols
        || header.type() != before_type || static_cast<std::size_t>(header.step) != before_step;

    if (!data_changed && !meta_changed) {
        return false;
    }

    auto const aliases_original =
        before_start != nullptr && header.data >= before_start && header.data < before_end;

    if (data_changed && !aliases_original) {
        destination = matx {header};
        return true;
    }

    // Preserve ownership when the header only changes metadata, or when it is
    // rebound to a sub-view of the original destination storage.
    destination.m_data = header.data;
    destination.m_rows = header.rows;
    destination.m_cols = header.cols;
    destination.m_type = header.type();
    destination.m_stride = static_cast<std::size_t>(header.step);
    destination.m_contiguous_stride =
        static_cast<std::size_t>(destination.m_cols) * destination.element_size();

    return true;
}

inline auto name() -> std::string
{
    return "modernmat";
}
}  // namespace modernmat

#endif  // MODERNMAT_MODERNMAT_HPP
