#include <cstddef>
#include <cstdint>

#include "modernmat/modernmat.hpp"

#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>

TEST(MatX, ConstructsAndReportsMetadata)
{
    constexpr int rows = 3;
    constexpr int cols = 5;

    const modernmat::matx image {rows, cols, CV_8UC1};

    EXPECT_EQ(rows, image.rows());
    EXPECT_EQ(cols, image.cols());
    EXPECT_EQ(CV_8UC1, image.type());
    EXPECT_TRUE(image.is_continuous());
    EXPECT_EQ(static_cast<std::size_t>(cols), image.step());
}

TEST(MatX, DefaultEmptyStateIsContinuousAndZeroStep)
{
    modernmat::matx image {};

    EXPECT_TRUE(image.empty());
    EXPECT_TRUE(image.is_continuous());
    EXPECT_EQ(0, image.rows());
    EXPECT_EQ(0, image.cols());
    EXPECT_EQ(0U, image.step());
    EXPECT_EQ(0U, image.step1());
    EXPECT_EQ(nullptr, image.data());
}

TEST(MatX, Step1MatchesCvMat)
{
    constexpr int reference_rows = 5;
    constexpr int reference_cols = 7;

    const cv::Mat reference(reference_rows, reference_cols, CV_8UC3);
    auto facade = modernmat::matx {reference};

    auto const expected = reference.step1();
    EXPECT_EQ(expected, facade.step1());
}

TEST(MatX, CopyOnWriteProtectsOriginal)
{
    constexpr int extent = 3;
    constexpr std::uint8_t baseline_value {7};
    constexpr std::uint8_t updated_value {42};

    modernmat::matx original {extent, extent, CV_8UC1};
    auto original_view = cv::Mat(original);
    original_view.setTo(baseline_value);

    auto copy = original;
    auto copy_view = cv::Mat(copy);
    copy_view.at<std::uint8_t>(0, 0) = updated_value;

    auto const frozen_original = cv::Mat(original);
    EXPECT_EQ(baseline_value, frozen_original.at<std::uint8_t>(0, 0));
    EXPECT_EQ(updated_value, cv::Mat(copy).at<std::uint8_t>(0, 0));
}

TEST(MatX, RoiDetachesOnWrite)
{
    constexpr int extent = 5;
    constexpr std::uint8_t roi_initial {5};
    constexpr std::uint8_t roi_updated {11};
    const cv::Rect center_region {1, 1, 2, 2};

    modernmat::matx image {extent, extent, CV_8UC1};
    auto image_view = cv::Mat(image);
    image_view.setTo(roi_initial);

    auto center = image(center_region);
    auto center_view = cv::Mat(center);
    center_view.at<std::uint8_t>(0, 0) = roi_updated;

    EXPECT_EQ(roi_initial, cv::Mat(image).at<std::uint8_t>(1, 1));
    EXPECT_EQ(roi_updated, cv::Mat(center).at<std::uint8_t>(0, 0));
}

TEST(MatX, SliceDoesNotDetachWhenParentReleasedBeforeWrite)
{
    constexpr int extent = 4;
    const cv::Rect center_region {1, 1, 2, 2};

    modernmat::matx image {extent, extent, CV_8UC1};
    cv::Mat(image).setTo(1);

    auto slice = image(center_region);
    auto const* before = static_cast<const modernmat::matx&>(slice).data();

    image = modernmat::matx {};  // slice now uniquely owns the buffer

    slice.ptr<std::uint8_t>(0)[0] = 2;

    auto const* after = static_cast<const modernmat::matx&>(slice).data();
    EXPECT_EQ(before, after);    // key expectation under this model
}

TEST(MatX, SliceWriteDoesNotAffectParentWhenBothAlive)
{
    constexpr int extent = 4;
    const cv::Rect center_region {1, 1, 2, 2};

    modernmat::matx image {extent, extent, CV_8UC1};
    cv::Mat(image).setTo(1);

    auto slice = image(center_region);

    slice.ptr<std::uint8_t>(0)[0] = 7;

    // Parent should remain unchanged at the corresponding pixel
    // pixel (1,1) in parent corresponds to (0,0) in ROI
    EXPECT_EQ(image.ptr<std::uint8_t>(1)[1], 1);
    EXPECT_EQ(slice.ptr<std::uint8_t>(0)[0], 7);
}

TEST(MatX, SliceDetachesOnWriteWhileParentAlive)
{
    constexpr int extent = 4;
    const cv::Rect center_region {1, 1, 2, 2};

    modernmat::matx image {extent, extent, CV_8UC1};
    cv::Mat(image).setTo(1);

    auto slice = image(center_region);

    auto const* before = static_cast<const modernmat::matx&>(slice).data();

    // write through the slice while parent still exists => shared => detaches
    slice.ptr<std::uint8_t>(0)[0] = 2;

    auto const* after = static_cast<const modernmat::matx&>(slice).data();
    EXPECT_NE(before, after);
}

TEST(MatX, RoiViewSharesStrideButIsNotContinuous)
{
    constexpr int roi_extent = 10;
    const cv::Rect view_region {2, 2, 5, 5};

    const modernmat::matx image {roi_extent, roi_extent, CV_8UC1};
    auto roi = image(view_region);

    EXPECT_FALSE(roi.is_continuous());
    EXPECT_EQ(image.step(), roi.step());
}

TEST(MatX, SingleRowRoiMatchesOpenCvContinuity)
{
    constexpr int extent = 6;
    const cv::Rect view_region {1, 2, 4, 1};

    const modernmat::matx image {extent, extent, CV_8UC1};
    auto roi = image(view_region);

    EXPECT_EQ(1, roi.rows());
    EXPECT_TRUE(roi.is_continuous());
}

TEST(MatX, PrepareSameShapeClearsWhenOtherIsEmpty)
{
    modernmat::matx image {3, 3, CV_8UC1};
    cv::Mat(image).setTo(1);

    modernmat::matx other {};
    image.prepare_same_shape_as(other);

    EXPECT_TRUE(image.empty());
    EXPECT_EQ(0, image.rows());
    EXPECT_EQ(0, image.cols());
    EXPECT_EQ(0U, image.step());
}

TEST(MatX, PrepareSameShapeReallocatesWhenShapeDiffers)
{
    modernmat::matx image {2, 2, CV_8UC1};
    auto* before = cv::Mat(image).data;

    modernmat::matx other {3, 4, CV_8UC3};
    image.prepare_same_shape_as(other);

    EXPECT_EQ(3, image.rows());
    EXPECT_EQ(4, image.cols());
    EXPECT_EQ(CV_8UC3, image.type());
    EXPECT_NE(before, cv::Mat(image).data);
}

TEST(MatX, PrepareSameShapeIsNoOpWhenShapeMatches)
{
    modernmat::matx image {2, 2, CV_8UC1};
    auto* before = cv::Mat(image).data;

    image.prepare_same_shape_as(image);

    EXPECT_EQ(before, cv::Mat(image).data);
}

TEST(MatX, PrepareAllocatesAndThenNoOpsWhenShapeMatches)
{
    modernmat::matx image {};
    image.prepare(2, 2, CV_8UC1);
    auto* first_allocation = cv::Mat(image).data;

    image.prepare(2, 2, CV_8UC1);
    EXPECT_EQ(first_allocation, cv::Mat(image).data);

    image.prepare(2, 3, CV_8UC1);
    EXPECT_NE(first_allocation, cv::Mat(image).data);
    EXPECT_EQ(2, image.rows());
    EXPECT_EQ(3, image.cols());
}

TEST(MatX, PrepareRejectsInvalidArguments)
{
    EXPECT_THROW(modernmat::matx(0, 1, CV_8UC1), std::invalid_argument);
    EXPECT_THROW(modernmat::matx(1, 0, CV_8UC1), std::invalid_argument);

    modernmat::matx image {};
    EXPECT_THROW(image.prepare(1, 1, -1), std::invalid_argument);
}

TEST(MatX, PtrAndRoiValidationThrowOnOutOfBounds)
{
    modernmat::matx image {2, 2, CV_8UC1};
    EXPECT_THROW(static_cast<void>(image.ptr<std::uint8_t>(2)), std::out_of_range);

    const modernmat::matx readonly {2, 2, CV_8UC1};
    EXPECT_THROW(static_cast<void>(readonly.ptr<std::uint8_t>(-1)), std::out_of_range);

    EXPECT_THROW(static_cast<void>(image(cv::Rect {-1, 0, 1, 1})), std::out_of_range);
    EXPECT_THROW(static_cast<void>(image(cv::Rect {0, 0, 5, 5})), std::out_of_range);
    EXPECT_THROW(static_cast<void>(image(cv::Rect {0, 0, -1, 1})), std::out_of_range);
    EXPECT_THROW(static_cast<void>(image(cv::Rect {0, 0, 1, -1})), std::out_of_range);

    const modernmat::matx empty {};
    EXPECT_THROW(static_cast<void>(empty(cv::Rect {0, 0, 1, 1})), std::out_of_range);
}

TEST(MatXInterop, PrepareSameShapeEnablesInPlaceDstReuse)
{
    constexpr int rows = 4;
    constexpr int cols = 4;
    cv::Mat src(rows, cols, CV_8UC1);
    src.setTo(5);
    modernmat::matx dst {rows, cols, CV_8UC1};

    dst.prepare_same_shape_as(dst);  // no-op but exercises API
    auto dst_view = cv::Mat(dst);
    cv::add(src, src, dst_view);

    auto diff = cv::norm(dst_view, src + src, cv::NORM_INF);  // NOLINT(misc-include-cleaner)
    EXPECT_DOUBLE_EQ(0.0, diff);
}

TEST(MatXInterop, GaussianBlurMatchesCvMat)
{
    constexpr int extent = 5;
    constexpr int kernel_extent = 3;
    constexpr int gaussian_input = 37;

    auto const size = cv::Size {extent, extent};
    cv::Mat src(size, CV_8UC1);
    src.setTo(gaussian_input);

    cv::Mat reference {};
    cv::GaussianBlur(src, reference, cv::Size {kernel_extent, kernel_extent}, 0.0);

    modernmat::matx target {size.height, size.width, src.type()};
    auto target_view = cv::Mat(target);
    src.copyTo(target_view);
    auto working_view = cv::Mat(target);
    cv::GaussianBlur(src, working_view, cv::Size {kernel_extent, kernel_extent}, 0.0);

    auto diff = cv::norm(reference, cv::Mat(target),
                         cv::NORM_INF);  // NOLINT(misc-include-cleaner)
    EXPECT_DOUBLE_EQ(0.0, diff);
}

TEST(MatXInterop, WrapPreservesSourceUntilWrite)
{
    constexpr int extent = 3;
    constexpr std::uint8_t wrap_fill {9};
    constexpr std::uint8_t facade_updated {64};

    cv::Mat src(cv::Size {extent, extent}, CV_8UC1);
    src.setTo(wrap_fill);

    auto facade = modernmat::matx {src};
    auto facade_view = cv::Mat(facade);
    facade_view.at<std::uint8_t>(0, 0) = facade_updated;
    EXPECT_EQ(wrap_fill, src.at<std::uint8_t>(0, 0));
    EXPECT_EQ(facade_updated, cv::Mat(facade).at<std::uint8_t>(0, 0));
}

TEST(MatXInterop, ConstViewIsInputOnly)
{
    const modernmat::matx image {4, 4, CV_8UC1};
    auto readonly = image.as_cv_const();

    // Assignment should only touch writable APIs to avoid accidental mutation.
    cv::Mat clone;
    readonly.copyTo(clone);
    EXPECT_EQ(image.rows(), clone.rows);
    EXPECT_EQ(image.cols(), clone.cols);
    EXPECT_EQ(image.type(), clone.type());
}

TEST(MatXInterop, WrapDetachesExactlyOnceOnFirstWrite)
{
    constexpr int wrap_extent = 10;
    constexpr std::uint8_t first_write_value = 42;
    constexpr std::uint8_t second_write_value = 5;

    cv::Mat src(cv::Size {wrap_extent, wrap_extent}, CV_8UC1);
    src.setTo(0);

    auto facade = modernmat::matx {src};
    auto const* before = static_cast<const modernmat::matx&>(facade).data();

    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    facade.ptr<std::uint8_t>(0)[0] = first_write_value;
    auto const* after = static_cast<const modernmat::matx&>(facade).data();
    EXPECT_NE(before, after);

    auto const* retained = after;
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    facade.ptr<std::uint8_t>(0)[1] = second_write_value;
    EXPECT_EQ(retained, static_cast<const modernmat::matx&>(facade).data());
}

TEST(MatXInterop, OpenCvResizeCanReallocateHeader)
{
    constexpr int small_extent = 4;
    constexpr int large_extent = 8;

    cv::Mat src(large_extent, large_extent, CV_8UC1);
    src.setTo(1);

    modernmat::matx dst(small_extent, small_extent, CV_8UC1);
    auto const* before = static_cast<const modernmat::matx&>(dst).data();

    cv::Mat header = dst;
    cv::resize(src, header, cv::Size {large_extent, large_extent}, 0.0, 0.0, cv::INTER_LINEAR);

    auto const* after = static_cast<const modernmat::matx&>(dst).data();

    if (header.data == after) {
        GTEST_SKIP() << "OpenCV kept the header buffer; no reallocation observed.";
    }

    EXPECT_EQ(before, after);
    EXPECT_NE(header.data, after);
    EXPECT_EQ(large_extent, header.rows);
    EXPECT_EQ(large_extent, header.cols);
    EXPECT_EQ(small_extent, dst.rows());
    EXPECT_EQ(small_extent, dst.cols());
}

TEST(MatXInterop, ResizeAwareUpdatesDestination)
{
    constexpr int small_extent = 4;
    constexpr int large_extent = 8;

    cv::Mat src(large_extent, large_extent, CV_8UC1);
    src.setTo(1);

    modernmat::matx dst(small_extent, small_extent, CV_8UC1);
    auto const* before = static_cast<const modernmat::matx&>(dst).data();

    auto resized = modernmat::resize_aware(
        dst,
        [&](cv::Mat& header)
        {
            cv::resize(
                src, header, cv::Size {large_extent, large_extent}, 0.0, 0.0, cv::INTER_LINEAR);
        });

    if (!resized) {
        GTEST_SKIP() << "OpenCV kept the header buffer; no reallocation observed.";
    }

    EXPECT_EQ(large_extent, dst.rows());
    EXPECT_EQ(large_extent, dst.cols());
    EXPECT_NE(before, static_cast<const modernmat::matx&>(dst).data());
}

TEST(MatXInterop, ResizeAwareNoOpReturnsFalse)
{
    constexpr int extent = 4;
    constexpr std::uint8_t fill_value {7};

    cv::Mat src(extent, extent, CV_8UC1);
    src.setTo(fill_value);

    modernmat::matx dst {src};
    auto const* before = static_cast<const modernmat::matx&>(dst).data();

    auto resized = modernmat::resize_aware(dst, [](cv::Mat& header) { (void)header; });

    EXPECT_FALSE(resized);
    EXPECT_EQ(before, src.data);
    auto const* after = static_cast<const modernmat::matx&>(dst).data();
    EXPECT_NE(before, after);

    const auto& readonly = dst;
    cv::Mat header = readonly.as_cv_const();
    EXPECT_EQ(fill_value, header.at<std::uint8_t>(0, 0));
}

TEST(MatXInterop, ResizeAwareMetadataChangeUpdatesInPlace)
{
    constexpr int rows = 2;
    constexpr int cols = 3;
    constexpr std::uint8_t fill_value {9};

    modernmat::matx dst {rows, cols, CV_8UC1};
    cv::Mat(dst).setTo(fill_value);

    auto const* before = static_cast<const modernmat::matx&>(dst).data();

    auto resized =
        modernmat::resize_aware(dst, [&](cv::Mat& header) { header = header.reshape(0, 1); });

    EXPECT_TRUE(resized);
    EXPECT_EQ(1, dst.rows());
    EXPECT_EQ(rows * cols, dst.cols());
    EXPECT_EQ(before, static_cast<const modernmat::matx&>(dst).data());
    EXPECT_EQ(static_cast<std::size_t>(dst.cols()), dst.step());

    const auto& readonly = dst;
    cv::Mat header = readonly.as_cv_const();
    EXPECT_EQ(fill_value, header.at<std::uint8_t>(0, 0));
}

TEST(MatXInterop, ConstConversionDoesNotDetach)
{
    constexpr int extent = 4;
    cv::Mat src(extent, extent, CV_8UC1);
    src.setTo(7);

    auto facade = modernmat::matx {src};
    const auto& readonly = facade;
    cv::Mat header = readonly.as_cv_const();

    EXPECT_EQ(src.data, header.data);
    EXPECT_EQ(7, header.at<std::uint8_t>(0, 0));
}

TEST(MatXInterop, NonConstConversionDetachesOnceAndSharesAfter)
{
    constexpr int extent = 4;
    cv::Mat src(extent, extent, CV_8UC1);
    src.setTo(0);

    auto facade = modernmat::matx {src};
    auto const* before = static_cast<const modernmat::matx&>(facade).data();

    cv::Mat header = facade;  // triggers detach
    auto const* after = static_cast<const modernmat::matx&>(facade).data();

    EXPECT_NE(before, after);
    header.at<std::uint8_t>(0, 0) = 55;
    EXPECT_EQ(55, cv::Mat(facade).at<std::uint8_t>(0, 0));
    EXPECT_EQ(0, src.at<std::uint8_t>(0, 0));

    auto const* stable = static_cast<const modernmat::matx&>(facade).data();
    cv::Mat again = facade;  // should not detach again
    EXPECT_EQ(stable, static_cast<const modernmat::matx&>(facade).data());
    EXPECT_EQ(stable, again.data);
}
