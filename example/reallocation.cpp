#include <cstdint>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "modernmat/modernmat.hpp"

namespace
{

constexpr int small_extent = 4;
constexpr int large_extent = 8;
constexpr std::uint8_t fill_value = 42;
constexpr double no_scale = 0.0;

auto show_reallocation_caveat() -> void
{
    cv::Mat src(large_extent, large_extent, CV_8UC1, cv::Scalar(fill_value));
    modernmat::matx dst(small_extent, small_extent, CV_8UC1);

    auto const* before = static_cast<const modernmat::matx&>(dst).data();
    cv::Mat header = dst;

    // Resizing with a mismatched destination lets OpenCV reallocate the header.
    cv::resize(
        src, header, cv::Size {large_extent, large_extent}, no_scale, no_scale, cv::INTER_LINEAR);

    auto const* after = static_cast<const modernmat::matx&>(dst).data();

    std::cout << "Header size: " << header.rows << "x" << header.cols << '\n';
    std::cout << "MatX size: " << dst.rows() << "x" << dst.cols() << '\n';
    std::cout << "Header data: " << static_cast<const void*>(header.data) << '\n';
    std::cout << "MatX data (before): " << static_cast<const void*>(before) << '\n';
    std::cout << "MatX data (after): " << static_cast<const void*>(after) << '\n';

    if (header.data != after) {
        std::cout << "OpenCV reallocated the header; MatX still points at the old buffer.\n";
    }
}

auto show_safe_pattern() -> void
{
    cv::Mat src(large_extent, large_extent, CV_8UC1, cv::Scalar(fill_value));
    cv::Mat tmp;

    cv::resize(
        src, tmp, cv::Size {large_extent, large_extent}, no_scale, no_scale, cv::INTER_LINEAR);

    modernmat::matx wrapped {tmp};
    std::cout << "Wrapped size: " << wrapped.rows() << "x" << wrapped.cols() << '\n';
}

auto show_resize_aware() -> void
{
    cv::Mat src(large_extent, large_extent, CV_8UC1, cv::Scalar(fill_value));
    modernmat::matx dst(small_extent, small_extent, CV_8UC1);

    auto resized = modernmat::resize_aware(dst,
                                           [&](cv::Mat& header)
                                           {
                                               cv::resize(src,
                                                          header,
                                                          cv::Size {large_extent, large_extent},
                                                          no_scale,
                                                          no_scale,
                                                          cv::INTER_LINEAR);
                                           });

    std::cout << "resize_aware resized: " << std::boolalpha << resized << '\n';
    std::cout << "resize_aware size: " << dst.rows() << "x" << dst.cols() << '\n';
}

}  // namespace

auto main() -> int
{
    show_reallocation_caveat();
    show_safe_pattern();
    show_resize_aware();
    return 0;
}
