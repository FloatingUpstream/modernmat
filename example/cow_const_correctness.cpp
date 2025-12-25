#include <cstdint>
#include <iostream>

#include <opencv2/core.hpp>

#include "modernmat/modernmat.hpp"

namespace
{

constexpr int extent = 4;
constexpr std::uint8_t base_value = 10;
constexpr std::uint8_t updated_value = 77;

void show_const_view_is_read_only()
{
    modernmat::matx image(extent, extent, CV_8UC1);
    cv::Mat writable_view = image;
    writable_view.setTo(base_value);

    const modernmat::matx& readonly = image;
    cv::Mat header = readonly.as_cv_const();

    std::cout << "Const view data pointer: " << static_cast<const void*>(header.data) << '\n';
    std::cout << "Const view value: " << static_cast<int>(header.at<std::uint8_t>(0, 0)) << '\n';
}

void show_copy_on_write()
{
    modernmat::matx image(extent, extent, CV_8UC1);
    cv::Mat original_view = image;
    original_view.setTo(base_value);

    auto copy = image;
    auto const* original_ptr = static_cast<const modernmat::matx&>(image).data();
    auto const* shared_ptr = static_cast<const modernmat::matx&>(copy).data();

    cv::Mat copy_view = copy;  // triggers detach
    auto const* detached_ptr = static_cast<const modernmat::matx&>(copy).data();
    copy_view.at<std::uint8_t>(0, 0) = updated_value;

    std::cout << "Original value: " << static_cast<int>(cv::Mat(image).at<std::uint8_t>(0, 0))
              << '\n';
    std::cout << "Copy value: " << static_cast<int>(cv::Mat(copy).at<std::uint8_t>(0, 0)) << '\n';
    std::cout << "Original data: " << static_cast<const void*>(original_ptr) << '\n';
    std::cout << "Shared data: " << static_cast<const void*>(shared_ptr) << '\n';
    std::cout << "Detached data: " << static_cast<const void*>(detached_ptr) << '\n';
}

}  // namespace

auto main() -> int
{
    show_const_view_is_read_only();
    show_copy_on_write();
    return 0;
}
