#include <cstdint>
#include <exception>
#include <iostream>

#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>

#include "modernmat/modernmat.hpp"

namespace
{

constexpr std::uint8_t checker_black = 0U;
constexpr std::uint8_t checker_white = 255U;
constexpr std::uint8_t highlight_value = 180U;
constexpr int checker_rows = 8;
constexpr int checker_cols = 8;

auto make_checkerboard(int rows, int cols) -> modernmat::matx
{
    cv::Mat seed(rows, cols, CV_8UC1);
    for (int row_index = 0; row_index < rows; ++row_index) {
        for (int column_index = 0; column_index < cols; ++column_index) {
            auto const select_black = ((row_index + column_index) % 2 == 0);
            seed.at<std::uint8_t>(row_index, column_index) =
                select_black ? checker_black : checker_white;
        }
    }
    return modernmat::matx {seed};
}

void blur_in_place(modernmat::matx& image)
{
    auto header = cv::Mat(image);
    cv::GaussianBlur(header, header, cv::Size {3, 3}, 0.0);
}

void paint_roi(modernmat::matx& image, const cv::Rect& roi, std::uint8_t value)
{
    const cv::Mat header = image;
    cv::Mat roi_view(header, roi);
    roi_view.setTo(value);
}

void write_through_nonconst_ref(modernmat::matx& image)
{
    auto header = cv::Mat(image);  // binds to cv::Mat& APIs
    cv::threshold(header, header, 128.0, 255.0, cv::THRESH_BINARY);
}

auto wrap_resize_flow() -> modernmat::matx
{
    cv::Mat tmp;
    cv::Mat src(4, 4, CV_8UC1, cv::Scalar(7));
    // OpenCV allocates dst internally; construct MatX after the fact.
    cv::resize(src, tmp, cv::Size {8, 8}, 0.0, 0.0, cv::INTER_LINEAR);
    return modernmat::matx {tmp};
}

}  // namespace

auto main() -> int
{
    try {
        auto board = make_checkerboard(checker_rows, checker_cols);

        auto blurred = board.clone();
        blur_in_place(blurred);

        paint_roi(blurred, cv::Rect {2, 2, 4, 4}, highlight_value);

        write_through_nonconst_ref(blurred);

        auto wrapped = wrap_resize_flow();

        std::cout << "Checkerboard top-left value: "
                  << static_cast<int>(cv::Mat(board).at<std::uint8_t>(0, 0)) << '\n';
        std::cout << "Blurred top-left value: "
                  << static_cast<int>(cv::Mat(blurred).at<std::uint8_t>(0, 0)) << '\n';
        std::cout << "Wrapped resize rows x cols: " << wrapped.rows() << " x " << wrapped.cols()
                  << '\n';

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Example failed: " << ex.what() << '\n';
        return 1;
    }
}
