#include <cstdint>

#include "modernmat/modernmat.hpp"

#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>

constexpr int gradient_modulo = 256;
constexpr int gradient_extent = 15;
constexpr int blur_kernel_extent = 5;
constexpr double blur_sigma = 1.5;

TEST(ModernMatOpencvDiff, GaussianBlurMatchesCvMat)
{
    auto const size = cv::Size {gradient_extent, gradient_extent};
    cv::Mat src(size, CV_8UC1);
    for (int row_index = 0; row_index < size.height; ++row_index) {
        for (int column_index = 0; column_index < size.width; ++column_index) {
            auto const blended = (row_index + column_index) % gradient_modulo;
            src.at<std::uint8_t>(row_index, column_index) = static_cast<std::uint8_t>(blended);
        }
    }

    cv::Mat reference {};
    cv::GaussianBlur(src, reference, cv::Size {blur_kernel_extent, blur_kernel_extent}, blur_sigma);

    auto subject = modernmat::matx {src}.clone();
    auto working = cv::Mat(subject);
    cv::GaussianBlur(src, working, cv::Size {blur_kernel_extent, blur_kernel_extent}, blur_sigma);

    auto const diff = cv::norm(reference, cv::Mat(subject), cv::NORM_INF);
    EXPECT_DOUBLE_EQ(0.0, diff);
}

TEST(ModernMatOpencvDiff, WrapsOpenCvAllocatedOutput)
{
    constexpr int extent = 8;
    cv::Mat src(extent, extent, CV_8UC1);
    cv::randu(src, 0, 255);

    cv::Mat tmp;
    cv::GaussianBlur(src, tmp, cv::Size {5, 5}, 1.0);

    auto wrapped = modernmat::matx {tmp};
    auto diff = cv::norm(tmp, cv::Mat(wrapped), cv::NORM_INF);
    EXPECT_DOUBLE_EQ(0.0, diff);
}
