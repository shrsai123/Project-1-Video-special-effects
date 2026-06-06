#include <cmath>

#include <opencv2/core.hpp>

static uchar clampToByte(double value) {
    if (value > 255.0) {
        return 255;
    }

    if (value < 0.0) {
        return 0;
    }

    return static_cast<uchar>(value);
}

int greyscale(cv::Mat &src, cv::Mat &dst) {
    if (src.empty()) {
        return -1;
    }

    dst.create(src.rows, src.cols, CV_8UC3);

    for (int row = 0; row < src.rows; row++) {
        cv::Vec3b *dstRow = dst.ptr<cv::Vec3b>(row);

        if (src.channels() == 3) {
            const cv::Vec3b *srcRow = src.ptr<cv::Vec3b>(row);

            for (int col = 0; col < src.cols; col++) {
                uchar altGray = 255 - srcRow[col][2];
                dstRow[col] = cv::Vec3b(altGray, altGray, altGray);
            }
        } else if (src.channels() == 1) {
            const uchar *srcRow = src.ptr<uchar>(row);

            for (int col = 0; col < src.cols; col++) {
                uchar altGray = 255 - srcRow[col];
                dstRow[col] = cv::Vec3b(altGray, altGray, altGray);
            }
        } else {
            return -1;
        }
    }

    return 0;
}

int sepia(cv::Mat &src, cv::Mat &dst) {
    if (src.empty() || src.channels() != 3) {
        return -1;
    }

    dst.create(src.rows, src.cols, CV_8UC3);

    for (int row = 0; row < src.rows; row++) {
        const cv::Vec3b *srcRow = src.ptr<cv::Vec3b>(row);
        cv::Vec3b *dstRow = dst.ptr<cv::Vec3b>(row);

        for (int col = 0; col < src.cols; col++) {
            double oldBlue = srcRow[col][0];
            double oldGreen = srcRow[col][1];
            double oldRed = srcRow[col][2];

            double newBlue = 0.272 * oldRed + 0.534 * oldGreen + 0.131 * oldBlue;
            double newGreen = 0.349 * oldRed + 0.686 * oldGreen + 0.168 * oldBlue;
            double newRed = 0.393 * oldRed + 0.769 * oldGreen + 0.189 * oldBlue;

            dstRow[col][0] = clampToByte(newBlue);
            dstRow[col][1] = clampToByte(newGreen);
            dstRow[col][2] = clampToByte(newRed);
        }
    }

    return 0;
}

int blur5x5_1(cv::Mat &src, cv::Mat &dst) {
    if (src.empty() || src.channels() != 3) {
        return -1;
    }

    src.copyTo(dst);

    if (src.rows < 5 || src.cols < 5) {
        return 0;
    }

    int kernel[5][5] = {
        {1, 2, 4, 2, 1},
        {2, 4, 8, 4, 2},
        {4, 8, 16, 8, 4},
        {2, 4, 8, 4, 2},
        {1, 2, 4, 2, 1}
    };

    for (int row = 2; row < src.rows - 2; row++) {
        for (int col = 2; col < src.cols - 2; col++) {
            int blueSum = 0;
            int greenSum = 0;
            int redSum = 0;

            for (int kernelRow = -2; kernelRow <= 2; kernelRow++) {
                for (int kernelCol = -2; kernelCol <= 2; kernelCol++) {
                    cv::Vec3b pixel = src.at<cv::Vec3b>(row + kernelRow, col + kernelCol);
                    int weight = kernel[kernelRow + 2][kernelCol + 2];

                    blueSum += pixel[0] * weight;
                    greenSum += pixel[1] * weight;
                    redSum += pixel[2] * weight;
                }
            }

            dst.at<cv::Vec3b>(row, col)[0] = static_cast<uchar>(blueSum / 100);
            dst.at<cv::Vec3b>(row, col)[1] = static_cast<uchar>(greenSum / 100);
            dst.at<cv::Vec3b>(row, col)[2] = static_cast<uchar>(redSum / 100);
        }
    }

    return 0;
}

int blur5x5_2(cv::Mat &src, cv::Mat &dst) {
    if (src.empty() || src.channels() != 3) {
        return -1;
    }

    src.copyTo(dst);

    if (src.rows < 5 || src.cols < 5) {
        return 0;
    }

    cv::Mat temp(src.rows, src.cols, CV_16SC3);

    for (int row = 0; row < src.rows; row++) {
        const cv::Vec3b *srcRow = src.ptr<cv::Vec3b>(row);
        cv::Vec3s *tempRow = temp.ptr<cv::Vec3s>(row);

        for (int col = 2; col < src.cols - 2; col++) {
            for (int channel = 0; channel < 3; channel++) {
                tempRow[col][channel] =
                    srcRow[col - 2][channel] +
                    2 * srcRow[col - 1][channel] +
                    4 * srcRow[col][channel] +
                    2 * srcRow[col + 1][channel] +
                    srcRow[col + 2][channel];
            }
        }
    }

    for (int row = 2; row < src.rows - 2; row++) {
        cv::Vec3b *dstRow = dst.ptr<cv::Vec3b>(row);
        const cv::Vec3s *tempRowMinus2 = temp.ptr<cv::Vec3s>(row - 2);
        const cv::Vec3s *tempRowMinus1 = temp.ptr<cv::Vec3s>(row - 1);
        const cv::Vec3s *tempRow = temp.ptr<cv::Vec3s>(row);
        const cv::Vec3s *tempRowPlus1 = temp.ptr<cv::Vec3s>(row + 1);
        const cv::Vec3s *tempRowPlus2 = temp.ptr<cv::Vec3s>(row + 2);

        for (int col = 2; col < src.cols - 2; col++) {
            for (int channel = 0; channel < 3; channel++) {
                int sum =
                    tempRowMinus2[col][channel] +
                    2 * tempRowMinus1[col][channel] +
                    4 * tempRow[col][channel] +
                    2 * tempRowPlus1[col][channel] +
                    tempRowPlus2[col][channel];

                dstRow[col][channel] = static_cast<uchar>(sum / 100);
            }
        }
    }

    return 0;
}

int sobelX3x3(cv::Mat &src, cv::Mat &dst) {
    if (src.empty() || src.channels() != 3) {
        return -1;
    }

    dst = cv::Mat::zeros(src.rows, src.cols, CV_16SC3);

    if (src.rows < 3 || src.cols < 3) {
        return 0;
    }

    cv::Mat temp = cv::Mat::zeros(src.rows, src.cols, CV_16SC3);

    for (int row = 0; row < src.rows; row++) {
        const cv::Vec3b *srcRow = src.ptr<cv::Vec3b>(row);
        cv::Vec3s *tempRow = temp.ptr<cv::Vec3s>(row);

        for (int col = 1; col < src.cols - 1; col++) {
            for (int channel = 0; channel < 3; channel++) {
                tempRow[col][channel] =
                    -srcRow[col - 1][channel] +
                    srcRow[col + 1][channel];
            }
        }
    }

    for (int row = 1; row < src.rows - 1; row++) {
        const cv::Vec3s *tempRowAbove = temp.ptr<cv::Vec3s>(row - 1);
        const cv::Vec3s *tempRow = temp.ptr<cv::Vec3s>(row);
        const cv::Vec3s *tempRowBelow = temp.ptr<cv::Vec3s>(row + 1);
        cv::Vec3s *dstRow = dst.ptr<cv::Vec3s>(row);

        for (int col = 1; col < src.cols - 1; col++) {
            for (int channel = 0; channel < 3; channel++) {
                dstRow[col][channel] =
                    (tempRowAbove[col][channel] +
                    2 * tempRow[col][channel] +
                    tempRowBelow[col][channel]) / 4;
            }
        }
    }

    return 0;
}

int sobelY3x3(cv::Mat &src, cv::Mat &dst) {
    if (src.empty() || src.channels() != 3) {
        return -1;
    }

    dst = cv::Mat::zeros(src.rows, src.cols, CV_16SC3);

    if (src.rows < 3 || src.cols < 3) {
        return 0;
    }

    cv::Mat temp = cv::Mat::zeros(src.rows, src.cols, CV_16SC3);

    for (int row = 1; row < src.rows - 1; row++) {
        const cv::Vec3b *srcRowAbove = src.ptr<cv::Vec3b>(row - 1);
        const cv::Vec3b *srcRowBelow = src.ptr<cv::Vec3b>(row + 1);
        cv::Vec3s *tempRow = temp.ptr<cv::Vec3s>(row);

        for (int col = 0; col < src.cols; col++) {
            for (int channel = 0; channel < 3; channel++) {
                tempRow[col][channel] =
                    srcRowAbove[col][channel] -
                    srcRowBelow[col][channel];
            }
        }
    }

    for (int row = 1; row < src.rows - 1; row++) {
        const cv::Vec3s *tempRow = temp.ptr<cv::Vec3s>(row);
        cv::Vec3s *dstRow = dst.ptr<cv::Vec3s>(row);

        for (int col = 1; col < src.cols - 1; col++) {
            for (int channel = 0; channel < 3; channel++) {
                dstRow[col][channel] =
                    (tempRow[col - 1][channel] +
                    2 * tempRow[col][channel] +
                    tempRow[col + 1][channel]) / 4;
            }
        }
    }

    return 0;
}

int magnitude(cv::Mat &sobelX, cv::Mat &sobelY, cv::Mat &dst) {
    if (sobelX.empty() || sobelY.empty() ||
        sobelX.channels() != 3 || sobelY.channels() != 3 ||
        sobelX.size() != sobelY.size()) {
        return -1;
    }

    dst = cv::Mat::zeros(sobelX.rows, sobelX.cols, CV_8UC3);

    for (int row = 0; row < sobelX.rows; row++) {
        const cv::Vec3s *sobelXRow = sobelX.ptr<cv::Vec3s>(row);
        const cv::Vec3s *sobelYRow = sobelY.ptr<cv::Vec3s>(row);
        cv::Vec3b *dstRow = dst.ptr<cv::Vec3b>(row);

        for (int col = 0; col < sobelX.cols; col++) {
            for (int channel = 0; channel < 3; channel++) {
                double xVal = static_cast<double>(sobelXRow[col][channel]);
                double yVal = static_cast<double>(sobelYRow[col][channel]);
                dstRow[col][channel] = clampToByte(std::sqrt(xVal * xVal + yVal * yVal));
            }
        }
    }

    return 0;
}

int blurQuantize(cv::Mat &src, cv::Mat &dst, int levels) {
    if (src.empty() || src.channels() != 3 || levels <= 0) {
        return -1;
    }

    cv::Mat blurred;

    if (blur5x5_2(src, blurred) != 0) {
        return -1;
    }

    blurred.copyTo(dst);

    int bucketSize = 255 / levels;

    if (bucketSize < 1) {
        bucketSize = 1;
    }

    for (int row = 0; row < dst.rows; row++) {
        cv::Vec3b *dstRow = dst.ptr<cv::Vec3b>(row);

        for (int col = 0; col < dst.cols; col++) {
            for (int channel = 0; channel < 3; channel++) {
                int value = dstRow[col][channel];
                int quantizedValue = (value / bucketSize) * bucketSize;

                dstRow[col][channel] = static_cast<uchar>(quantizedValue);
            }
        }
    }

    return 0;
}

int negative(cv::Mat &src, cv::Mat &dst) {
    if (src.empty() || src.channels() != 3) {
        return -1;
    }

    dst.create(src.rows, src.cols, CV_8UC3);

    for (int row = 0; row < src.rows; row++) {
        const cv::Vec3b *srcRow = src.ptr<cv::Vec3b>(row);
        cv::Vec3b *dstRow = dst.ptr<cv::Vec3b>(row);

        for (int col = 0; col < src.cols; col++) {
            dstRow[col][0] = 255 - srcRow[col][0];
            dstRow[col][1] = 255 - srcRow[col][1];
            dstRow[col][2] = 255 - srcRow[col][2];
        }
    }

    return 0;
}

int emboss(cv::Mat &src, cv::Mat &dst) {
    if (src.empty() || src.channels() != 3) {
        return -1;
    }

    cv::Mat sobelX;
    cv::Mat sobelY;

    if (sobelX3x3(src, sobelX) != 0 || sobelY3x3(src, sobelY) != 0) {
        return -1;
    }

    dst = cv::Mat(src.rows, src.cols, CV_8UC3, cv::Scalar(128, 128, 128));

    for (int row = 0; row < src.rows; row++) {
        const cv::Vec3s *xRow = sobelX.ptr<cv::Vec3s>(row);
        const cv::Vec3s *yRow = sobelY.ptr<cv::Vec3s>(row);
        cv::Vec3b *dstRow = dst.ptr<cv::Vec3b>(row);

        for (int col = 0; col < src.cols; col++) {
            double signedX = (xRow[col][0] + xRow[col][1] + xRow[col][2]) / 3.0;
            double signedY = (yRow[col][0] + yRow[col][1] + yRow[col][2]) / 3.0;
            uchar shade = clampToByte(128.0 + 0.7071 * signedX + 0.7071 * signedY);

            dstRow[col] = cv::Vec3b(shade, shade, shade);
        }
    }

    return 0;
}
