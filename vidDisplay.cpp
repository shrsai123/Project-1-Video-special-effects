#include <iostream>
#include <string>
#include <chrono>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "DA2Network.hpp"
#include "faceDetect.h"
#include "filters.h"

static int runDA2Depth(DA2Network &depthNetwork, cv::Mat &src, cv::Mat &depth8u) {
    if (src.empty() || src.channels() != 3) {
        return -1;
    }

    int smallSide = std::min(src.rows, src.cols);
    float scaleFactor = 384.0f / static_cast<float>(smallSide);
    if (scaleFactor > 1.0f) {
        scaleFactor = 1.0f;
    }

    depthNetwork.set_input(src, scaleFactor);
    depthNetwork.run_network(depth8u, src.size());
    return 0;
}

static int depthSpotlightFilter(cv::Mat &src, cv::Mat &depth8u, cv::Mat &dst) {
    if (src.empty() || depth8u.empty() || src.size() != depth8u.size()) {
        return -1;
    }

    cv::Mat depthFloat;
    cv::Mat mask;
    cv::Mat blurred;
    cv::Mat farWorld;
    cv::Mat teal(src.size(), src.type(), cv::Scalar(90, 170, 150));
    cv::Mat edges;

    depth8u.convertTo(depthFloat, CV_32FC1, 1.0 / 255.0);
    cv::GaussianBlur(depthFloat, mask, cv::Size(0, 0), 5.0);

    for (int row = 0; row < mask.rows; row++) {
        float *maskRow = mask.ptr<float>(row);
        for (int col = 0; col < mask.cols; col++) {
            float value = (maskRow[col] - 0.42f) / 0.30f;
            maskRow[col] = std::max(0.0f, std::min(1.0f, value));
        }
    }

    cv::GaussianBlur(src, blurred, cv::Size(0, 0), 12.0);
    cv::addWeighted(blurred, 0.58, teal, 0.42, 0.0, farWorld);
    cv::Canny(depth8u, edges, 50, 130);
    cv::dilate(edges, edges, cv::Mat(), cv::Point(-1, -1), 1);
    farWorld.setTo(cv::Scalar(40, 240, 255), edges);

    dst.create(src.rows, src.cols, CV_8UC3);
    for (int row = 0; row < src.rows; row++) {
        const cv::Vec3b *srcRow = src.ptr<cv::Vec3b>(row);
        const cv::Vec3b *farRow = farWorld.ptr<cv::Vec3b>(row);
        const float *maskRow = mask.ptr<float>(row);
        cv::Vec3b *dstRow = dst.ptr<cv::Vec3b>(row);

        for (int col = 0; col < src.cols; col++) {
            float nearWeight = maskRow[col];
            float farWeight = 1.0f - nearWeight;
            for (int channel = 0; channel < 3; channel++) {
                dstRow[col][channel] = static_cast<uchar>(
                    srcRow[col][channel] * nearWeight + farRow[col][channel] * farWeight);
            }
        }
    }

    return 0;
}

static int faceColorIsolation(cv::Mat &src, cv::Mat &dst) {
    if (src.empty() || src.channels() != 3) {
        return -1;
    }

    cv::Mat gray;
    std::vector<cv::Rect> faces;

    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(gray, dst, cv::COLOR_GRAY2BGR);

    if (detectFaces(gray, faces) != 0) {
        return -1;
    }

    for (int i = 0; i < static_cast<int>(faces.size()); i++) {
        cv::Rect face = faces[i] & cv::Rect(0, 0, src.cols, src.rows);

        if (face.width > 50 && face.height > 50) {
            src(face).copyTo(dst(face));
            cv::rectangle(dst, face, cv::Scalar(70, 220, 255), 3);
        }
    }

    return 0;
}

static std::string modeBaseName(int displayMode) {
    if (displayMode == 1) {
        return "cvtcolor_gray";
    } else if (displayMode == 2) {
        return "alt_gray";
    } else if (displayMode == 3) {
        return "sepia";
    } else if (displayMode == 4) {
        return "blur5x5";
    } else if (displayMode == 5) {
        return "sobel_x";
    } else if (displayMode == 6) {
        return "sobel_y";
    } else if (displayMode == 7) {
        return "gradient_magnitude";
    } else if (displayMode == 8) {
        return "blur_quantize";
    } else if (displayMode == 9) {
        return "face_detection";
    } else if (displayMode == 10) {
        return "depth_image";
    } else if (displayMode == 11) {
        return "depth_filter";
    } else if (displayMode == 12) {
        return "negative";
    } else if (displayMode == 13) {
        return "emboss";
    } else if (displayMode == 14) {
        return "face_color";
    }

    return "original";
}

static void drawCaption(cv::Mat &frame, const std::string &caption) {
    if (frame.empty() || caption.empty()) {
        return;
    }

    cv::Mat colorFrame;
    bool converted = false;

    if (frame.channels() == 1) {
        cv::cvtColor(frame, colorFrame, cv::COLOR_GRAY2BGR);
        converted = true;
    } else {
        colorFrame = frame;
    }

    int baseline = 0;
    double fontScale = 0.8;
    int thickness = 2;
    int margin = 18;
    cv::Size textSize = cv::getTextSize(caption, cv::FONT_HERSHEY_SIMPLEX, fontScale, thickness, &baseline);

    while (textSize.width > colorFrame.cols - 2 * margin && fontScale > 0.35) {
        fontScale -= 0.05;
        textSize = cv::getTextSize(caption, cv::FONT_HERSHEY_SIMPLEX, fontScale, thickness, &baseline);
    }

    int x = std::max(margin, (colorFrame.cols - textSize.width) / 2);
    int y = colorFrame.rows - margin;
    cv::Point textOrigin(x, y);
    cv::Rect background(
        x - 10,
        y - textSize.height - 10,
        textSize.width + 20,
        textSize.height + baseline + 20);

    background &= cv::Rect(0, 0, colorFrame.cols, colorFrame.rows);
    cv::rectangle(colorFrame, background, cv::Scalar(0, 0, 0), cv::FILLED);
    cv::putText(colorFrame, caption, textOrigin, cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(255, 255, 255), thickness, cv::LINE_AA);

    if (converted) {
        colorFrame.copyTo(frame);
    }
}

int main(int argc, char** argv) {
    int cameraIndex = 0;
    std::string depthModelPath = "model_fp16.onnx";
    bool saveDepthOnce = false;
    bool saveEffectsOnce = false;
    bool saveBasicOnce = false;
    std::string caption = "CS5330 Vision";

    if (argc >= 2) {
        if (std::string(argv[1]) == "--save-depth-once") {
            saveDepthOnce = true;
        } else if (std::string(argv[1]) == "--save-effects-once") {
            saveEffectsOnce = true;
        } else if (std::string(argv[1]) == "--save-basic-once") {
            saveBasicOnce = true;
        } else if (std::string(argv[1]) == "--caption" && argc >= 3) {
            caption = argv[2];
        } else {
            cameraIndex = std::stoi(argv[1]);
        }
    }

    if (argc >= 3 && std::string(argv[1]) != "--caption") {
        depthModelPath = argv[2];
    }

    cv::VideoCapture camera(cameraIndex);

    if (!camera.isOpened()) {
        std::cerr << "Error: could not open camera channel " << cameraIndex << std::endl;
        return -1;
    }

    if (saveDepthOnce) {
        cv::Mat frame;
        cv::Mat depthImage;
        cv::Mat depthFilter;

        camera >> frame;
        if (frame.empty()) {
            std::cerr << "Error: captured an empty frame." << std::endl;
            return -1;
        }

        try {
            DA2Network depthNetwork(depthModelPath.c_str());

            if (runDA2Depth(depthNetwork, frame, depthImage) != 0 ||
                depthSpotlightFilter(frame, depthImage, depthFilter) != 0) {
                std::cerr << "Error: DA2 one-shot depth capture failed." << std::endl;
                return -1;
            }
        } catch (const std::exception &error) {
            std::cerr << "Error: could not run DA2 model: " << error.what() << std::endl;
            return -1;
        }

        cv::imwrite("depth_image.png", depthImage);
        cv::imwrite("depth_filter.png", depthFilter);
        std::cout << "Saved depth_image.png and depth_filter.png" << std::endl;
        return 0;
    }

    if (saveBasicOnce) {
        cv::Mat frame;
        cv::Mat cvGray;
        cv::Mat altGray;
        cv::Mat sepiaFrame;
        cv::Mat blurNaive;
        cv::Mat blurFast;
        cv::Mat sobelX;
        cv::Mat sobelY;
        cv::Mat sobelXVis;
        cv::Mat sobelYVis;
        cv::Mat magFrame;
        cv::Mat blurQuantFrame;

        camera >> frame;
        if (frame.empty()) {
            std::cerr << "Error: captured an empty frame." << std::endl;
            return -1;
        }

        auto naiveStart = std::chrono::high_resolution_clock::now();
        int naiveStatus = blur5x5_1(frame, blurNaive);
        auto naiveEnd = std::chrono::high_resolution_clock::now();

        auto fastStart = std::chrono::high_resolution_clock::now();
        int fastStatus = blur5x5_2(frame, blurFast);
        auto fastEnd = std::chrono::high_resolution_clock::now();

        double naiveMs = std::chrono::duration<double, std::milli>(naiveEnd - naiveStart).count();
        double fastMs = std::chrono::duration<double, std::milli>(fastEnd - fastStart).count();

        cv::cvtColor(frame, cvGray, cv::COLOR_BGR2GRAY);

        if (greyscale(frame, altGray) != 0 ||
            sepia(frame, sepiaFrame) != 0 ||
            naiveStatus != 0 ||
            fastStatus != 0 ||
            sobelX3x3(frame, sobelX) != 0 ||
            sobelY3x3(frame, sobelY) != 0 ||
            magnitude(sobelX, sobelY, magFrame) != 0 ||
            blurQuantize(frame, blurQuantFrame, 10) != 0) {
            std::cerr << "Error: basic filter one-shot capture failed." << std::endl;
            return -1;
        }

        cv::convertScaleAbs(sobelX, sobelXVis);
        cv::convertScaleAbs(sobelY, sobelYVis);

        cv::imwrite("basic_original.png", frame);
        cv::imwrite("image1_cvtcolor_gray.png", cvGray);
        cv::imwrite("image2_alt_gray.png", altGray);
        cv::imwrite("image3_sepia.png", sepiaFrame);
        cv::imwrite("image4_blur5x5.png", blurFast);
        cv::imwrite("image5_sobel_x.png", sobelXVis);
        cv::imwrite("image5_sobel_y.png", sobelYVis);
        cv::imwrite("image5_gradient_magnitude.png", magFrame);
        cv::imwrite("image6_blur_quantize.png", blurQuantFrame);

        std::cout << "Saved basic_original.png and image1-image6 outputs." << std::endl;
        std::cout << "blur5x5_1 time: " << naiveMs << " ms" << std::endl;
        std::cout << "blur5x5_2 time: " << fastMs << " ms" << std::endl;
        return 0;
    }

    if (saveEffectsOnce) {
        cv::Mat frame;
        cv::Mat negativeFrame;
        cv::Mat embossFrame;
        cv::Mat faceColorFrame;

        camera >> frame;
        if (frame.empty()) {
            std::cerr << "Error: captured an empty frame." << std::endl;
            return -1;
        }

        if (negative(frame, negativeFrame) != 0 ||
            emboss(frame, embossFrame) != 0 ||
            faceColorIsolation(frame, faceColorFrame) != 0) {
            std::cerr << "Error: special effects one-shot capture failed." << std::endl;
            return -1;
        }

        cv::imwrite("effects_original.png", frame);
        cv::imwrite("effect10_negative.png", negativeFrame);
        cv::imwrite("effect11_emboss.png", embossFrame);
        cv::imwrite("effect12_face_color.png", faceColorFrame);
        std::cout << "Saved effects_original.png, effect10_negative.png, effect11_emboss.png, and effect12_face_color.png" << std::endl;
        return 0;
    }

    const std::string windowName = "Live Video";
    cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);

    std::cout << "Camera channel " << cameraIndex << " opened." << std::endl;
    std::cout << "Controls:" << std::endl;
    std::cout << "  q : quit" << std::endl;
    std::cout << "  o : original color video" << std::endl;
    std::cout << "  g : toggle grayscale" << std::endl;
    std::cout << "  h : toggle alternative grayscale" << std::endl;
    std::cout << "  p : toggle sepia tone" << std::endl;
    std::cout << "  b : toggle 5x5 blur" << std::endl;
    std::cout << "  x : toggle Sobel X" << std::endl;
    std::cout << "  y : toggle Sobel Y" << std::endl;
    std::cout << "  m : toggle gradient magnitude" << std::endl;
    std::cout << "  l : toggle blur and quantize" << std::endl;
    std::cout << "  f : toggle face detection" << std::endl;
    std::cout << "  d : toggle Depth Anything V2 grayscale depth" << std::endl;
    std::cout << "  z : toggle creative depth spotlight filter" << std::endl;
    std::cout << "  n : toggle color negative" << std::endl;
    std::cout << "  e : toggle emboss" << std::endl;
    std::cout << "  c : toggle colorful face / grayscale background" << std::endl;
    std::cout << "  s : save current frame using the active mode filename" << std::endl;
    std::cout << "  v : start/stop slow-motion video recording" << std::endl;
    std::cout << "  t : toggle caption overlay" << std::endl;
    std::cout << "Caption text: " << caption << std::endl;
    std::cout << "Depth model: " << depthModelPath << std::endl;

    cv::Mat frame;
    cv::Mat grayFrame;
    cv::Mat sobelFrame;
    cv::Mat sobelXFrame;
    cv::Mat sobelYFrame;
    cv::Mat displayFrame;
    std::vector<cv::Rect> faces;
    int displayMode = 0;
    DA2Network *depthNetwork = nullptr;
    cv::VideoWriter videoWriter;
    bool recording = false;
    int recordingCount = 0;
    const double recordingFps = 30.0;
    const int slowMotionFactor = 2;
    std::string recordingVideoFilename;
    bool showCaption = false;

    while (true) {
        camera >> frame;

        if (frame.empty()) {
            std::cerr << "Error: captured an empty frame." << std::endl;
            break;
        }

        if (displayMode == 1) {
            cv::cvtColor(frame, displayFrame, cv::COLOR_BGR2GRAY);
        } else if (displayMode == 2) {
            if (greyscale(frame, displayFrame) != 0) {
                std::cerr << "Error: alternative grayscale conversion failed." << std::endl;
                break;
            }
        } else if (displayMode == 3) {
            if (sepia(frame, displayFrame) != 0) {
                std::cerr << "Error: sepia conversion failed." << std::endl;
                break;
            }
        } else if (displayMode == 4) {
            if (blur5x5_2(frame, displayFrame) != 0) {
                std::cerr << "Error: blur conversion failed." << std::endl;
                break;
            }
        } else if (displayMode == 5) {
            if (sobelX3x3(frame, sobelFrame) != 0) {
                std::cerr << "Error: Sobel X conversion failed." << std::endl;
                break;
            }

            cv::convertScaleAbs(sobelFrame, displayFrame);
        } else if (displayMode == 6) {
            if (sobelY3x3(frame, sobelFrame) != 0) {
                std::cerr << "Error: Sobel Y conversion failed." << std::endl;
                break;
            }

            cv::convertScaleAbs(sobelFrame, displayFrame);
        } else if (displayMode == 7) {
            if (sobelX3x3(frame, sobelXFrame) != 0 ||
                sobelY3x3(frame, sobelYFrame) != 0 ||
                magnitude(sobelXFrame, sobelYFrame, displayFrame) != 0) {
                std::cerr << "Error: gradient magnitude conversion failed." << std::endl;
                break;
            }
        } else if (displayMode == 8) {
            if (blurQuantize(frame, displayFrame, 10) != 0) {
                std::cerr << "Error: blur/quantize conversion failed." << std::endl;
                break;
            }
        } else if (displayMode == 9) {
            frame.copyTo(displayFrame);
            cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);

            if (detectFaces(grayFrame, faces) != 0 ||
                drawBoxes(displayFrame, faces, 50, 1.0f) != 0) {
                std::cerr << "Error: face detection failed." << std::endl;
                break;
            }
        } else if (displayMode == 10) {
            if (depthNetwork == nullptr) {
                try {
                    depthNetwork = new DA2Network(depthModelPath.c_str());
                } catch (const std::exception &error) {
                    std::cerr << "Error: could not load DA2 model: " << error.what() << std::endl;
                    break;
                }
            }

            if (runDA2Depth(*depthNetwork, frame, displayFrame) != 0) {
                std::cerr << "Error: DA2 depth estimate failed." << std::endl;
                break;
            }
        } else if (displayMode == 11) {
            if (depthNetwork == nullptr) {
                try {
                    depthNetwork = new DA2Network(depthModelPath.c_str());
                } catch (const std::exception &error) {
                    std::cerr << "Error: could not load DA2 model: " << error.what() << std::endl;
                    break;
                }
            }

            cv::Mat depthImage;
            if (runDA2Depth(*depthNetwork, frame, depthImage) != 0 ||
                depthSpotlightFilter(frame, depthImage, displayFrame) != 0) {
                std::cerr << "Error: DA2 creative depth filter failed." << std::endl;
                break;
            }
        } else if (displayMode == 12) {
            if (negative(frame, displayFrame) != 0) {
                std::cerr << "Error: negative effect failed." << std::endl;
                break;
            }
        } else if (displayMode == 13) {
            if (emboss(frame, displayFrame) != 0) {
                std::cerr << "Error: emboss effect failed." << std::endl;
                break;
            }
        } else if (displayMode == 14) {
            if (faceColorIsolation(frame, displayFrame) != 0) {
                std::cerr << "Error: face color isolation effect failed." << std::endl;
                break;
            }
        } else {
            displayFrame = frame;
        }

        if (showCaption) {
            drawCaption(displayFrame, caption);
        }

        if (recording) {
            cv::Mat videoFrame;

            if (displayFrame.channels() == 1) {
                cv::cvtColor(displayFrame, videoFrame, cv::COLOR_GRAY2BGR);
            } else {
                videoFrame = displayFrame;
            }

            for (int repeat = 0; repeat < slowMotionFactor; repeat++) {
                videoWriter.write(videoFrame);
            }
        }

        cv::imshow(windowName, displayFrame);

        int key = cv::waitKey(30) & 0xFF;

        if (key == 'q') {
            break;
        }

        if (key == 'o') {
            displayMode = 0;
            std::cout << "Original color video on" << std::endl;
        }

        if (key == 'g') {
            displayMode = (displayMode == 1) ? 0 : 1;
            std::cout << "OpenCV grayscale " << (displayMode == 1 ? "on" : "off") << std::endl;
        }

        if (key == 'h') {
            displayMode = (displayMode == 2) ? 0 : 2;
            std::cout << "Alternative grayscale " << (displayMode == 2 ? "on" : "off") << std::endl;
        }

        if (key == 'p') {
            displayMode = (displayMode == 3) ? 0 : 3;
            std::cout << "Sepia tone " << (displayMode == 3 ? "on" : "off") << std::endl;
        }

        if (key == 'b') {
            displayMode = (displayMode == 4) ? 0 : 4;
            std::cout << "5x5 blur " << (displayMode == 4 ? "on" : "off") << std::endl;
        }

        if (key == 'x') {
            displayMode = (displayMode == 5) ? 0 : 5;
            std::cout << "Sobel X " << (displayMode == 5 ? "on" : "off") << std::endl;
        }

        if (key == 'y') {
            displayMode = (displayMode == 6) ? 0 : 6;
            std::cout << "Sobel Y " << (displayMode == 6 ? "on" : "off") << std::endl;
        }

        if (key == 'm') {
            displayMode = (displayMode == 7) ? 0 : 7;
            std::cout << "Gradient magnitude " << (displayMode == 7 ? "on" : "off") << std::endl;
        }

        if (key == 'l') {
            displayMode = (displayMode == 8) ? 0 : 8;
            std::cout << "Blur/quantize " << (displayMode == 8 ? "on" : "off") << std::endl;
        }

        if (key == 'f') {
            displayMode = (displayMode == 9) ? 0 : 9;
            std::cout << "Face detection " << (displayMode == 9 ? "on" : "off") << std::endl;
        }

        if (key == 'd') {
            displayMode = (displayMode == 10) ? 0 : 10;
            std::cout << "Depth Anything V2 grayscale depth " << (displayMode == 10 ? "on" : "off") << std::endl;
        }

        if (key == 'z') {
            displayMode = (displayMode == 11) ? 0 : 11;
            std::cout << "Depth spotlight filter " << (displayMode == 11 ? "on" : "off") << std::endl;
        }

        if (key == 'n') {
            displayMode = (displayMode == 12) ? 0 : 12;
            std::cout << "Color negative " << (displayMode == 12 ? "on" : "off") << std::endl;
        }

        if (key == 'e') {
            displayMode = (displayMode == 13) ? 0 : 13;
            std::cout << "Emboss " << (displayMode == 13 ? "on" : "off") << std::endl;
        }

        if (key == 'c') {
            displayMode = (displayMode == 14) ? 0 : 14;
            std::cout << "Color face / grayscale background " << (displayMode == 14 ? "on" : "off") << std::endl;
        }

        if (key == 'v') {
            if (recording) {
                videoWriter.release();
                recording = false;
                std::cout << "Stopped recording " << recordingVideoFilename << std::endl;
            } else {
                std::string baseFilename = "recording_" + modeBaseName(displayMode) + "_" + std::to_string(recordingCount);
                recordingVideoFilename = baseFilename + "_slow.avi";
                int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
                videoWriter.open(recordingVideoFilename, fourcc, recordingFps, frame.size(), true);

                if (videoWriter.isOpened()) {
                    recording = true;
                    recordingCount++;
                    std::cout << "Recording slow-motion video " << recordingVideoFilename << std::endl;
                } else {
                    std::cout << "Could not start recording " << recordingVideoFilename << std::endl;
                }
            }
        }

        if (key == 't') {
            showCaption = !showCaption;
            std::cout << "Caption overlay " << (showCaption ? "on" : "off") << std::endl;
        }

        if (key == 's') {
            std::string filename = "original_image.png";

            if (displayMode == 1) {
                filename = "image1_cvtcolor_gray.png";
            } else if (displayMode == 2) {
                filename = "image2_alt_gray.png";
            } else if (displayMode == 3) {
                filename = "image3_sepia.png";
            } else if (displayMode == 4) {
                filename = "image4_blur5x5.png";
            } else if (displayMode == 5) {
                filename = "image5_sobel_x.png";
            } else if (displayMode == 6) {
                filename = "image5_sobel_y.png";
            } else if (displayMode == 7) {
                filename = "image5_gradient_magnitude.png";
            } else if (displayMode == 8) {
                filename = "image6_blur_quantize.png";
            } else if (displayMode == 9) {
                filename = "face_detection.png";
            } else if (displayMode == 10) {
                filename = "depth_image.png";
            } else if (displayMode == 11) {
                filename = "depth_filter.png";
            } else if (displayMode == 12) {
                filename = "effect10_negative.png";
            } else if (displayMode == 13) {
                filename = "effect11_emboss.png";
            } else if (displayMode == 14) {
                filename = "effect12_face_color.png";
            }

            bool saved = cv::imwrite(filename, displayFrame);

            if (saved) {
                std::cout << "Saved " << filename << std::endl;
            } else {
                std::cout << "Could not save " << filename << std::endl;
            }
        }
    }

    delete depthNetwork;
    if (recording) {
        videoWriter.release();
    }
    camera.release();
    cv::destroyAllWindows();

    return 0;
}
