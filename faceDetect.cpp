/*
  Bruce A. Maxwell
  Spring 2024
  CS 5330 Computer Vision

  Functions for finding faces and drawing boxes around them.
*/

#include <cstdio>
#include <cstdlib>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>

#include "faceDetect.h"

int detectFaces(cv::Mat &grey, std::vector<cv::Rect> &faces) {
    static cv::Mat half;
    static cv::CascadeClassifier faceCascade;
    static cv::String faceCascadeFile(FACE_CASCADE_FILE);

    if (faceCascade.empty()) {
        if (!faceCascade.load(faceCascadeFile)) {
            std::printf("Unable to load face cascade file\n");
            std::printf("Terminating\n");
            std::exit(-1);
        }
    }

    faces.clear();

    cv::resize(grey, half, cv::Size(grey.cols / 2, grey.rows / 2));
    cv::equalizeHist(half, half);
    faceCascade.detectMultiScale(half, faces);

    for (int i = 0; i < static_cast<int>(faces.size()); i++) {
        faces[i].x *= 2;
        faces[i].y *= 2;
        faces[i].width *= 2;
        faces[i].height *= 2;
    }

    return 0;
}

int drawBoxes(cv::Mat &frame, std::vector<cv::Rect> &faces, int minWidth, float scale) {
    cv::Scalar boxColor(170, 120, 110);

    for (int i = 0; i < static_cast<int>(faces.size()); i++) {
        if (faces[i].width > minWidth) {
            cv::Rect face(faces[i]);
            face.x = static_cast<int>(face.x * scale);
            face.y = static_cast<int>(face.y * scale);
            face.width = static_cast<int>(face.width * scale);
            face.height = static_cast<int>(face.height * scale);
            cv::rectangle(frame, face, boxColor, 3);
        }
    }

    return 0;
}
