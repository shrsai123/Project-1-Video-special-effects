#include <iostream>
#include <opencv2/core.hpp>        
#include <opencv2/imgcodecs.hpp>   
#include <opencv2/highgui.hpp>     
#include <opencv2/imgproc.hpp>     

int main(int argc, char** argv) {
    // Get the image path either from the command line or use a default.
    std::string imagePath = (argc >= 2) ? argv[1] : "test.jpg";

    // Load the image. cv::IMREAD_COLOR loads as 3-channel BGR.
    cv::Mat originalImage = cv::imread(imagePath, cv::IMREAD_COLOR);

    if (originalImage.empty()) {
        std::cerr << "Error: could not open or find image: " << imagePath << std::endl;
        return -1;
    }

    std::cout << "Loaded image: " << imagePath
              << " (" << originalImage.cols << " x " << originalImage.rows << ")" << std::endl;
    std::cout << "Controls:\n"
              << "  q : quit\n"
              << "  g : toggle grayscale\n"
              << "  b : toggle Gaussian blur\n"
              << "  r : reset to original\n"
              << "  s : save current view as output.png\n";

    // displayedImage is what we show. We keep originalImage untouched
    // so we can always reset to it.
    cv::Mat displayedImage = originalImage.clone();

    const std::string windowName = "Image Display";
    cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);

    // Main event loop.
    while (true) {
        cv::imshow(windowName, displayedImage);
        int key = cv::waitKey(0) & 0xFF;

        if (key == 'q') {
            break;
        } else if (key == 'g') {
            // Convert to grayscale and back to BGR so imshow displays consistently.
            cv::Mat gray;
            cv::cvtColor(originalImage, gray, cv::COLOR_BGR2GRAY);
            cv::cvtColor(gray, displayedImage, cv::COLOR_GRAY2BGR);
        } else if (key == 'b') {
            cv::GaussianBlur(originalImage, displayedImage, cv::Size(15, 15), 0, 0);
        } else if (key == 'r') {
            displayedImage = originalImage.clone();
        } else if (key == 's') {
            bool ok = cv::imwrite("output.png", displayedImage);
            std::cout << (ok ? "Saved output.png" : "Save failed") << std::endl;
        } else {
            std::cout << "Key " << key << " pressed (no action bound)" << std::endl;
        }
    }

    cv::destroyAllWindows();
    return 0;
}