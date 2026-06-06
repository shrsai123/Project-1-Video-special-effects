Project 1 - Real-Time Video Filters

This project is a C++ OpenCV video application that captures live webcam video and applies several image-processing filters in real time. It also includes face detection, Depth Anything V2 depth estimation with ONNX Runtime, extra creative effects, and two extensions.

Setup:
  This project was built on Windows with MinGW, OpenCV 4.10, and ONNX Runtime 1.16.3.

  Expected folder layout:
    opencv\                                      OpenCV source folder
    build\                                      OpenCV build output
      bin\                                      OpenCV DLLs
      lib\                                      OpenCV import libraries
    third_party\onnxruntime-win-x64-1.16.3\
      include\                                  ONNX Runtime headers
      lib\                                      ONNX Runtime DLL/import library
    model_fp16.onnx                             Depth Anything V2 model

  Install MinGW build tools with MSYS2:
    pacman -S --needed mingw-w64-x86_64-gcc mingw-w64-x86_64-cmake mingw-w64-x86_64-make git unzip

OpenCV installation:
  If the opencv folder is missing, download OpenCV 4.10 source into the project root:
    git clone --branch 4.10.0 https://github.com/opencv/opencv.git opencv

  Build OpenCV with MinGW. This creates the build folder used by this project:
    cmake -S opencv -B build -G "MinGW Makefiles" -D CMAKE_BUILD_TYPE=Release -D BUILD_SHARED_LIBS=ON -D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF -D BUILD_EXAMPLES=OFF
    cmake --build build --parallel

  After the build, confirm these files exist:
    build\bin\libopencv_core4100.dll
    build\lib\libopencv_core4100.dll.a
    build\opencv2\opencv_modules.hpp

ONNX Runtime installation:
  Download onnxruntime-win-x64-1.16.3.zip from the ONNX Runtime releases page and place it in third_party, then extract it:
    New-Item -ItemType Directory -Force third_party
    Expand-Archive third_party\onnxruntime-win-x64-1.16.3.zip -DestinationPath third_party -Force

  After extraction, confirm these files exist:
    third_party\onnxruntime-win-x64-1.16.3\include\onnxruntime_cxx_api.h
    third_party\onnxruntime-win-x64-1.16.3\lib\onnxruntime.dll
    third_party\onnxruntime-win-x64-1.16.3\lib\onnxruntime.lib

  The depth features also require model_fp16.onnx in the project root.

Build:
  Compile the application from the project root:
  g++ -std=c++17 vidDisplay.cpp filters.cpp faceDetect.cpp -Ibuild -Iopencv/include -Iopencv/modules/core/include -Iopencv/modules/highgui/include -Iopencv/modules/imgcodecs/include -Iopencv/modules/imgproc/include -Iopencv/modules/videoio/include -Iopencv/modules/objdetect/include -I"third_party/onnxruntime-win-x64-1.16.3/include" -Lbuild/lib -L"third_party/onnxruntime-win-x64-1.16.3/lib" -lopencv_core4100 -lopencv_highgui4100 -lopencv_imgcodecs4100 -lopencv_imgproc4100 -lopencv_videoio4100 -lopencv_objdetect4100 -lonnxruntime -o vidDisplay.exe

If DLLs are not found:
  $env:PATH = "$PWD;$PWD\build\bin;$PWD\third_party\onnxruntime-win-x64-1.16.3\lib;C:\msys64\mingw64\bin;$env:PATH"

Run:
  .\vidDisplay.exe

Optional caption:
  .\vidDisplay.exe --caption "Depth is dramatic"

Live controls:
  o : original color video
  g : OpenCV grayscale using cv::cvtColor
  h : custom alternative grayscale
  p : sepia tone
  b : 5x5 blur
  x : Sobel X
  y : Sobel Y
  m : gradient magnitude
  l : blur and quantize
  f : face detection
  d : Depth Anything V2 grayscale depth
  z : creative depth spotlight filter
  n : color negative
  e : emboss
  c : colorful face with grayscale background
  t : toggle caption overlay
  v : start/stop slow-motion AVI recording of the active mode
  s : save the current active mode image
  q : quit

One-shot capture commands:
  .\vidDisplay.exe --save-basic-once
    Saves basic_original.png and image1-image6 outputs.

  .\vidDisplay.exe --save-depth-once
    Saves depth_image.png and depth_filter.png.

  .\vidDisplay.exe --save-effects-once
    Saves effects_original.png, effect10_negative.png, effect11_emboss.png, and effect12_face_color.png.

OpenCV grayscale:
  cv::cvtColor with COLOR_BGR2GRAY uses Y = 0.299 R + 0.587 G + 0.114 B. Green has the largest weight because human vision is most sensitive to green brightness.

Alternative grayscale:
  My custom grayscale uses 255 minus the red channel, then copies that value to B, G, and R.

Sepia:
  The filter stores original B, G, and R before computing new channels, so modified values are not reused.

Blur timing:
  blur5x5_1 time: 87.3745 ms
  blur5x5_2 time: 14.966 ms
  The faster version uses separable 1x5 passes and pointer access.

Depth:
  Depth estimation uses the provided DA2Network.hpp wrapper from da2-code.zip and model_fp16.onnx. The d key shows the depth map, and z uses the depth map for a creative spotlight effect.

Extra effects:
  Negative is a pixel-wise inversion.
  Emboss uses Sobel X/Y and a directional dot product.
  Color face / grayscale background uses the face detector.

Extensions:
  Slow-motion video recording:
    Start the app with .\vidDisplay.exe.
    Choose the effect to record, such as o for original, g for grayscale, d for depth, or z for the depth spotlight filter.
    Press v in the video window to start recording the current active mode.
    The terminal prints "Recording slow-motion video <filename>" when recording starts.
    Press v again to stop recording. The terminal prints "Stopped recording <filename>".
    Output files are named like recording_original_0_slow.avi or recording_depth_filter_0_slow.avi.
    Recording also stops cleanly if you quit with q.
    The recorded files are MJPG AVI files and are video-only.
  Caption overlay: run with --caption and press t. Saved images and recordings include the caption when enabled.

Source files:
  vidDisplay.cpp
  filters.cpp / filters.h
  faceDetect.cpp / faceDetect.h
  DA2Network.hpp
  model_fp16.onnx
