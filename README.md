# Project 1 - Real-Time Video Filters

This project is a C++ OpenCV video application that captures live webcam video and applies several image-processing filters in real time. It also includes face detection, Depth Anything V2 depth estimation with ONNX Runtime, extra creative effects, and two extensions.

## Setup

The project was built on Windows with MinGW, OpenCV 4.10, and ONNX Runtime 1.16.3. The compile command assumes this local folder layout:

```text
Project-1-Video-special-effects/
  opencv/                                      OpenCV source folder
  build/                                       OpenCV build output
    bin/                                       OpenCV DLLs
    lib/                                       OpenCV import libraries
  third_party/
    onnxruntime-win-x64-1.16.3/
      include/                                 ONNX Runtime headers
      lib/                                     ONNX Runtime DLL/import library
  model_fp16.onnx                              Depth Anything V2 model
```

Install the MinGW build tools first. With MSYS2/MinGW, install GCC, CMake, Make, Git, and unzip:

```powershell
pacman -S --needed mingw-w64-x86_64-gcc mingw-w64-x86_64-cmake mingw-w64-x86_64-make git unzip
```

### OpenCV Installation

If the `opencv` folder is not already present, download OpenCV 4.10 source into the project root:

```powershell
git clone --branch 4.10.0 https://github.com/opencv/opencv.git opencv
```

Build OpenCV with MinGW. This creates the `build` folder used by this project:

```powershell
cmake -S opencv -B build -G "MinGW Makefiles" -D CMAKE_BUILD_TYPE=Release -D BUILD_SHARED_LIBS=ON -D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF -D BUILD_EXAMPLES=OFF
cmake --build build --parallel
```

After the build, confirm these files exist:

```text
build/bin/libopencv_core4100.dll
build/lib/libopencv_core4100.dll.a
build/opencv2/opencv_modules.hpp
```

### ONNX Runtime Installation

Download `onnxruntime-win-x64-1.16.3.zip` from the ONNX Runtime releases page and place it in `third_party`, then extract it:

```powershell
New-Item -ItemType Directory -Force third_party
Expand-Archive third_party\onnxruntime-win-x64-1.16.3.zip -DestinationPath third_party -Force
```

After extraction, confirm these files exist:

```text
third_party/onnxruntime-win-x64-1.16.3/include/onnxruntime_cxx_api.h
third_party/onnxruntime-win-x64-1.16.3/lib/onnxruntime.dll
third_party/onnxruntime-win-x64-1.16.3/lib/onnxruntime.lib
```

The depth features also require `model_fp16.onnx` in the project root.

## Build

Compile the application from the project root:

```powershell
g++ -std=c++17 vidDisplay.cpp filters.cpp faceDetect.cpp -Ibuild -Iopencv/include -Iopencv/modules/core/include -Iopencv/modules/highgui/include -Iopencv/modules/imgcodecs/include -Iopencv/modules/imgproc/include -Iopencv/modules/videoio/include -Iopencv/modules/objdetect/include -I"third_party/onnxruntime-win-x64-1.16.3/include" -Lbuild/lib -L"third_party/onnxruntime-win-x64-1.16.3/lib" -lopencv_core4100 -lopencv_highgui4100 -lopencv_imgcodecs4100 -lopencv_imgproc4100 -lopencv_videoio4100 -lopencv_objdetect4100 -lonnxruntime -o vidDisplay.exe
```

If the program cannot find DLLs, run:

```powershell
$env:PATH = "$PWD;$PWD\build\bin;$PWD\third_party\onnxruntime-win-x64-1.16.3\lib;C:\msys64\mingw64\bin;$env:PATH"
```

## Run

```powershell
.\vidDisplay.exe
```

Optional caption:

```powershell
.\vidDisplay.exe --caption "Depth is dramatic"
```

## Live Controls

- `o`: original color video
- `g`: OpenCV grayscale using `cv::cvtColor`
- `h`: custom alternative grayscale
- `p`: sepia tone
- `b`: 5x5 blur
- `x`: Sobel X
- `y`: Sobel Y
- `m`: gradient magnitude
- `l`: blur and quantize
- `f`: face detection
- `d`: Depth Anything V2 grayscale depth
- `z`: creative depth spotlight filter
- `n`: color negative
- `e`: emboss
- `c`: colorful face with grayscale background
- `t`: toggle caption overlay
- `v`: start/stop slow-motion AVI recording of the active mode
- `s`: save the current active mode image
- `q`: quit

When `s` is pressed, the saved filename depends on the active mode. For example, original mode saves `original_image.png`, OpenCV grayscale saves `image1_cvtcolor_gray.png`, sepia saves `image3_sepia.png`, and depth mode saves `depth_image.png`.

## One-Shot Capture Commands

Save Required Images 1-6:

```powershell
.\vidDisplay.exe --save-basic-once
```

This saves:

- `basic_original.png`
- `image1_cvtcolor_gray.png`
- `image2_alt_gray.png`
- `image3_sepia.png`
- `image4_blur5x5.png`
- `image5_sobel_x.png`
- `image5_sobel_y.png`
- `image5_gradient_magnitude.png`
- `image6_blur_quantize.png`

Save Required Images 8-9:

```powershell
.\vidDisplay.exe --save-depth-once
```

This saves:

- `depth_image.png`
- `depth_filter.png`

Save Required Images 10-12:

```powershell
.\vidDisplay.exe --save-effects-once
```

This saves:

- `effects_original.png`
- `effect10_negative.png`
- `effect11_emboss.png`
- `effect12_face_color.png`

## Filter Notes

OpenCV grayscale uses the standard luminance conversion:

```text
Y = 0.299 R + 0.587 G + 0.114 B
```

Green receives the largest weight because human vision is most sensitive to green brightness.

The custom alternative grayscale uses `255 - red` and copies that value to all three channels. This gives a visibly different grayscale image because red-heavy regions are inverted instead of converted by luminance.

The sepia filter stores the original B, G, and R values before computing the new channels, so no modified channel is accidentally reused in another channel computation.

The 5x5 blur was implemented twice. The naive version uses the full 5x5 kernel and `cv::Mat::at`; the faster version uses separable 1x5 horizontal and vertical passes with pointer access.

Blur timing from the captured frame:

- `blur5x5_1`: `87.3745 ms`
- `blur5x5_2`: `14.966 ms`

## Depth Anything V2

Depth estimation uses the provided `DA2Network.hpp` wrapper from `da2-code.zip` and the included `model_fp16.onnx` model. The `d` mode displays the depth map as grayscale. The `z` mode uses the depth map for a creative spotlight effect that keeps near regions sharper while blurring/tinting farther regions and outlining depth edges.

## Extra Effects

- Color negative: pixel-wise inversion of all BGR channels.
- Emboss: uses Sobel X and Sobel Y, then takes a directional dot product to create a raised gray relief effect.
- Color face / grayscale background: uses face detection to keep the detected face in color while converting the rest of the image to grayscale.

## Extensions

Slow-motion video recording:

- Start the app with `.\vidDisplay.exe`.
- Choose the effect you want to record, such as `o` for original, `g` for grayscale, `d` for depth, or `z` for the depth spotlight filter.
- Press `v` in the video window to start recording the current active mode.
- The terminal prints `Recording slow-motion video <filename>` when recording starts.
- Press `v` again to stop recording. The terminal prints `Stopped recording <filename>`.
- Output files look like `recording_original_0_slow.avi` or `recording_depth_filter_0_slow.avi`.
- Recording also stops cleanly if you quit with `q`.
- The recorded files are MJPG AVI files and are video-only.

Caption / meme overlay:

- Run with `.\vidDisplay.exe --caption "your text here"`.
- Press `t` to toggle the caption.
- Saved images and recorded videos include the caption when it is enabled.

## Source Files

- `vidDisplay.cpp`: main live video application and keyboard controls
- `filters.cpp`, `filters.h`: custom image filters
- `faceDetect.cpp`, `faceDetect.h`: face detection helpers
- `DA2Network.hpp`: Depth Anything V2 ONNX Runtime wrapper
- `model_fp16.onnx`: Depth Anything V2 model
