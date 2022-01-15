// knn_ocr_2.h : включаемый файл для стандартных системных включаемых файлов
// или включаемые файлы для конкретного проекта.

#pragma once

#include <omp.h>
#include <iostream>
#include <numeric>
#include <filesystem>

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

// include "cuda_runtime.h"
// include "device_launch_parameters.h"

constexpr int CHAR_RES = 20;
constexpr int MIN_TOLERANT_TOKEN_SIZE = (CHAR_RES * CHAR_RES) / 2;

struct ExtractedCharacter {
public:
	uint x, y, w, h;
	cv::Mat mat;
};

// TODO: установите здесь ссылки на дополнительные заголовки, требующиеся для программы.
