#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/iterator/constant_iterator.h>

#include <omp.h>
#include <chrono>
#include <iostream>
#include <filesystem>

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

// Раскомментить для дебага CUDA-кернелей
// #define DBG_CUDA_KERNEL

#define CHECK_CUDA(ans, abt, ...) { gpuAssert(ans, __FILE__, __LINE__, abt, ##__VA_ARGS__); }
template<typename... Args>
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort, Args... args)
{
	if (code != cudaSuccess)
	{
		std::stringstream errMsgStream;
		errMsgStream << "Error on the GPU side:" << std::endl;
		(errMsgStream << ... << args) << std::endl;
		errMsgStream << "Error code : " << code << std::endl;
		errMsgStream << "File: " << file << "; line: " << line << std::endl;
		if (abort) {
			throw std::exception(errMsgStream.str().c_str());
		}
		else {
			std::cerr << errMsgStream.str();
		}
	}
}

struct CharacterClassification {
public:
	char cls;
	cv::Mat* loc;
};

class KNNClassifier {
public:
	KNNClassifier() = delete;
	KNNClassifier(std::vector<std::string> &fileNames, int resolution);
	std::vector<CharacterClassification> classifyCharacters(std::vector<cv::Mat> &chars, int k);
	~KNNClassifier();
private:
	int resolution;
	int dataChunkSize;
	int trainDataSize;
	uint8_t* trainDataPtr;
	char* trainClsPtr;
};

void testOMP();