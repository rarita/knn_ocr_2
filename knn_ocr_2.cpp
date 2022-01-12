#include "knn_ocr_2.h"
#include "knn.cuh"

using namespace std;
namespace fs = std::filesystem; // слава тебе боже

void findAllImagesInDirectory(std::vector<std::string> &container, std::string dirPath) {
	try {
		for (auto& entry : fs::directory_iterator(dirPath)) {
			std::string fname = entry.path().generic_u8string();
			if (entry.is_directory())
				findAllImagesInDirectory(container, fname);
			else if (entry.is_regular_file() || entry.is_symlink()) {
				// if (cv::haveImageReader(fname)) очень долго отрабатывает, дропаем
				container.push_back(fname);
			}
		}
	}
	catch (std::exception& exc) {
		std::cerr << "Error reading training samples: " << exc.what() << std::endl;
	}
}

std::vector<cv::Mat> preprocessInput(const std::string filename) {
	
	std::vector<cv::Mat> vec;
	cv::Mat a2;
	a2 = cv::imread("C:\\Users\\rarita\\source\\repos\\knn_ocr_2\\img\\A2.png", cv::IMREAD_GRAYSCALE);
	cv::threshold(a2, a2, 178, 255, cv::ThresholdTypes::THRESH_BINARY);
	for (int idx = 0; idx < 100; idx++)
		vec.push_back(a2);
	// vec.push_back(cv::imread("C:\\Users\\rarita\\source\\repos\\knn_ocr_2\\img\\B.png", cv::IMREAD_GRAYSCALE));
	// vec.push_back(cv::imread("C:\\Users\\rarita\\source\\repos\\knn_ocr_2\\img\\C.png", cv::IMREAD_GRAYSCALE));
	return vec;

}

void showOutput(std::vector<CharacterClassification> outp) {
	for (auto& cc : outp) {
		std::cout << "Letter: " << (char)cc.cls << " imgp: " << &cc.loc << std::endl;
	}
}

int main()
{	
	std::vector<std::string> files;
	findAllImagesInDirectory(files, "C:\\Users\\rarita\\source\\repos\\knn_ocr_2\\chars");
	std::cout << "Found " << files.size() << " files to load!" << std::endl;
	
	KNNClassifier classifier(files, 20);
	std::vector<cv::Mat> chars = preprocessInput("image.png");
	std::vector<CharacterClassification> classifications = classifier.classifyCharacters(chars, 20);
	
	showOutput(classifications);

	return 0;
}
