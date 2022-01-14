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

std::vector<ExtractedCharacter> preprocessInput(const std::string filename) {

	cv::Mat inp;
	inp = cv::imread(filename, cv::IMREAD_GRAYSCALE);
	assert(!inp.empty());

	cv::threshold(inp, inp, 178, 255, cv::ThresholdTypes::THRESH_BINARY);

	std::vector<vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(inp, contours, hierarchy, cv::RETR_TREE, cv::ContourApproximationModes::CHAIN_APPROX_SIMPLE);
	
	std::vector<cv::Rect> letterBounds;
	for (int idx = 0; idx < contours.size(); idx++) {
		if (hierarchy[idx][3] == 0) {
			letterBounds.push_back(cv::boundingRect(contours[idx]));
		}
	}

	std::vector<ExtractedCharacter> letters;
	for (auto& bounds : letterBounds) {
		cv::Mat letter = inp(bounds); // это только референс!!!

		assert(letter.type() == CV_8UC1);
		
		int dimensionMax = max(bounds.width, bounds.height);

		cv::Mat formattedLetter(cv::Size(dimensionMax, dimensionMax), CV_8UC1, cv::Scalar(255)); // cv::Mat::ones * 255.
		assert(!formattedLetter.empty());
		// cv::Mat formattedLetter = cv::Mat::zeros(dimensionMax, dimensionMax, letter.type());
		// cv::imshow("l", formattedLetter);
		// cv::waitKey();

		if (bounds.width > bounds.height) {
			int y = dimensionMax / 2 - bounds.height / 2;
			cv::Rect bb(0, y, bounds.width, bounds.height);
			letter.copyTo(formattedLetter(bb));
		}
		else if (bounds.width < bounds.height) {
			int x = dimensionMax / 2 - bounds.width / 2;
			cv::Rect bb(x, 0, bounds.width, bounds.height);
			letter.copyTo(formattedLetter(bb));
		}
		else {
			letter.copyTo(formattedLetter);
		}
		
		ExtractedCharacter ec;
		ec.x = bounds.x;
		ec.y = bounds.y;
		ec.w = bounds.width;
		ec.h = bounds.height;
		cv::Mat resized;
		cv::resize(formattedLetter, resized, cv::Size(CHAR_RES, CHAR_RES), 0, 0, cv::INTER_AREA);
		ec.mat = resized;

		letters.push_back(ec);

	}

	return letters;

}

void showOutput(std::vector<CharacterClassification> outp) {
	for (auto& cc : outp) {
		std::cout << "Letter: " << (char)cc.cls << " coords (x, y, w, h): " << 
			cc.x << " " << cc.y << " " << cc.w << " " << cc.h << std::endl;
	}
}

int main()
{	
	std::vector<std::string> files;
	findAllImagesInDirectory(files, "C:\\Users\\rarita\\source\\repos\\knn_ocr_2\\chars");
	std::cout << "Found " << files.size() << " files to load!" << std::endl;
	
	// big black mat
	// cv::Mat bigMat(cv::Size(512, 512), CV_8UC1, cv::Scalar(0));
	// small white mat
	// cv::Mat smallMat(cv::Size(256, 256), CV_8UC1, cv::Scalar(255));
	// copy small to big with ROI
	// smallMat.copyTo(bigMat(cv::Rect(255, 255, 255, 255)));

	// display big mat
	// cv::imshow("big", bigMat);
	// cv::waitKey();

	// return 0;

	KNNClassifier classifier(files, 20);
	std::vector<ExtractedCharacter> chars = preprocessInput("C:/Users/rarita/source/repos/knn_ocr_2/bigshot.png");
	std::vector<CharacterClassification> classifications = classifier.classifyCharacters(chars, 20);
	
	showOutput(classifications);

	return 0;
}
