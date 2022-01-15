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
		
		if ((bounds.width * bounds.height) < MIN_TOLERANT_TOKEN_SIZE) {
			continue;
		}

		cv::Mat letter = inp(bounds); // это только референс!!!
		
		// Sanity checks
		assert(!letter.empty());
		assert(letter.type() == CV_8UC1);
		
		int dimensionMax = max(bounds.width, bounds.height);

		cv::Mat formattedLetter(cv::Size(dimensionMax, dimensionMax), CV_8UC1, cv::Scalar(255)); // cv::Mat::ones * 255.
		assert(!formattedLetter.empty());

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

std::string toMultilineString(std::vector<CharacterClassification> outp) {
	
	// Найти высоту одной буквы (должна быть равная у всех, но найдем минимальную)
	uint letterHeight = UINT_MAX;
	for (auto& cc : outp) {
		if (cc.h < letterHeight) {
			letterHeight = cc.h;
		}
	}

	// Разбить на строки исходя из высоты букв
	std::map<uint, std::vector<CharacterClassification>> heightMap;
	for (auto& cc : outp) {
		
		int keyidx = -1;
		for (auto& kv : heightMap) {
			if (abs((long)(cc.y - kv.first)) <= letterHeight) {
				keyidx = kv.first;
			}
		}

		if (keyidx == -1) {
			keyidx = cc.y;
			heightMap[keyidx] = std::vector<CharacterClassification>();
		}

		heightMap[keyidx].push_back(cc);

	}

	// Отсортировать буквы в heightMap по координате X для каждой строки
	// В C++ std::map по дефолту отсортирована в порядке возрастания ключей (operator <)
	for (auto& kv : heightMap) {
		std::sort(kv.second.begin(), kv.second.end(), 
			[](CharacterClassification cc1, CharacterClassification cc2) { 
				return cc1.x < cc2.x; 
			}
		);
	}

	// Вставить пробелы (?) для каждой строки
	std::string result;
	for (auto& kv : heightMap) {

		std::string res;

		// найти расст. между буквами без пробела
		int letterDist = INT_MAX;
		int minLetterWidth = INT_MAX;
		std::vector<int> dists;
		for (int idx = 0; idx < kv.second.size() - 1; idx++) {
			// символы уже отсортированы по Х
			int dist = kv.second[idx + 1].x - (kv.second[idx].x + kv.second[idx].w);
			dists.push_back(dist);
			if (dist < letterDist) {
				letterDist = dist;
			}
			int letterWidth = kv.second[idx].w;
			if (letterWidth < minLetterWidth) {
				minLetterWidth = letterWidth;
			}
		}

		if (!dists.empty()) {
			int meanDist = std::reduce(dists.begin(), dists.end()) / dists.size();

			for (int idx = 0; idx < kv.second.size() - 1; idx++) {
				int dist = kv.second[idx + 1].x - (kv.second[idx].x + kv.second[idx].w);
				res += kv.second[idx].cls;
				if (dist > meanDist) {
					res += " ";
				}
			}
		}

		result += res + kv.second[kv.second.size() - 1].cls + "\n";

	}

	return result;

}

void printUsageInfo(std::string err) {
	std::cout << "KNN OCR Application" << std::endl <<
		"Usage: ./knn_ocr <path_to_training_data> <path_to_image_to_process> <K hyperparameter>" << std::endl;
	if (!err.empty()) {
		std::cout << err << std::endl;
	}
}

int main(int argc, char* argv[]) {

	if (argc != 4) {
		printUsageInfo("invalid args quantity");
		return 0;
	}

	std::string trainDataFolder, inputFile;
	int k;
	try {
		trainDataFolder = argv[1];
		inputFile = argv[2];
		k = std::stoi(argv[3]);
	}
	catch (...) {
		printUsageInfo("invalid args format");
		return 0;
	}

	std::vector<std::string> files;
	findAllImagesInDirectory(files, trainDataFolder);
	std::cout << "Found " << files.size() << " files to load!" << std::endl;

	KNNClassifier classifier(files, k);
	std::vector<ExtractedCharacter> chars = preprocessInput(inputFile);
	std::vector<CharacterClassification> classifications = classifier.classifyCharacters(chars, k);
	
	std::cout << "Recognized: " << std::endl << toMultilineString(classifications) << std::endl;

	return 0;

}
