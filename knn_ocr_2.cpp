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

int main()
{	
	std::vector<std::string> files;
	findAllImagesInDirectory(files, "C:\\Users\\rarita\\source\\repos\\knn_ocr_2\\chars");
	std::cout << "Found " << files.size() << " files to load!" << std::endl;
	
	KNNClassifier classifier(files, 20);
	return 0;
}
