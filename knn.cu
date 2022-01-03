#include "knn.cuh"

__global__
void add(int n, float* x, float* y)
{
	for (int i = 0; i < n; i++)
		y[i] = x[i] + y[i];
}

__global__ void init(int n, float* x, float* y) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride) {
		x[i] = 1.0f;
		y[i] = 2.0f;
	}
}

KNNClassifier::KNNClassifier(std::vector<std::string>& fileNames, int resolution)
{
	
	// ����� �������� resolution � trainDataSize, ��� ��� ������ �����������
	this->resolution = resolution;
	this->trainDataSize = fileNames.size();

	// �������� ������ ��� �������� ������
	cudaError_t rsp;
	rsp = cudaMalloc(&this->trainDataPtr, sizeof(uint8_t) * resolution * resolution * this->trainDataSize);
	if (rsp != cudaError::cudaSuccess) {
		throw std::exception("Could not allocate memory for training samples: " + rsp);
	}
	// �������� ������ ��� �������������� �������� ������
	rsp = cudaMalloc(&this->trainClsPtr, sizeof(char) * this->trainDataSize);
	if (rsp != cudaError::cudaSuccess) {
		throw std::exception("Could not allocate memory for training classifiers: " + rsp);
	}
	
	const int dataChunkSize = sizeof(uint8_t) * resolution * resolution;
	for (int idx = 0; idx < this->trainDataSize; idx++) {

		// ��������� � ����� � �������� (1�8U)
		cv::Mat mat = cv::imread(fileNames[idx], cv::ImreadModes::IMREAD_GRAYSCALE);
		if (mat.empty()) {
			std::stringstream errMsgStream;
			errMsgStream << "Error reading file " << fileNames[idx] << ". Mat is empty.";
			throw std::exception(errMsgStream.str().c_str());
		}
		else if (mat.size().height != resolution || mat.size().width != resolution) {
			std::stringstream errMsgStream;
			errMsgStream << "Error reading file " << fileNames[idx] << ". Training data image has wrong resolution: " << 
				mat.size().width << "x" << mat.size().height;
			throw std::exception(errMsgStream.str().c_str());
		}

		rsp = cudaMemcpy(this->trainDataPtr + idx * dataChunkSize, &mat, dataChunkSize, cudaMemcpyKind::cudaMemcpyHostToDevice);
		CHECK_CUDA(rsp, true, "Cannot load file ", fileNames[idx]);

		// ���� ��� �� � �� �������� ��������, ������� �� ��� � �����. ����������
		// �� *nix ��� ������� �������� �� �����, ����� ���������
		const char cls = fileNames[idx][fileNames[idx].find_last_of("//") + 1];
		rsp = cudaMemset(this->trainClsPtr + idx, cls, 1);
		CHECK_CUDA(rsp, true, "Cannot save classifier ", cls, " for file ", fileNames[idx]);

	}

	/*
	char* testSample = (char*) malloc(17 * sizeof(char));
	memset(testSample, 0, 17 * sizeof(char));
	rsp = cudaMemcpy(testSample, this->trainClsPtr, 16 * sizeof(char), cudaMemcpyKind::cudaMemcpyDeviceToHost);
	testSample[16] = '\0';
	std::cout << "16 first classifiers from GPU memory: " << testSample << std::endl;
	*/

	std::cout << this->trainDataSize << " training samples successfully loaded!" << std::endl <<
		"KNNClassifier " << this << " has been successfully initialized." << std::endl;

}

std::vector<CharacterClassification> KNNClassifier::classifyCharacters(std::vector<cv::Mat>& chars)
{
	// ������ CUDA Streams �� ���������� ���� �� �������������
	// ����� ���������� ���������� �����
	cudaError_t rsp;
	std::vector<cudaStream_t*> streams;

	for (cv::Mat& mat : chars) {
		
		// ������� �����
		cudaStream_t* stream;
		rsp = cudaStreamCreate(stream);
		CHECK_CUDA(rsp, true, "Cannot initialize CUDA stream.");
		
		// �������� � �������� ����� ������� ����� �������� �� GPU
		uint8_t* requestedMatPtr;
		rsp = cudaMallocAsync(&requestedMatPtr, this->dataChunkSize, *stream);
		CHECK_CUDA(rsp, true, "Could not allocate memory for supplied image");
		rsp = cudaMemcpyAsync(requestedMatPtr, &mat, this->dataChunkSize, cudaMemcpyKind::cudaMemcpyHostToDevice, *stream);
		CHECK_CUDA(rsp, true, "Could not transfer data of the supplied image to the GPU");

		// ���� ��� ��, �������������� ������, � ������� ����� ��������� ����������
		// ����� �������������� ���������� � �������. �������� ��� �������� �������
		uint32_t* distPtr;
		rsp = cudaMallocAsync(&distPtr, this->trainDataSize * sizeof(uint32_t), *stream);
		CHECK_CUDA(rsp, true, "Could not allocate memory for neighbor distances");
		rsp = cudaMemsetAsync(distPtr, UINT_MAX, this->trainDataSize, *stream);
		CHECK_CUDA(rsp, true, "Could not initialize distances array");

		// �� ������ ���������� ������ � ����� ����� ����������

	}

	return std::vector<CharacterClassification>();

}

KNNClassifier::~KNNClassifier()
{

	cudaError_t rsp;

	rsp = cudaFree(this->trainDataPtr);
	CHECK_CUDA(rsp, true, "Could not free train data memory on GPU...");
	
	rsp = cudaFree(this->trainClsPtr);
	CHECK_CUDA(rsp, true, "Could not free train classifiers memory on GPU...");

}

int tutorial(void) {

	int N = 1 << 20;

	float* x, * y;
	cudaMallocManaged(&x, N * sizeof(float));
	cudaMallocManaged(&y, N * sizeof(float));

	// initialize x and y arrays on the host
	for (int i = 0; i < N; i++) {
		x[i] = 1.0f;
		y[i] = 2.0f;
	}

	// initialize x and y arrays on the host
	add<<<1, 1 >>> (N, x, y);

	// Wait for GPU to finish before accessing on host
	cudaDeviceSynchronize();
	cudaError err = cudaGetLastError();

	// Check for errors (all values should be 3.0f)
	float maxError = 0.0f;
	for (int i = 0; i < N; i++)
		maxError = fmax(maxError, fabs(y[i] - 3.0f));
	std::cout << "Max error: " << maxError << std::endl;

	// Free memory
	cudaFree(x);
	cudaFree(y);

	return 0;

}