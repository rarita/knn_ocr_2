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

__global__ void dist(uint8_t* trainPtr, uint8_t* inputPtr, uint32_t* distPtr, int res, int tds) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	int resSq = res * res;
	for (int idx = index; idx < resSq * tds; idx += stride) {
		// XOR-�� �����. ������� ������������� �������� � �������� �� ���� ��������
		uint8_t xorEd = trainPtr[idx] ^ inputPtr[idx % resSq];
		// ��������� � ������ ����������
		distPtr[idx / resSq] += xorEd / 255; // �� 0 �� 1
	}
}

KNNClassifier::KNNClassifier(std::vector<std::string>& fileNames, int resolution)
{
	
	// ����� ������ �������������
	const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

	// ����� �������� � �������� ��� ��������� ���������, ��� ��� ������ �����������
	this->resolution = resolution;
	this->trainDataSize = fileNames.size();
	this->dataChunkSize = resolution * resolution * sizeof(uint8_t);

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

	#pragma omp parallel for
	for (int idx = 0; idx < this->trainDataSize; idx++) {

		// ��������� � ����� � �������� (1�8U)
		cv::Mat mat = cv::imread(fileNames[idx], cv::ImreadModes::IMREAD_GRAYSCALE);
		// ��������� ��� ������� � ��� �� � ������������
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

		rsp = cudaMemcpyAsync(this->trainDataPtr + idx * this->dataChunkSize, &mat, this->dataChunkSize, cudaMemcpyKind::cudaMemcpyHostToDevice);
		CHECK_CUDA(rsp, true, "Cannot load file ", fileNames[idx]);

		// ���� ��� �� � �� �������� ��������, ������� �� ��� � �����. ����������
		// �� *nix ��� ������� �������� �� �����, ����� ���������
		const char cls = fileNames[idx][fileNames[idx].find_last_of("//") + 1];
		rsp = cudaMemset(this->trainClsPtr + idx, cls, 1);
		CHECK_CUDA(rsp, true, "Cannot save classifier ", cls, " for file ", fileNames[idx]);

	}

	rsp = cudaDeviceSynchronize();
	CHECK_CUDA(rsp, true, "Could not synchronize CUDA device after uploading data");

	/*
	char* testSample = (char*) malloc(17 * sizeof(char));
	memset(testSample, 0, 17 * sizeof(char));
	rsp = cudaMemcpy(testSample, this->trainClsPtr, 16 * sizeof(char), cudaMemcpyKind::cudaMemcpyDeviceToHost);
	testSample[16] = '\0';
	std::cout << "16 first classifiers from GPU memory: " << testSample << std::endl;
	*/


	const std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

	std::cout << this->trainDataSize << " training samples successfully loaded in " <<
		std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() << " ms. " <<
		"KNNClassifier " << this << " has been successfully initialized." << std::endl;

	int attr;
	rsp = cudaDeviceGetAttribute(&attr, cudaDevAttrMemoryPoolsSupported, 0);
	CHECK_CUDA(rsp, true, "Can't probe for memory pools support");
	printf("Device supports memory pooling (async memset): %d\n", attr);

}

std::vector<CharacterClassification> KNNClassifier::classifyCharacters(std::vector<cv::Mat>& chars, int k)
{
	// ������ CUDA Streams �� ���������� ���� �� �������������
	// ����� ���������� ���������� �����
	cudaError_t rsp;
	std::vector<cudaStream_t*> streams;
	std::vector<CharacterClassification> result(chars.size());

	for (cv::Mat& mat : chars) {
		
		// ������� �����
		cudaStream_t stream;
		rsp = cudaStreamCreate(&stream);
		CHECK_CUDA(rsp, true, "Cannot initialize CUDA stream.");
		
		// �������� � �������� ����� ������� ����� �������� �� GPU
		uint8_t* requestedMatPtr;
		rsp = cudaMallocAsync(&requestedMatPtr, this->dataChunkSize, stream);
		CHECK_CUDA(rsp, true, "Could not allocate memory for supplied image");
		rsp = cudaMemcpyAsync(requestedMatPtr, &mat, this->dataChunkSize, cudaMemcpyKind::cudaMemcpyHostToDevice, stream);
		CHECK_CUDA(rsp, true, "Could not transfer data of the supplied image to the GPU");

		// ���� ��� ��, �������������� ������, � ������� ����� ��������� ����������
		// ����� �������������� ���������� � �������. �������� ��� �������� �������
		uint32_t* distPtr;
		rsp = cudaMallocAsync(&distPtr, this->trainDataSize * sizeof(uint32_t), stream);
		CHECK_CUDA(rsp, true, "Could not allocate memory for neighbor distances");
		rsp = cudaMemsetAsync(distPtr, UINT_MAX, this->trainDataSize, stream);
		CHECK_CUDA(rsp, true, "Could not initialize distances array");
		// ��� ���������� ������ � ����������������, ����� ��� ������������ in-place � ������� thrust
		// ������, �������� ����� ������� ���������������
		thrust::device_vector<char> clsCopyVec(this->trainDataSize);
		// thrust-��������� �� ������������ ������ ���������������
		thrust::device_ptr<char> clsDevPtr(this->trainClsPtr);
		thrust::copy(thrust::cuda::par.on(stream), clsDevPtr, clsDevPtr + this->trainDataSize, clsCopyVec.begin());
		// thrust::copy(thrust::cuda::par(*stream), this->trainClsPtr, this->trainClsPtr + sizeof(uint8_t) * this->trainDataSize, clsVec.begin());

		// ��������� �������, ������� ��������� ��� ���������� �����
		// ������� ������������ � ������ �������������� �������.
		// ������ ������� ��� �� �������� thrust::transform � ������������� thrust::bitwise_xor
		dist<<<1, 256, 0, stream>>> (trainDataPtr, requestedMatPtr, distPtr, this->resolution, this->trainDataSize);

		// ���� ������� TOP-K ������� �������������� �����������
		// thrust::device_ptr<uint32_t> keysVecPtr = thrust::device_pointer_cast<uint32_t>(distPtr);
		// thrust::sort_by_key(thrust::cuda::par(*stream), keysVecPtr, keysVecPtr + this->trainDataSize, clsVec.begin());
		
		// ���� ������������� ����� ����� TOP-K �������
		// ��� ����� ������� ������ �������� ����� reduce_by_key ��� ������� ����� ������ �������
		// thrust::device_ptr<uint8_t> clsVecPtr = thrust::device_pointer_cast(clsVec.data());
		// �������� ����������� constant_iterator<uint8_t>
		// thrust::device_vector<uint8_t> onesVec(k, 1);
		// thrust::device_vector<uint8_t> clsOut(k);
		// thrust::device_vector<uint8_t> cntOut(k);
		// thrust::reduce_by_key(thrust::cuda::par(*stream), clsVecPtr, clsVecPtr + k * sizeof(uint8_t), onesVec.begin(), clsOut.begin(), cntOut.begin());
		// ���� ���-1 � ������������� ������ �� ��������� 
		// thrust::device_ptr<uint8_t> cntOutPtr = thrust::device_pointer_cast<uint8_t>(onesVec.data());
		// thrust::sort_by_key(thrust::cuda::par(*stream), cntOut.begin(), cntOut.end(), clsOut.begin());
		
		// ��������� ������������� ����� � ������� ������� ������ �� ����� �� ����
		// thrust::host_vector<uint8_t> clsHostVec(clsVec);
		// CharacterClassification cc;
		// cc.cls = static_cast<char>(clsHostVec[0]);
		// cc.loc = &mat;
		// result.push_back(cc);

		// �� �������� ���������� ������ � ����� ����� ����������
		rsp = cudaFreeAsync(requestedMatPtr, stream);
		CHECK_CUDA(rsp, true, "Could not free memory for input texture");
		rsp = cudaFreeAsync(distPtr, stream);
		CHECK_CUDA(rsp, true, "Could not allocate memory for input neighbours distances");
		rsp = cudaStreamDestroy(stream);
		CHECK_CUDA(rsp, true, "Could not shut down CUDA stream", stream);

	}

	return result;

}

KNNClassifier::~KNNClassifier(){

	cudaError_t rsp;

	rsp = cudaFree(this->trainDataPtr);
	CHECK_CUDA(rsp, true, "Could not free train data memory on GPU...");
	
	rsp = cudaFree(this->trainClsPtr);
	CHECK_CUDA(rsp, true, "Could not free train classifiers memory on GPU...");

}

void testOMP() {

	printf("Availiable OMP threads system-wide : %d\n", omp_get_max_threads());
	# pragma omp parallel for
	for (int idx = 0; idx < 10; idx++) {
		assert(omp_get_num_threads() > 1);
	}

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