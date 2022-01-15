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
		// ��� �������. �������� ������� 0xFF
		uint8_t xorEd = trainPtr[idx] ^ inputPtr[idx % resSq];
		#ifdef DBG_CUDA_KERNEL
		printf("dist: index:%d; idx: %d; stride: %d; train data at idx: %d; input at idx: %d; xor at idx: %d; char: %d\n", 
			index, idx, stride, trainPtr[idx], inputPtr[idx % resSq], xorEd, idx / resSq
		);
		#endif
		// ��������� � ������ ����������
		// ��������� ��� ��� race condition :( ��� �� �� ������� �� ����
		distPtr[idx] = xorEd / 255; // �� 0 �� 1
	}
}

__global__ void dist_reduce(uint32_t* distPtr, uint32_t* distRedPtr, int res, int tds) {
	
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int resSq = res * res;
	if (index > tds) {
		return;
	}

	#pragma unroll
	for (int idxRes = 0; idxRes < resSq; idxRes++) {
		#ifdef DBG_CUDA_KERNEL
		printf("dist_reduce: index: %d; idx: %d; idxRes: %d\n", index, idxRes);
		#endif
		distRedPtr[index] += distPtr[(index * resSq) + idxRes];
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

		// ��������� �� ������ ������
		cv::threshold(mat, mat, 178, 255, cv::ThresholdTypes::THRESH_BINARY);

		cv::Mat flat = mat.reshape(1, mat.total() * mat.channels());
		std::vector<uchar> vec = mat.isContinuous() ? flat : flat.clone();

		rsp = cudaMemcpyAsync(this->trainDataPtr + idx * this->dataChunkSize, vec.data(), this->dataChunkSize, cudaMemcpyKind::cudaMemcpyHostToDevice);
		CHECK_CUDA(rsp, true, "Cannot load file ", fileNames[idx]);

		// ���� ��� �� � �� �������� ��������, ������� �� ��� � �����. ����������
		// �� *nix ��� ������� �������� �� �����, ����� ���������
		const std::string fname = fileNames[idx].substr(fileNames[idx].find_last_of("//") + 1);
		char cls;
		if (fname.find('.') == 1) {
			cls = fname[0];
		}
		else if ((fname.find('.') == 2) && (fname.find('_') == 0)) {
			cls = fname[1];
		}
		else {
			std::string errText = "Malformed test data sample name: " + fname;
			throw std::exception(errText.c_str());
		}

		// const char cls = fileNames[idx][fileNames[idx].find_last_of("//") + 1];
		rsp = cudaMemsetAsync(this->trainClsPtr + idx, cls, 1);
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

}

void uploadMatToDev(thrust::device_vector<uint8_t>& dVec, int offset, cv::Mat& mat) {
	// ���������, ��� �������� ����� ������ ���
	assert(mat.type() == CV_8UC1);
	cv::Mat flat = mat.reshape(1, mat.total() * mat.channels());
	std::vector<uchar> vec = mat.isContinuous() ? flat : flat.clone();
	thrust::host_vector<uint8_t> requestedMatHVec(vec);
	thrust::copy(requestedMatHVec.begin(), requestedMatHVec.end(), dVec.begin() + offset);
}

void printDeviceTexture(thrust::device_vector<uint8_t>& dVec, int res, int offset) {
	thrust::host_vector<uint8_t> matsHVec(dVec);
	printf("Image in device memory (8 bit single channel):\n");
	for (int y = 0; y < res; y++) {
		for (int x = 0; x < res; x++) {
			printf("%03d|", matsHVec[y * res + x + (offset * res * res)]);
		}
		printf("\n");
	}
	printf("\n");
}

template<typename T>
void printDeviceVector(thrust::device_vector<T> dVec) {
	thrust::host_vector<T> hVec(dVec);
	printf("Vector size (%d) : ", hVec.size());
	for (T& elem : hVec) {
		printf("%d|", elem);
	}
	printf("\n");
}

void checkCudaAsyncMemMgmtSupport() {
	int attr;
	cudaError_t rsp;
	rsp = cudaDeviceGetAttribute(&attr, cudaDevAttrMemoryPoolsSupported, 0);
	CHECK_CUDA(rsp, true, "Can't probe for memory pools support");
	printf("Device supports memory pooling (async memset): %d\n", attr);
}

std::vector<CharacterClassification> KNNClassifier::classifyCharacters(std::vector<ExtractedCharacter>& chars, int k)
{
	// ������ CUDA Streams �� ���������� ���� �� �������������
	// ����� ���������� ���������� �����
	cudaError_t rsp;
	std::vector<cudaStream_t> streams;
	thrust::device_vector<uint8_t> verdict(chars.size());
	
	const int texSize = this->resolution * this->resolution;

	// ��������� ���������� � ������ GPU ��� chars
	const std::chrono::steady_clock::time_point startLoad = std::chrono::steady_clock::now();
	thrust::device_vector<uint8_t> matsDVec(chars.size() * texSize);
	#pragma omp parallel for
	for (int idx = 0; idx < chars.size(); idx++) {
		// ����� ������ ������������ streams, �������� �� ������������ �����. ���������� �������
		uploadMatToDev(matsDVec, idx * texSize, chars[idx].mat);
	}

	const std::chrono::steady_clock::time_point startProcessing = std::chrono::steady_clock::now();
	std::cout << chars.size() << " characters to analyze uploaded in " <<
		std::chrono::duration_cast<std::chrono::milliseconds>(startProcessing - startLoad).count() << " ms. ";

	for (int idx = 0; idx < chars.size(); idx++) {
		
		// ������� ���������� �����
		cudaStream_t stream;
		rsp = cudaStreamCreate(&stream);
		CHECK_CUDA(rsp, true, "Could not create stream");

		streams.push_back(stream);
		// �������� � �������� ����� ������� ����� �������� � ������ GPU
		thrust::device_vector<uint8_t> requestedMatDVec(
			matsDVec.begin() + texSize * idx, 
			matsDVec.begin() + texSize * (idx + 1)
		);
		// cv::Mat& mat = chars[idx].mat;
		// cv::Mat flat = mat.reshape(1, mat.total() * mat.channels());
		// std::vector<uchar> vec = mat.isContinuous() ? flat : flat.clone();
		// thrust::host_vector<uint8_t> requestedMatHVec(vec);
		// thrust::copy(requestedMatHVec.begin(), requestedMatHVec.end(), requestedMatDVec.begin());

		// ���� ��� ��, �������������� ������, � ������� ����� ��������� ����������
		// ����� �������������� ���������� � �������. �������� ��� �������� �������
		thrust::device_vector<uint32_t> distVec(this->trainDataSize * this->resolution * this->resolution);
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

		uint8_t* reqMatRawPtr = thrust::raw_pointer_cast(requestedMatDVec.data());
		uint32_t* distRawPtr = thrust::raw_pointer_cast(distVec.data());
		
		const int threadsPerBlock = 256; // ���������� (?) ��� ���� ����� (MX130)

		dist<<<4, threadsPerBlock, 0, stream>>> (this->trainDataPtr, reqMatRawPtr, distRawPtr, this->resolution, this->trainDataSize);
		rsp = cudaGetLastError();
		CHECK_CUDA(rsp, true, "Kernel launch was unsuccessful", "dist kernel");

		// ��������� ���������� XOR ��� ������ �������� ����������
		thrust::device_vector<uint32_t> distRedVec(this->trainDataSize);
		uint32_t* distRedVecRawPtr = thrust::raw_pointer_cast(distRedVec.data());

		const int blockCount = (this->trainDataSize / threadsPerBlock) + 1;
		dist_reduce <<<blockCount, threadsPerBlock, 0, stream >>> (distRawPtr, distRedVecRawPtr, this->resolution, this->trainDataSize);
		rsp = cudaGetLastError();
		CHECK_CUDA(rsp, true, "Kernel launch was unsuccessful", "dist_reduce kernel");

		// ���� ������� TOP-K ������� �������������� �����������
		thrust::sort_by_key(thrust::cuda::par.on(stream), distRedVec.begin(), distRedVec.end(), clsCopyVec.begin());
		thrust::sort(thrust::cuda::par.on(stream), clsCopyVec.begin(), clsCopyVec.begin() + k);
		
		// ���� ������������� ����� ����� TOP-K �������
		// ��� ����� ������� ������ �������� ����� reduce_by_key ��� ������� ����� ������ �������
		// �������� ����������� constant_iterator<uint8_t>
		//thrust::device_vector<uint8_t> onesVec(this->trainDataSize, 1);
		thrust::constant_iterator<uint8_t> onesConstIter(1);
		thrust::device_vector<uint8_t> clsOut(k);
		thrust::device_vector<uint8_t> cntOut(k);
		thrust::reduce_by_key(thrust::cuda::par.on(stream), 
			clsCopyVec.begin(), clsCopyVec.begin() + k, 
			onesConstIter, clsOut.begin(), cntOut.begin()
		);

		// ���� ���-1 � ������������� ������ �� ��������� 
		thrust::sort_by_key(
			thrust::cuda::par.on(stream), cntOut.begin(), cntOut.end(), 
			clsOut.begin(), thrust::greater<uint8_t>()
		);

		// �������� ��������� �������������
		verdict[idx] = clsOut[0];

	}
	printf("All streams are loaded\n");

	// ���� ���� ���������� ��� ������ � ����������� ��
	rsp = cudaDeviceSynchronize();
	CHECK_CUDA(rsp, true, "Cannot synchronize the CUDA Device");
	for (auto& stream : streams) {
		rsp = cudaStreamDestroy(stream);
		CHECK_CUDA(rsp, true, "Could not properly shut down CUDA stream", stream);
	}

	const std::chrono::steady_clock::time_point endProcessing = std::chrono::steady_clock::now();
	std::cout << chars.size() << " characters analyzed in " <<
		std::chrono::duration_cast<std::chrono::milliseconds>(endProcessing - startProcessing).count() << " ms. ";

	thrust::host_vector<uint8_t> hVerdict(verdict);
	std::vector<CharacterClassification> result;
	for (int idx = 0; idx < chars.size(); idx++) {
		CharacterClassification cc;
		cc.cls = hVerdict[idx];
		cc.x = chars[idx].x;
		cc.y = chars[idx].y;
		cc.w = chars[idx].w;
		cc.h = chars[idx].h;
		result.push_back(cc);
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