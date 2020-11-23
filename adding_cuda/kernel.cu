#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <windows.h>

using namespace std;

template <typename  T>
cudaError_t LeibnitzFormula(T* numbers, int N);

template <typename  T>
cudaError_t Sum(T*  numbers, int N, T SumCPU, LARGE_INTEGER TimeCPU);

template <typename  T>
__global__ void LeibnitzFormulaGPU(T* numbers, int N, int offset) {		// <- Nooo mo¿e prawie. Chcê wyznaczyæ Pi wiêc mno¿ê jeszcze razy 4
	int index = (blockIdx.x + offset)*blockDim.x + threadIdx.x;
	if (index < N)
		numbers[index] = (index % 2 == 0) ? (T)(4 / (T)(2 * index + 1)) : (T)(-4 / (T)(2 * index + 1));
}

template <typename  T>
__global__ void SumGPU(T* numbers, int N, int offset) {
	int Tindex = threadIdx.x;										// <-wspó³rzêdne dla shared'a
	int Bindex = blockIdx.x + offset;								// <-wspó³rzêdne bloku (miejsca na, które zostanie przeniesiona suma)
	int Gindex = (blockIdx.x + offset) * 2 * blockDim.x + threadIdx.x;	// <-wspó³rzêdne globalne
	extern __shared__ __align__(sizeof(T)) unsigned char s[];
	T *temp = reinterpret_cast<T *>(s);
	if (Gindex + blockDim.x < N)											// <-jeœli globalne wspó³rzêdne s¹ mniejsze od ilosci próbek...
		temp[Tindex] = numbers[Gindex] + numbers[Gindex + blockDim.x];		// <-weŸ wartoœæ do shared'a
	else if (Gindex < N)
		temp[Tindex] = numbers[Gindex];
	else
		temp[Tindex] = 0;
	__syncthreads();

	for (int i = blockDim.x / 2; i > 0; i /= 2) {	//<- i to po³owa bloku
		if (Tindex < i)
			temp[Tindex] += temp[Tindex + i];
		__syncthreads();
	}
	if (Tindex == 0)					// <-spe³nione tylko dla jednego w¹tku w bloku
		numbers[Bindex] = temp[Tindex];	// <-przypisanie wartoœci sumy danego bloku na komórkê o indeksie bloku z którego pochodzi
}

int main() {
	// o=====<+>=====<+>=====<+>=====<+>=====<+>=====<+>=====o //
	//					  DODAWANIE FLOAT'ÓW					   //
	// o=====<+>=====<+>=====<+>=====<+>=====<+>=====<+>=====o //
	int N = 343154432;
	//N /= 2;
	float* numbersf = new float[N];
	cudaError_t cudaStatus = LeibnitzFormula(numbersf, N);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "LeibnitzFormula failed!");
		return 1;
	}

	cout << "DODAWANIE FLOAT'OW" << endl;
	cout << "N: " << N << endl;


	LARGE_INTEGER StartCPUf, StopCPUf, TimeCPUf;
	LARGE_INTEGER frequency;
	QueryPerformanceFrequency(&frequency);
	QueryPerformanceCounter(&StartCPUf);



	float SumCPUf = 0;
	for (int i = 0; i < N; i++) {
		SumCPUf = SumCPUf + numbersf[i];
	}



	QueryPerformanceCounter(&StopCPUf);
	TimeCPUf.QuadPart = StopCPUf.QuadPart - StartCPUf.QuadPart;
	TimeCPUf.QuadPart = TimeCPUf.QuadPart * 1000000 / frequency.QuadPart / 1000;	//ms

	cout << setprecision(50) << "PI=\t\t3.141592653589793238462643383279502884197169399375105820" << endl;
	cout << "SumCPU= \t" << SumCPUf << endl;
	cout << "Czas dodawania na CPU wynosi:" << TimeCPUf.QuadPart << "ms" << endl;
	cout << endl;



	cudaStatus = Sum(numbersf, N, SumCPUf, TimeCPUf);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Sum failed!");
		cout << endl << cudaGetErrorString(cudaStatus) << endl;
		return 1;
	}


	delete[] numbersf;
	cout << endl;
	cout << endl;
	cout << endl;
	// o=====<+>=====<+>=====<+>=====<+>=====<+>=====<+>=====o //
	//					 DODAWANIE DOUBLE'I					   //
	// o=====<+>=====<+>=====<+>=====<+>=====<+>=====<+>=====o //
	N = N / 2;
	double* numbersd = new double[N];
	cudaStatus = LeibnitzFormula(numbersd, N);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "LeibnitzFormula failed!");
		return 1;
	}
	cout << "DODAWANIE DOUBLE'I" << endl;
	cout << "N: " << N << endl;

	LARGE_INTEGER StartCPUd, StopCPUd, TimeCPUd;
	QueryPerformanceFrequency(&frequency);
	QueryPerformanceCounter(&StartCPUd);



	double SumCPUd = 0;
	for (int i = 0; i < N; i++) {
		SumCPUd = SumCPUd + numbersd[i];
	}



	QueryPerformanceCounter(&StopCPUd);
	TimeCPUd.QuadPart = StopCPUd.QuadPart - StartCPUd.QuadPart;
	TimeCPUd.QuadPart = TimeCPUd.QuadPart * 1000000 / frequency.QuadPart / 1000;	//ms

	cout << setprecision(50) << "PI=\t\t3.141592653589793238462643383279502884197169399375105820" << endl;
	cout << "SumCPU= \t" << SumCPUd << endl;
	cout << "Czas dodawania na CPU wynosi: " << TimeCPUd.QuadPart << "ms" << endl;
	cout << endl;




	cudaStatus = Sum(numbersd, N, SumCPUd, TimeCPUd);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Sum failed!");
		cout << endl << cudaGetErrorString(cudaStatus) << endl;
		return 1;
	}






	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}

// Helper function for using CUDA to add vectors in parallel.
template <typename  T>
cudaError_t LeibnitzFormula(T* numbers, int N)
{
	T *dev_n = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	//for (;;) {						//<-Sprawdzajka do wyznaczania dok³adnej iloœci 
	//	cout << "N= " << N << endl;
	cudaStatus = cudaMalloc((void**)&dev_n, N * sizeof(T));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	//cudaFree(dev_n);
	//	N = N + 1;
	//}

	cudaStatus = cudaMemcpy(dev_n, numbers, N * sizeof(T), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	int ThreadsPerBlocks = 1024;
	int BlocksPerGrid = 65536;
	int BlocksAmount = (N%ThreadsPerBlocks == 0) ? N / ThreadsPerBlocks : N / ThreadsPerBlocks + 1;
	// Launch a kernel on the GPU with one thread for each element.
	LeibnitzFormulaGPU << <BlocksAmount, ThreadsPerBlocks >> > (dev_n, N, 0);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "LeibnitzFormulaGPU launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching LeibnitzFormulaGPU!\n", cudaStatus);
		goto Error;
	}
	//}
	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(numbers, dev_n, N * sizeof(T), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_n);
	return cudaStatus;
}

template <typename  T>
cudaError_t Sum(T* numbers, int N, T SumCPU, LARGE_INTEGER TimeCPU)
{
	T *dev_n = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}


	cudaStatus = cudaMalloc((void**)&dev_n, N * sizeof(T));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_n, numbers, N * sizeof(T), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy numbers failed!");
		goto Error;
	}



	int ThreadsPerBlocks = 32;		//<-rozmiar bloku
	int BlocksPerGrid = 65536;	//<-rozmiar siatki
	int SamplesAmount = N;		//<-iloœæ próbek
	int BlocksAmount = 0;		//<-liczba bloków
	int GridsAmount = 0;		//<-liczba siatek


	LARGE_INTEGER StartGPU, StopGPU, TimeGPU;
	LARGE_INTEGER frequency;
	QueryPerformanceFrequency(&frequency);
	QueryPerformanceCounter(&StartGPU);
	for (;;) {
		BlocksAmount = (SamplesAmount%ThreadsPerBlocks == 0) ? SamplesAmount / ThreadsPerBlocks : SamplesAmount / ThreadsPerBlocks + 1;	// <-liczba bloków
		GridsAmount = (BlocksAmount%BlocksPerGrid == 0) ? BlocksAmount / BlocksPerGrid : BlocksAmount / BlocksPerGrid + 1;			// <-Liczenie siatek
		for (int i = 0; i < GridsAmount; i++) {
			SumGPU << <BlocksPerGrid, ThreadsPerBlocks, ThreadsPerBlocks * sizeof(T) >> > (dev_n, SamplesAmount, i*BlocksPerGrid);
		}
		//SamplesAmount = BlocksAmount2;																	//<-Ka¿dy blok daje sumê, a te trzeba zsumowaæ, wiêc stanowi¹ one nowe próbki
		SamplesAmount = BlocksAmount;
		if (BlocksAmount == 1)																		//<- Jeœli liczba bloków by³a równa 1 to znaczy, ¿e wszystkie próbki siê zsumowa³y
			break;
	}
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "SumGPU launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching SumGPU!\n", cudaStatus);
		goto Error;
	}

	QueryPerformanceCounter(&StopGPU);
	TimeGPU.QuadPart = StopGPU.QuadPart - StartGPU.QuadPart;
	TimeGPU.QuadPart = TimeGPU.QuadPart * 1000000 / frequency.QuadPart / 1000;	//ms


	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(numbers, dev_n, N * sizeof(T), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cout << setprecision(50) << "PI=\t\t3.141592653589793238462643383279502884197169399375105820" << endl;
	cout << "SumGPU= \t" << numbers[0] << endl;
	cout << "Czas dodawania na GPU wynosi: " << TimeGPU.QuadPart << "ms" << endl;
	cout << endl << "GPU jest " << (T)(TimeCPU.QuadPart / (T)TimeGPU.QuadPart) << "x szybsze od CPU" << endl;
	cout << "Roznica miedzy suma liczona na CPU i GPU wynosi: " << SumCPU - numbers[0] << endl;

Error:
	cudaFree(dev_n);

	return cudaStatus;
}
