#include "LatticeBoltzmannExample.h"

#include <cuda_runtime.h>

#include "kernels.cuh"

__constant__ real relaxationDevice; //Probably not worth it for one float, but still, all threads will use this. 

void LatticeBoltzmannExample::cudaLBInit(const dim3& blockSize, const dim3& gridSize)
{
	std::cout << "Initialising...\n";

	thrustVectors = new ThrustVectors<real>(simulationParameters->xLength, simulationParameters->yLength);

	//copy the constant memory to device
	cudaMemcpyToSymbol(&relaxationDevice, &(simulationParameters->relaxation), sizeof(simulationParameters->relaxation));

	/*if (status != cudaSuccess)
	{ }
			std::cout << cudaGetErrorString(cudaError) << std::endl;*/

	initialise<real><<<blockSize, gridSize>>>(simulationParameters->initialDensity, thrustVectors->getDistributionsFunctionsPtr());
	waitForDevice();

}

void LatticeBoltzmannExample::cudaCleanup()
{
	cudaError_t cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess)
		std::cerr << "Error: cudaDeviceReset failed.\n";

}

void LatticeBoltzmannExample::simulate(const dim3& blockSize, const dim3& gridSize)
{

	int writeEvery = simulationParameters->writeEvery;

	bool store;

	const int MAX_TIME = 20000;

	for (int t = 0; t < MAX_TIME; t++)
	{
		if (!writeEvery)
			store = false;
		else
			store = (t % writeEvery  == 0);

		simulateIteration<real><<< blockSize, gridSize>>>(thrustVectors->getDistributionsFunctionsPtr(), 
														  thrustVectors->getMacroscopicVariablesPtr(),
														  &relaxationDevice, 
														  thrustVectors->getTmpDistributionsFunctionsPtr(), 
														  store, t);
		waitForDevice();
	
		thrustVectors->swapPtrs();

		if (store)
		{
			computeVorticity<real> <<<blockSize, gridSize >> >(thrustVectors->getMacroscopicVariablesPtr(), thrustVectors->getVorticityDevicePtr());
			waitForDevice();

			std::cout << "Retriving vorticity from device and saving at time " << t << "...\n";
			thrustVectors->downloadVorticity();
			
			thrustVectors->write(t);
			
		}
		
	}
}

bool LatticeBoltzmannExample::isCudaCompatible()
{
	

	int count;

	cudaError_t cudaError =cudaGetDeviceCount(&count);
	
	if (cudaError != cudaSuccess)
	{
		std::cout << cudaGetErrorString(cudaError);
		return false;
	}

	std::cout << "Cuda device found.\n";

	cudaDeviceProp properties;

	cudaGetDeviceProperties(&properties, 0); //Let's just stick to the first device found for now. 

	double khz = 1000;

	double clockRate = properties.clockRate / khz;


	std::cout << properties.name 
		      << "\nCompute major version: " << properties.major 
			  << "\nClock rate: " << clockRate << " KHz\n"; //could compare devices to find the best one for the calculations

	return true;
}

void LatticeBoltzmannExample::run()
{

	if (!isCudaCompatible())
		exit(EXIT_FAILURE);

	//map x strips onto threads. Will fail of course if exceeding the threads per block limit.
	dim3 blockSize(simulationParameters->xLength, 1, 1);

	//map y direction onto blocks
	dim3 gridSize(simulationParameters->yLength, 1, 1);

	cudaLBInit(blockSize, gridSize);

	simulate(blockSize, gridSize);

}

void LatticeBoltzmannExample::waitForDevice()
{
	cudaError_t cudaError = cudaDeviceSynchronize();
	if (cudaError != cudaSuccess)
		std::cout << cudaGetErrorString(cudaError) << std::endl;
}