#include "LatticeBoltzmannExample.h"

#include <cuda_runtime.h>
//#include <helper_functions.h> 
//#include <helper_cuda.h>

#include "kernels.cuh"



#define MEGA

//__constant__ float initialDensityDevice;
__constant__ real relaxationDevice; //Probably not worth it for one float, but still, all threads will use this. 

void LatticeBoltzmannExample::cudaLBInit(const dim3& blockSize, const dim3& gridSize)
{
	std::cout << "Initialising...\n";
	//std::unique_ptr<ThrustVector<real>> thrustVectors(new ThrustVector<real>(simulationParameters->xLength, simulationParameters->yLength));

	thrustVectors = new ThrustVector<real>(simulationParameters->xLength, simulationParameters->yLength);

	cudaMemcpyToSymbol(&relaxationDevice, &(simulationParameters->relaxation), sizeof(simulationParameters->relaxation));
	//cudaMemcpyToSymbol(&initialDensityDevice, &(simulationParameters->initialDensity), sizeof(simulationParameters->initialDensity));


	initialise<real><<<blockSize, gridSize>>>(simulationParameters->initialDensity, thrustVectors->getDistributionsFunctionsPtr());
	waitForDevice();


}

void LatticeBoltzmannExample::cudaRun()
{

}

void LatticeBoltzmannExample::simulate(const dim3& blockSize, const dim3& gridSize)
{
	for (int t = 0; t < 100; t++)
	{
		std::cout << "Time..............." << t << std::endl;
		simulateIteration<real><<< blockSize, gridSize>>>(thrustVectors->getDistributionsFunctionsPtr(), 
														  thrustVectors->getMacroscopicVariablesPtr(),
														  &relaxationDevice, 
														  thrustVectors->getTmpDistributionsFunctionsPtr(), 
														  true, t);
		waitForDevice();
	

		thrustVectors->swapPtrs();

		//thrustVectors->downloadMacroscopicVariables();

		


		
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

	std::cout << properties.name 
		      << "\nCompute major version: " << properties.major 
			  << "\nClock rate: " << properties.clockRate << " Mhz"; //could compare devices to find the best one for the calculations

	return true;
}

void LatticeBoltzmannExample::setupCuda()
{

	if (!isCudaCompatible())
	{
		exit(5);
	}

	dim3 blockSize(simulationParameters->xLength, 1, 1);
	dim3 gridSize(simulationParameters->yLength, 1, 1);

	cudaLBInit(blockSize, gridSize);

	simulate(blockSize, gridSize);

}

void LatticeBoltzmannExample::waitForDevice()
{
	cudaError_t cudaError = cudaDeviceSynchronize();
	if (cudaError != cudaSuccess)
		std::cout << cudaGetErrorString(cudaError) << std::endl;

	//cudaGetLastError();
}