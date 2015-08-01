#include "LatticeBoltzmannExample.h"

#include "cuda_runtime.h"

__constant__ float relaxationDevice; //Probably not worth it for one float, but still, all threads will use this. 

void LatticeBoltzmannExample::cudaLBInit()
{
	//std::unique_ptr<ThrustVector<real>> thrustVectors(new ThrustVector<real>(simulationParameters->xLength, simulationParameters->yLength));

	thrustVectors = new ThrustVector<real>(simulationParameters->xLength, simulationParameters->yLength);

	cudaMemcpyToSymbol(&relaxationDevice, &(simulationParameters->relaxation), sizeof(simulationParameters->relaxation));
}

void LatticeBoltzmannExample::cudaRun()
{
	if (!isCudaCompatible())
	{
		exit(5);
	}



	cudaLBInit();
}

bool LatticeBoltzmannExample::isCudaCompatible()
{
	cudaDeviceProp properties;

	int count;

	cudaError_t cudaError =cudaGetDeviceCount(&count);
	
	if (cudaError == cudaErrorNoDevice)
	{
		std::cout << "Error: No cuda capable device found.\n";
		return false;
	}
	else if (cudaError == cudaErrorInsufficientDriver)
	{
		std::cout << "Error: Insufficient cuda driver.\n";
		return false;
	}

	return true;
}

void LatticeBoltzmannExample::waitForDevice()
{
	if (cudaDeviceSynchronize() != cudaSuccess)
		cudaError();
}