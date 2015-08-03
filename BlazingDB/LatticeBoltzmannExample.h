#pragma once

#include "SimulationParameters.h"
#include "ThrustVectors.cuh"

//choose precision and recompile. Double is slower, more so for home user gpus
typedef float real;
//typedef double real;

class LatticeBoltzmannExample
{
public:
	LatticeBoltzmannExample(const int argc,  char **argv);
	~LatticeBoltzmannExample();

	void run(); // sets up cuda and runs the simulation
	

private:

	//=====CUDA members and variables ====

	//set up memory and problem
	void cudaLBInit(const dim3& blockSize, const dim3& gridSize);
	void cudaCleanup();

	//simple check to see if device is suitable
	bool isCudaCompatible(); 

	//starts and calculates the fluid dynamics
	void simulate(const dim3& blockSize, const dim3& gridSize);

	//synchronises host with device and checks for kernel errors
	void waitForDevice();

	//group the thrust vectors together, with some helpful methods and abstract them from the main class
	ThrustVectors<real> *thrustVectors;

	//=======

	//holder for the simulation parameters
	SimulationParameters<real> *simulationParameters;

};

