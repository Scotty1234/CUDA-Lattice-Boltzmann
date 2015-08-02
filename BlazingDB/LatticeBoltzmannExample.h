#pragma once

#include "SimulationParameters.h"
#include "ThrustVector.cuh"

#include <memory>



typedef float real;
//typedef double real;

class LatticeBoltzmannExample
{
public:
	LatticeBoltzmannExample(const int argc,  char **argv);
	~LatticeBoltzmannExample();

	void setupCuda();
	

private:

	//=====CUDA members and variables ====

	void cudaLBInit(const dim3& blockSize, const dim3& gridSize);
	void cudaRun();
	bool isCudaCompatible();
	void simulate(const dim3& blockSize, const dim3& gridSize);
	void waitForDevice();

	//std::unique_ptr<ThrustVector<real>> thrustVectors;

	ThrustVector<real> *thrustVectors;

	//=======

	void checkArguments(const int argc,  char *argv[]);
	void writeToDisk();
	
	//std::unique_ptr<SimulationParameters<real>> simulationParameters; 

	SimulationParameters<real> *simulationParameters;

	static const int EXPECTED_NUM_ARGS;



};

