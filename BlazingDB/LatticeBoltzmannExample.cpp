#include "LatticeBoltzmannExample.h"

#include <ctime>
#include <iostream>

LatticeBoltzmannExample::~LatticeBoltzmannExample()
{
	delete simulationParameters;
	delete thrustVectors;

	cudaCleanup();
}


LatticeBoltzmannExample::LatticeBoltzmannExample(const int argc, char **argv)
{

	real testDensity = 1.f; //Just a simple test density, doesn't matter for this example

	int xLength = atoi(argv[1]);

	if (xLength  < 10) //minimum domain size
		xLength = 10;

	int yLength = xLength; // square domain for problem

	int writeEvery = atoi(argv[2]);

	if (writeEvery < 0)
		writeEvery = 0; // ensure not negative. If so, set it to not write. 

	real relaxation = 3.f * 0.07f * (real)xLength / 3000.f + 0.5f; // related to viscosity of fluid. Close to 0.5 implies low viscosity

	simulationParameters = new SimulationParameters<real>(testDensity, relaxation, xLength, yLength, writeEvery);		
}


