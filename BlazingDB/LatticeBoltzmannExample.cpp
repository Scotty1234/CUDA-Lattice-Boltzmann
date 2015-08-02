#include "LatticeBoltzmannExample.h"

#include <iostream>

#define ARGC_NOT_EQUAL_TO_TWO 1
#define RELAXATION_LESS_THAN_MIN_RELAXATION 2


const int LatticeBoltzmannExample::EXPECTED_NUM_ARGS = 5;

/*LatticeBoltzmannExample::~LatticeBoltzmannExample()
{
	delete simulationParameters;
	delete thrustVectors;

	//cudaCleanUp();
}*/


LatticeBoltzmannExample::LatticeBoltzmannExample(const int argc, char **argv)
{

	real testDensity = 1.f; //Just a simple test density, doesn't matter for this example
	real relaxation = atof(argv[1]);

	if (relaxation < 0.5f)
		relaxation = 1.0f;

	int xLength = atoi(argv[2]);

	if (xLength == 0)
		xLength = 10;



	int yLength = atoi(argv[3]);

	if (yLength == 0)
		yLength = 10;

	int writeEvery = atoi(argv[4]);



	//std::unique_ptr<SimulationParameters<real>> simulationParameters(new SimulationParameters<real>(testDensity, relaxation, xLength, yLength, writeEvery));

	simulationParameters = new SimulationParameters<real>(testDensity, relaxation, xLength, yLength, writeEvery);
	

	
}


LatticeBoltzmannExample::~LatticeBoltzmannExample()
{
}

//############################




