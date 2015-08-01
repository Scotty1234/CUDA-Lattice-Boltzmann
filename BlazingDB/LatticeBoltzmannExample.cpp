#include "LatticeBoltzmannExample.h"

#include <iostream>

#define ARGC_NOT_EQUAL_TO_TWO 1
#define RELAXATION_LESS_THAN_MIN_RELAXATION 2


const int LatticeBoltzmannExample::EXPECTED_NUM_ARGS = 5;


LatticeBoltzmannExample::LatticeBoltzmannExample(const int argc, char **argv)
{
	checkArguments(argc, argv);

	real testDensity = 1.f; //Just a simple test density, doesn't matter for this example
	real relaxation = atof(argv[1]);
	int xLength = atoi(argv[2]);
	int yLength = atoi(argv[3]);
	int writeEvery = atoi(argv[4]);

	//std::unique_ptr<SimulationParameters<real>> simulationParameters(new SimulationParameters<real>(testDensity, relaxation, xLength, yLength, writeEvery));

	simulationParameters = new SimulationParameters<real>(testDensity, relaxation, xLength, yLength, writeEvery);
	

	
}


LatticeBoltzmannExample::~LatticeBoltzmannExample()
{
}

//############################

void LatticeBoltzmannExample::checkArguments(const int argc,  char **argv)
{

	if (! (argc == EXPECTED_NUM_ARGS))
	{
		std::cout << "Expected two arguments. " << argc << " supplied. First is the relaxation parameter, second is the write to disk frequency.\n";
		exit(ARGC_NOT_EQUAL_TO_TWO);
	}

}

void LatticeBoltzmannExample::simulate()
{

}
