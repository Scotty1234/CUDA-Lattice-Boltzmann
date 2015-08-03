//Helps to be tidy

#pragma once
#include <limits>

template <class T>
struct SimulationParameters
{
public:

	SimulationParameters(T initialRho, T mu, T xDim, T yDim, int writeFreq) : 
		initialDensity(initialRho),
		relaxation(mu),
		xLength(xDim),
		yLength(yDim),
		writeEvery(writeFreq)
	{


	}

	T initialDensity;
	T relaxation;
	int xLength;
	int yLength;
	int writeEvery;

private:


};
