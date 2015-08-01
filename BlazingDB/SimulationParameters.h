//Helps to be tidy

#include <limits>

template<typename T>
class SimulationParameters
{
public:

	SimulationParameters(T initialRho, T mu, T xDim, T yDim, int writeFreq) : 
		initialDensity(initialRho),
		relaxation(mu),
		xLength(xDim),
		yLength(yDim),
		writeEvery(writeFreq)
	{
		//checkDensity();
		//checkRelaxation();
		//checkLengths();

	}

	T initialDensity;
	T relaxation;
	int xLength;
	int yLength;
	int writeEvery;

private:

	static const T MIN_RELAXATION;
	static const T RELAXATION_LOW_WARNING_VALUE;
	static const T RELAXATION_HIGH_WARNING_VALUE;

	/*void checkDensity()
	{
		if (initialDensity <= std::numeric_limits<T>)
	}

	void checkRelaxation()
	{

	}
	
	checkLengths()
	{

	}*/

};

template <typename T>
const T SimulationParameters<T>::MIN_RELAXATION = 0.5;

template <typename T>
const T SimulationParameters<T>::RELAXATION_LOW_WARNING_VALUE = 0.6;

template <typename T>
const T SimulationParameters<T>::RELAXATION_HIGH_WARNING_VALUE = 2.0;