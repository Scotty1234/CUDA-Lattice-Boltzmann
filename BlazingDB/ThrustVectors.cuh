#pragma once

#include "SimpleVTKWriter.h"
#include "cuda_runtime.h"

#include <thrust\device_vector.h>
#include <thrust\host_vector.h>

#define WIN32_LEAN_AND_MEAN
#include <Windows.h> //to create a directory
#include <ctime>
#include <sstream>
#include <string>


template <class T>
class ThrustVectors
{
public:

	ThrustVectors(const int nx, const int ny) : m_nx(nx), m_ny(ny)
	{
		//resize the thrust vectors to required size
		dv_distributionFunctions.resize(m_nx * m_ny * NUM_COMPONENT_DIRECTIONS);
		dv_tmpDistributionFunctions = dv_distributionFunctions;

		hv_macroscopicVariables.resize(MACROVAR_COUNT * m_nx * m_ny);
		dv_macroscopicVariables.resize(MACROVAR_COUNT * m_nx * m_ny);

		hv_vorticity.resize(m_nx * m_ny);
		dv_vorticity.resize(m_nx * m_ny);

		//create dir for writing vorticity

		//get the time
			time_t currentTime;
			time(&currentTime);
			struct tm* timeInfo = localtime(&currentTime);

			const int maxSize = 80;

			char directoryName[maxSize];

			if (strftime(directoryName, maxSize, "%d-%m-%H-%M-%S-results", timeInfo) == 0) // put the day into a day-month-hour-minute-seconds format to create the folder name
			{
				std::cerr << "Creating directory failed. Exceeded buffer length for time. Error number: " << errno << std::endl; //shouldn't fail, but good programming always has error checking
				exit(EXIT_FAILURE);
			}

			if (!CreateDirectory(directoryName, NULL)) // Usually use boost, but this will do for now
			{
				std::cerr << "Could not create directory " << directoryName << std::endl;
				exit(EXIT_FAILURE);
			}

			simpleVTKWriter = new SimpleVTKWriter<T>(directoryName, thrust::raw_pointer_cast(hv_vorticity.data()), m_nx, m_ny,"vorticity");

	}

	~ThrustVectors() {}

	//get raw pointers to thrust vectors for use in kernels etc
	T*  getDistributionsFunctionsPtr() { return thrust::raw_pointer_cast(dv_distributionFunctions.data()); } 
	T* getTmpDistributionsFunctionsPtr(){ return thrust::raw_pointer_cast(dv_tmpDistributionFunctions.data()); }  
	T* getMacroscopicVariablesPtr(){ return thrust::raw_pointer_cast(dv_macroscopicVariables.data()); }  
	T* getVorticityDevicePtr(){ return thrust::raw_pointer_cast(dv_vorticity.data()); }  

	void downloadVorticity()
	{
		//copy vorticity device data to host data
		hv_vorticity = dv_vorticity;
		
	}

	void swapPtrs()
	{
		dv_distributionFunctions.swap(dv_tmpDistributionFunctions); // swap the pointers instead of doing copies, more efficient. 
	}

	void write(const int time)
	{
		std::stringstream stringStream; //use string stream to construct filename

		stringStream << "vorticity_" << time;

		std::string fileName(stringStream.str()); // convert to string 

		simpleVTKWriter->write(fileName.c_str());	 // string to c string as vtkwriter likes only c strings. 
	}

private:

	thrust::device_vector<T> dv_distributionFunctions;
	thrust::device_vector<T> dv_tmpDistributionFunctions;

	thrust::device_vector<T> dv_macroscopicVariables;
	thrust::host_vector<T> hv_macroscopicVariables;

	thrust::device_vector<T> dv_vorticity;
	thrust::host_vector<T> hv_vorticity;

	SimpleVTKWriter<T> *simpleVTKWriter; // writer object for vorticity

	static const int NUM_COMPONENT_DIRECTIONS;
	static const int MACROVAR_COUNT;

	//dimensions of the arrays
	 const int m_nx;
	 const int m_ny;

};

template <class T>
const int ThrustVectors<T>::NUM_COMPONENT_DIRECTIONS = 9;

template <class T>
const int ThrustVectors<T>::MACROVAR_COUNT= 3;
