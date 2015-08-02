#pragma once

#include "cuda_runtime.h"

#include <thrust\device_vector.h>
#include <thrust\host_vector.h>

#define WIN32_LEAN_AND_MEAN
#include <Windows.h> // for create directory..I prefer boost but I guess this will do!

#include <rapidxml\rapidxml.hpp>



template <class T>
class ThrustVector
{
public:

	ThrustVector(const int nx, const int ny) : m_nx(nx), m_ny(ny)
	{

		dv_distributionFunctions.resize(m_nx * m_ny * NUM_COMPONENT_DIRECTIONS);
		dv_tmpDistributionFunctions = dv_distributionFunctions;

		hv_macroscopicVariables.resize(MACROVAR_COUNT * m_nx * m_ny);
		dv_macroscopicVariables.resize(MACROVAR_COUNT * m_nx * m_ny);

		//CreateDirectory(".\\results", NULL);

	}

	~ThrustVector() {}

	T& operator() (const unsigned int x, const unsigned int y)
	{
		return dv_macroscopicVariables[x * m_nx + y];
	}

	T* getDistributionsFunctionsPtr(){ return thrust::raw_pointer_cast(dv_distributionFunctions.data()); }
	T* getTmpDistributionsFunctionsPtr(){ return thrust::raw_pointer_cast(dv_tmpDistributionFunctions.data()); }
	T* getMacroscopicVariablesPtr(){ return thrust::raw_pointer_cast(dv_macroscopicVariables.data()); }

	void downloadMacroscopicVariables()
	{
		hv_macroscopicVariables = dv_macroscopicVariables;
	}

	void swapPtrs()
	{
		dv_distributionFunctions.swap(dv_tmpDistributionFunctions); // swap the pointers instead of doing copies.
	}

	void write(const int time)
	{
		rapidxml::xml_document<> doc;
		



	}

private:

	thrust::device_vector<T> dv_distributionFunctions;
	thrust::device_vector<T> dv_tmpDistributionFunctions;

	thrust::device_vector<T> dv_macroscopicVariables;
	thrust::host_vector<T> hv_macroscopicVariables;

	static const int NUM_COMPONENT_DIRECTIONS;
	static const int MACROVAR_COUNT;

	 const int m_nx;
	 const int m_ny;

};

template <class T>
const int ThrustVector<T>::NUM_COMPONENT_DIRECTIONS = 9;

template <class T>
const int ThrustVector<T>::MACROVAR_COUNT= 3;

	/*
public:
	ThrustVector(const unsigned int nx, const unsigned int ny) : m_nx(nx), m_ny(ny)
	{

	}

	void hostToDeviceData()
	{
		deviceVector = hostVector;
	}

	void deviceToHostData()
	{
		hostVector = deviceVector;
	}

	T* getGPUPtr() { return thrust::raw_pointer_cast(deviceVector.data()); };

	ThrustVector operator()(const unsigned int x, const unsigned int y)
	{
		return hostVector[];
	}

private:

	thrust::device_vector<T> deviceVector;
	thrust::host_vector<T> hostVector;

	const unsigned int m_stride;
	const unsigned int m_nx;
	const unsigned int m_ny;
};
*/