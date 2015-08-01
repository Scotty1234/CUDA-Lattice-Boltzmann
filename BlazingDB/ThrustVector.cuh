#include <thrust\device_vector.h>
#include <thrust\host_vector.h>

template <class T>
struct ThrustVector
{
public:

	ThrustVector(const int nx, const int ny) : m_nx(nx), m_ny(ny)
	{
		dv_distributionFunctions.resize(m_nx * m_ny * NUM_COMPONENT_DIRECTIONS);
		dv_tmpDistributionFunctions = dv_distributionFunctions;

		hv_macroscopicVariables.resize(m_nx * m_ny);
		dv_macroscopicVariables = hv_macroscopicVariables;
	}

	static T* getGPUPtr(thrust::device_vector dVector) { return thrust::raw_pointer_cast(dVector.data()); }


	thrust::device_vector<T> dv_distributionFunctions;
	thrust::device_vector<T> dv_tmpDistributionFunctions;

	thrust::device_vector<T> dv_macroscopicVariables;
	thrust::host_vector<T> hv_macroscopicVariables;

	static const int NUM_COMPONENT_DIRECTIONS;

	const int m_nx;
	const int m_ny;

};

template <typename T>
const int ThrustVector<T>::NUM_COMPONENT_DIRECTIONS = 9;

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