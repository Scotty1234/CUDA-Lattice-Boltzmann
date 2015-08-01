#include "cuda_runtime.h"

#define CENTRAL_WEIGHT (4.f / 9.f)
#define CROSS_WEIGHT (1.f / 9.f)
#define DIAGONAL_WEIGHT (1.f / 36.f)

#define NEGATIVE_UNIT_X -1.f
#define NEGATIVE_UNIT_Y -1.f

#define POSITIVE_UNIT_X 1.f
#define POSITIVE_UNIT_Y 1.f

#define ZERO_UNIT_VELOCITY 0.f

template <typename T>
__device__ void calculateDistributionFcns(const T density, const T *zerothDistributionFunctionComponent, const T vx, const T vy, const int x, const int y)
{

	//(*dfPtr)(0, x, y, z) = 1.f / 3.f * density * (1 - (1.5f * (vx*vx + vy*vy + vz*vz)));


	*zerothDistributionFunctionComponent       = equilibriumDistributionComponent(density, ZERO_UNIT_VELOCITY, ZERO_UNIT_VELOCITY, vx, vy, CENTRAL_WEIGHT);
	*(zerothDistributionFunctionComponent + 1) = equilibriumDistributionComponent(density, POSITIVE_UNIT_X   , ZERO_UNIT_VELOCITY, vx, vy, CROSS_WEIGHT);
	*(zerothDistributionFunctionComponent + 2) = equilibriumDistributionComponent(density, ZERO_UNIT_VELOCITY, POSITIVE_UNIT_Y   , vx, vy, CROSS_WEIGHT);
	*(zerothDistributionFunctionComponent + 3) = equilibriumDistributionComponent(density, NEGATIVE_UNIT_X   , NEGATIVE_UNIT_Y   , vx, vy, CROSS_WEIGHT);
	*(zerothDistributionFunctionComponent + 4) = equilibriumDistributionComponent(density, ZERO_UNIT_VELOCITY, NEGATIVE_UNIT_Y   , vx, vy, CROSS_WEIGHT);
	*(zerothDistributionFunctionComponent + 5) = equilibriumDistributionComponent(density, POSITIVE_UNIT_X   , POSITIVE_UNIT_Y   , vx, vy, DIAGONAL_WEIGHT);
	*(zerothDistributionFunctionComponent + 6) = equilibriumDistributionComponent(density, NEGATIVE_UNIT_X   , POSITIVE_UNIT_Y   , vx, vy, DIAGONAL_WEIGHT);
	*(zerothDistributionFunctionComponent + 7) = equilibriumDistributionComponent(density, NEGATIVE_UNIT_X   , NEGATIVE_UNIT_Y   , vx, vy, DIAGONAL_WEIGHT);
	*(zerothDistributionFunctionComponent + 8) = equilibriumDistributionComponent(density, POSITIVE_UNIT_X   , NEGATIVE_UNIT_Y   , vx, vy, DIAGONAL_WEIGHT);

}

template <typename T>
__device__ T equilibriumDistributionComponent( const T density, const T ex, const T ey, const T velx, const T vely, const T weight)
{
	T eDotProductVel = ex * velx + ey * vely;
	T velocitySquared = velx * velx + vely * vely;

	return weight * density * (1.0f + 3.0f * eDotProductVel + 4.5f * eDotProductVel - 1.5f * velocitySquared); // Taylor expansion of the maxwell distribution for low mach numbers for the ith direction component
}

template <typename T>
void initialise(T *distributionFunctions)
{
	int x = 1;
	int y = 2;

	T xVelocity = xInitialVelocity();
	T yVelocity = yInitialVelocity();

	calculateDistributionFcns(density, xVelocity, yVelocity, x, y);


}
/*
template <typename T>
__device__ T xInitialVelocity(const int i, const int j)
{

}

template <typename T>
__device__ T yInitialVelocity(const int i, const int j)
{

}*/