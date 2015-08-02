/*#include "cuda_runtime.h"

#include "kernels.cuh"

#define CENTRAL_WEIGHT (4.f / 9.f)
#define CROSS_WEIGHT (1.f / 9.f)
#define DIAGONAL_WEIGHT (1.f / 36.f)

#define NEGATIVE_UNIT_X -1.f
#define NEGATIVE_UNIT_Y -1.f

#define POSITIVE_UNIT_X 1.f
#define POSITIVE_UNIT_Y 1.f

#define ZERO_UNIT_VELOCITY 0.f

#define NUM_COMPONENTS 9

#define PI 3.141592654f

#define DELTA 0.05f;
#define KAPPA 20.f

template <typename T>
__device__ void calculateDistributionFcns(const T density, const T *zerothDistributionFunctionComponent, const T vx, const T vy, const int x, const int y)
{

	zerothDistributionFunctionComponent[0] = equilibriumDistributionComponent(density, ZERO_UNIT_VELOCITY, ZERO_UNIT_VELOCITY, vx, vy, CENTRAL_WEIGHT);
	zerothDistributionFunctionComponent[1] = equilibriumDistributionComponent(density, POSITIVE_UNIT_X   , ZERO_UNIT_VELOCITY, vx, vy, CROSS_WEIGHT);
	zerothDistributionFunctionComponent[2] = equilibriumDistributionComponent(density, ZERO_UNIT_VELOCITY, POSITIVE_UNIT_Y   , vx, vy, CROSS_WEIGHT);
	zerothDistributionFunctionComponent[3] = equilibriumDistributionComponent(density, NEGATIVE_UNIT_X   , NEGATIVE_UNIT_Y   , vx, vy, CROSS_WEIGHT);
	zerothDistributionFunctionComponent[4] = equilibriumDistributionComponent(density, ZERO_UNIT_VELOCITY, NEGATIVE_UNIT_Y   , vx, vy, CROSS_WEIGHT);
	zerothDistributionFunctionComponent[5] = equilibriumDistributionComponent(density, POSITIVE_UNIT_X   , POSITIVE_UNIT_Y   , vx, vy, DIAGONAL_WEIGHT);
	zerothDistributionFunctionComponent[6] = equilibriumDistributionComponent(density, NEGATIVE_UNIT_X   , POSITIVE_UNIT_Y   , vx, vy, DIAGONAL_WEIGHT);
	zerothDistributionFunctionComponent[7] = equilibriumDistributionComponent(density, NEGATIVE_UNIT_X   , NEGATIVE_UNIT_Y   , vx, vy, DIAGONAL_WEIGHT);
	zerothDistributionFunctionComponent[8] = equilibriumDistributionComponent(density, POSITIVE_UNIT_X   , NEGATIVE_UNIT_Y   , vx, vy, DIAGONAL_WEIGHT);

}

template <typename T>
__device__ T equilibriumDistributionComponent( const T density, const T ex, const T ey, const T velx, const T vely, const T weight)
{
	T eDotProductVel = ex * velx + ey * vely;
	T velocitySquared = velx * velx + vely * vely;

	return weight * density * (1.0f + 3.0f * eDotProductVel + 4.5f * eDotProductVel - 1.5f * velocitySquared); // Taylor expansion of the maxwell distribution for low mach numbers for the ith direction component
}

template <typename T>
__device__ T getDensity(const T *zerothDistributionFunctionComponent)
{
	T density = 0;
	
#pragma unroll

	for (int i = 0; i < NUM_COMPONENTS; i++)
		density += zerothDistributionFunctionComponent[i];

	return density;
}

template <typename T>
__device__ T getXVelocity(const T *zerothDistributionFunctionComponent)
{
	T xVelocity = 0;

	xVelocity += zerothDistributionFunctionComponent[1];
	xVelocity -= zerothDistributionFunctionComponent[3];
	xVelocity += zerothDistributionFunctionComponent[5];
	xVelocity -= zerothDistributionFunctionComponent[6];
	xVelocity -= zerothDistributionFunctionComponent[7];
	xVelocity += zerothDistributionFunctionComponent[8];

	return xVelocity;
}

template <typename T>
__device__ T getYVelocity(const T *zerothDistributionFunctionComponent)
{
	T yVelocity = 0;

	yVelocity += zerothDistributionFunctionComponent[2];
	yVelocity -= zerothDistributionFunctionComponent[4];
	yVelocity += zerothDistributionFunctionComponent[5];
	yVelocity += zerothDistributionFunctionComponent[6];
	yVelocity -= zerothDistributionFunctionComponent[7];
	yVelocity -= zerothDistributionFunctionComponent[8];

	return yVelocity;
}

//template <typename T>
__global__ void initialise(float initialDensity, float *distributionFunctions)
{
	int x = threadIdx.x;
	int y = blockIdx.x;

	T xVelocity = xInitialVelocity<T>(x,y);
	T yVelocity = yInitialVelocity<T>(x,y);

	calculateDistributionFcns<T>(initialDensity, xVelocity, yVelocity, x, y);
}

template <typename T>
__device__ T xInitialVelocity(const int y)
{
	T normalisedY = y / gridDim.x;

	if (normalisedY <= 0.5)
		return tanh(KAPPA * (normalisedY - 0.25f));
	else
		return tanh(KAPPA * (0.75f - normalisedY));
}

template <typename T>
__device__ T yInitialVelocity(const int x)
{
	return sin(2.f * PI * (x + 0.25f)) * DELTA;
}*/