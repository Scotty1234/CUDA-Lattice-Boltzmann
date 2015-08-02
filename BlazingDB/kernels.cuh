#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define CENTRAL_WEIGHT (4.f / 9.f)
#define CROSS_WEIGHT (1.f / 9.f)
#define DIAGONAL_WEIGHT (1.f / 36.f)

#define NEGATIVE_UNIT_X -1.f
#define NEGATIVE_UNIT_Y -1.f

#define POSITIVE_UNIT_X 1.f
#define POSITIVE_UNIT_Y 1.f

#define ZERO_UNIT_VELOCITY 0.f

#define NUM_COMPONENTS 9
#define NUM_MACROVARS 3

#define PI 3.141592654f

#define DELTA 0.05f;
#define KAPPA 20.f

#define Idf3D(i,x,y,z,nx,ny,nz) ((i)*(nx)*(ny)*(nz) + (x) + (y)*(nx) + (z)*(nx)*(ny))
#define df3D(i,x,y,z,nx,ny,nz)  (df[    Idf3D(i,x,y,z,nx,ny,nz)])

template <typename T>
__device__  void calculateDistributionFunctions(const T density, T *zerothDistributionFunctionComponent, const T vx, const T vy, const int x, const int y);

template <typename T>
__device__   T calculateEquilibriumDistributionComponent(const T density, const T ex, const T ey, const T velx, const T vely, const T weight);

template <typename T>
__global__ void initialise(T initialDensity, T *distributionFunctions);

template <typename T>
__device__  T getDensity(const T *zerothDistributionFunctionComponent);

template <typename T>
__global__ void simulateIteration(T *distributionFunctions, T* macroscopicVariables, T* relaxation, T *tempDistributionFunctions, const bool store, const int time);

template <typename T>
__device__  T xInitialVelocity(const int y);

template <typename T>
__device__ T  yInitialVelocity(const int x);


//=================

template <typename T>
__device__  void calculateEquilibriumDistributionFunctions(const T density, T *zerothEquilibriumDistributionFunctionComponent, const T vx, const T vy, const int x, const int y)
{

	zerothEquilibriumDistributionFunctionComponent[0] = calculateEquilibriumDistributionComponent<T>(density, ZERO_UNIT_VELOCITY, ZERO_UNIT_VELOCITY, vx, vy, CENTRAL_WEIGHT);
	zerothEquilibriumDistributionFunctionComponent[1] = calculateEquilibriumDistributionComponent<T>(density, POSITIVE_UNIT_X, ZERO_UNIT_VELOCITY, vx, vy, CROSS_WEIGHT);
	zerothEquilibriumDistributionFunctionComponent[2] = calculateEquilibriumDistributionComponent<T>(density, ZERO_UNIT_VELOCITY, POSITIVE_UNIT_Y, vx, vy, CROSS_WEIGHT);
	zerothEquilibriumDistributionFunctionComponent[3] = calculateEquilibriumDistributionComponent<T>(density, NEGATIVE_UNIT_X, NEGATIVE_UNIT_Y, vx, vy, CROSS_WEIGHT);
	zerothEquilibriumDistributionFunctionComponent[4] = calculateEquilibriumDistributionComponent<T>(density, ZERO_UNIT_VELOCITY, NEGATIVE_UNIT_Y, vx, vy, CROSS_WEIGHT);
	zerothEquilibriumDistributionFunctionComponent[5] = calculateEquilibriumDistributionComponent<T>(density, POSITIVE_UNIT_X, POSITIVE_UNIT_Y, vx, vy, DIAGONAL_WEIGHT);
	zerothEquilibriumDistributionFunctionComponent[6] = calculateEquilibriumDistributionComponent<T>(density, NEGATIVE_UNIT_X, POSITIVE_UNIT_Y, vx, vy, DIAGONAL_WEIGHT);
	zerothEquilibriumDistributionFunctionComponent[7] = calculateEquilibriumDistributionComponent<T>(density, NEGATIVE_UNIT_X, NEGATIVE_UNIT_Y, vx, vy, DIAGONAL_WEIGHT);
	zerothEquilibriumDistributionFunctionComponent[8] = calculateEquilibriumDistributionComponent<T>(density, POSITIVE_UNIT_X, NEGATIVE_UNIT_Y, vx, vy, DIAGONAL_WEIGHT);

}

template <typename T>
__device__ inline T collideDistributionComponent(const T distributionComponent, const T equilibriumDistributionComponent, const T relaxation)
{
	T oneOverRelaxation = 1.f / relaxation;
	return (1.f - oneOverRelaxation) * distributionComponent + oneOverRelaxation * equilibriumDistributionComponent;
}

template <typename T>
__device__   T calculateEquilibriumDistributionComponent(const T density, const T ex, const T ey, const T velx, const T vely, const T weight)
{
	T eDotProductVel = ex * velx + ey * vely;
	T velocitySquared = velx * velx + vely * vely;

	return weight * density * (1.0f + 3.0f * eDotProductVel + 4.5f * eDotProductVel - 1.5f * velocitySquared); // Taylor expansion of the maxwell distribution for low mach numbers for the ith direction component
}

template <typename T>
__device__  T getDensity(const T *zerothDistributionFunctionComponent)
{
	T density = 0;

#pragma unroll

	for (int i = 0; i < NUM_COMPONENTS; i++)
		density += zerothDistributionFunctionComponent[i];

	return density;
}

template <typename T>
__device__  T getXVelocity(const T *zerothDistributionFunctionComponent)
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
__device__  T getYVelocity(const T *zerothDistributionFunctionComponent)
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

template <typename T>
__global__ void initialise(T initialDensity, T *distributionFunctions)
{
	int x = threadIdx.x;
	int y = blockIdx.x;

	int idx = x * gridDim.x + y;

	T xVelocity = xInitialVelocity<T>(y);
	T yVelocity = yInitialVelocity<T>(x);

	calculateEquilibriumDistributionFunctions<T>(initialDensity, &distributionFunctions[NUM_COMPONENTS * idx], xVelocity, yVelocity, x, y);

	/*for (int i = 0; i < 9; i++)
		printf("%i init  %G\n", i, distributionFunctions[NUM_COMPONENTS * idx + i]);*/

	/*macroscopicVariables[NUM_MACROVARS * idx] = getDensity(&distributionFunctions[idx]);
	macroscopicVariables[NUM_MACROVARS * idx + 1] = getXVelocity(&distributionFunctions[idx]);
	macroscopicVariables[NUM_MACROVARS * idx + 2] = getYVelocity(&distributionFunctions[idx]);*/


}


template <typename T>
__global__ void simulateIteration(T *distributionFunctions, T* macroscopicVariables, T *relaxation, T *tempDistributionFunctions, const bool store, const int time)
{
	T distributionFunctionComponents[NUM_COMPONENTS]; //put distribution functions into registers
	T equilibriumDistributionFunctionComponents[NUM_COMPONENTS];

	int x = threadIdx.x;
	int y = blockIdx.x;

	//int nx = gridDim.x;
	//int ny = gridDim.x;

	int nx = 10;
	int ny = 10;

	//int idx = x * nx+ y;

	// Streaming directions (periodic)
	int xMinus = (x == 0) ? nx - 1 : (x - 1);
	int xPlus = (x == nx - 1) ? 0 : (x + 1);

	int yMinus = (y == 0) ? ny - 1 : (y - 1);
	int yPlus = (y == ny - 1) ? 0 : (y + 1);

	//stream in components 
	distributionFunctionComponents[0] = distributionFunctions[NUM_COMPONENTS * (x * nx + y)];
	distributionFunctionComponents[1] = distributionFunctions[NUM_COMPONENTS * (x * nx + y) + 1];
	distributionFunctionComponents[2] = distributionFunctions[NUM_COMPONENTS * (x * nx + y) + 2];
	distributionFunctionComponents[3] = distributionFunctions[NUM_COMPONENTS * (x * nx + y) + 3];
	distributionFunctionComponents[4] = distributionFunctions[NUM_COMPONENTS * (x * nx + y) + 4];
	distributionFunctionComponents[5] = distributionFunctions[NUM_COMPONENTS * (x * nx + y) + 5];
	distributionFunctionComponents[6] = distributionFunctions[NUM_COMPONENTS * (x * nx + y) + 6];
	distributionFunctionComponents[7] = distributionFunctions[NUM_COMPONENTS * (x * nx + y) + 7];
	distributionFunctionComponents[8] = distributionFunctions[NUM_COMPONENTS * (x * nx + y) + 8];

	T density = getDensity(distributionFunctionComponents);

	if (x == 0 && y == 0)
	//{
		//printf("t: %i %G\n", time, density);
		//printf("nx: %i, ny: %i\n", nx, ny);
		for (int i = 0; i < 9; i++)
			printf("%i %G\n", i, distributionFunctionComponents[i]);
	//}

	T xVelocity = getXVelocity(distributionFunctionComponents);
	T yVelocity = getYVelocity(distributionFunctionComponents);

	if (store)
	{
		macroscopicVariables[NUM_MACROVARS * (x * nx + y)] = density;
		macroscopicVariables[NUM_MACROVARS * (x * nx + y) + 1] = xVelocity;
		macroscopicVariables[NUM_MACROVARS * (x * nx + y) + 2] = yVelocity;

		//printf("%G %G %G\n", macroscopicVariables[NUM_MACROVARS * (x * nx + y)], macroscopicVariables[NUM_MACROVARS * (x * nx + y) + 1], macroscopicVariables[NUM_MACROVARS * (x * nx + y) + 2]);

	}

	calculateEquilibriumDistributionFunctions(density, equilibriumDistributionFunctionComponents, xVelocity, yVelocity, x, y);
														

	//collide and stream out 
	tempDistributionFunctions[NUM_COMPONENTS * (x * nx + y)] = collideDistributionComponent<T>(distributionFunctionComponents[0], equilibriumDistributionFunctionComponents[0], 1.0f);
	tempDistributionFunctions[NUM_COMPONENTS * (xPlus * nx + y) + 1]       = collideDistributionComponent<T>(distributionFunctionComponents[1], equilibriumDistributionFunctionComponents[1], 1.0f);
	tempDistributionFunctions[NUM_COMPONENTS * (x * nx + yPlus) + 2]       = collideDistributionComponent<T>(distributionFunctionComponents[2], equilibriumDistributionFunctionComponents[2], 1.0f);
	tempDistributionFunctions[NUM_COMPONENTS * (xMinus * nx + y) + 3]      = collideDistributionComponent<T>(distributionFunctionComponents[3], equilibriumDistributionFunctionComponents[3], 1.0f);
	tempDistributionFunctions[NUM_COMPONENTS * (x * nx + yMinus) + 4]      = collideDistributionComponent<T>(distributionFunctionComponents[4], equilibriumDistributionFunctionComponents[4], 1.0f);
	tempDistributionFunctions[NUM_COMPONENTS * (xPlus * nx + yPlus) + 5]   = collideDistributionComponent<T>(distributionFunctionComponents[5], equilibriumDistributionFunctionComponents[5], 1.0f);
	tempDistributionFunctions[NUM_COMPONENTS * (xMinus * nx + yPlus) + 6]  = collideDistributionComponent<T>(distributionFunctionComponents[6], equilibriumDistributionFunctionComponents[6], 1.0f);
	tempDistributionFunctions[NUM_COMPONENTS * (xMinus * nx + yMinus) + 7] = collideDistributionComponent<T>(distributionFunctionComponents[7], equilibriumDistributionFunctionComponents[7], 1.0f);
	tempDistributionFunctions[NUM_COMPONENTS * (xPlus * nx + yMinus) + 8]  = collideDistributionComponent<T>(distributionFunctionComponents[8], equilibriumDistributionFunctionComponents[8], 1.0f);

	
}

/*template <typename T>
__device__ void streamIn(T * distributionFunctions, const int x, const int y)
{

}

__device__ void streamOut()
{

}*/

template <typename T>
__device__  T xInitialVelocity(const int y)
{
	return 0.f;

	/*T normalisedY = y / gridDim.x;

	if (normalisedY <= 0.5)
		return tanh(KAPPA * (normalisedY - 0.25f));
	else
		return tanh(KAPPA * (0.75f - normalisedY));*/
}

template <typename T>
__device__  T yInitialVelocity(const int x)
{
	return 0.f;// sin(2.f * PI * (x + 0.25f)) * DELTA;
}

