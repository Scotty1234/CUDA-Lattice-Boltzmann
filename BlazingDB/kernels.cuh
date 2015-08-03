#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

//weights for the lattice components
#define CENTRAL_WEIGHT (4.f / 9.f)
#define CROSS_WEIGHT (1.f / 9.f)
#define DIAGONAL_WEIGHT (1.f / 36.f)


//unit vector lattice directions
#define NEGATIVE_UNIT_X -1.f
#define NEGATIVE_UNIT_Y -1.f

#define POSITIVE_UNIT_X 1.f
#define POSITIVE_UNIT_Y 1.f

#define ZERO_UNIT_VELOCITY 0.f

#define NUM_COMPONENTS 9 // 9 possible directions for the mesoscopic particles to stream across
#define NUM_MACROVARS 3 // 3 macroscopic variables - density, x velocity, y velocity

#define PI 3.141592654f

//Constants for use in the initial velocity field
#define ALPHA 0.07
#define DELTA 0.05f;
#define KAPPA 80.f



template <typename T>
__device__  void calculateDistributionFunctions(const T density, T *zerothDistributionFunctionComponent, const T vx, const T vy, const int x, const int y);

template <typename T>
__device__   T calculateEquilibriumDistributionComponent(const T density, const T ex, const T ey, const T velx, const T vely, const T weight);

template <typename T>
__global__ void computeVorticity(const T* macroscopicVariables, T* vorticity);

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
	zerothEquilibriumDistributionFunctionComponent[3] = calculateEquilibriumDistributionComponent<T>(density, NEGATIVE_UNIT_X, ZERO_UNIT_VELOCITY, vx, vy, CROSS_WEIGHT);
	zerothEquilibriumDistributionFunctionComponent[4] = calculateEquilibriumDistributionComponent<T>(density, ZERO_UNIT_VELOCITY, NEGATIVE_UNIT_Y, vx, vy, CROSS_WEIGHT);
	zerothEquilibriumDistributionFunctionComponent[5] = calculateEquilibriumDistributionComponent<T>(density, POSITIVE_UNIT_X, POSITIVE_UNIT_Y, vx, vy, DIAGONAL_WEIGHT);
	zerothEquilibriumDistributionFunctionComponent[6] = calculateEquilibriumDistributionComponent<T>(density, NEGATIVE_UNIT_X, POSITIVE_UNIT_Y, vx, vy, DIAGONAL_WEIGHT);
	zerothEquilibriumDistributionFunctionComponent[7] = calculateEquilibriumDistributionComponent<T>(density, NEGATIVE_UNIT_X, NEGATIVE_UNIT_Y, vx, vy, DIAGONAL_WEIGHT);
	zerothEquilibriumDistributionFunctionComponent[8] = calculateEquilibriumDistributionComponent<T>(density, POSITIVE_UNIT_X, NEGATIVE_UNIT_Y, vx, vy, DIAGONAL_WEIGHT);

}

template <typename T>
__device__ inline T collideDistributionComponent(const T distributionComponent, const T equilibriumDistributionComponent, const T relaxation)
{
	//Lattice boltzmann collision equation for the mesoscopic particles to transfer momentum
	T oneOverRelaxation = 1.f / relaxation;
	return (1.f - oneOverRelaxation) * distributionComponent + oneOverRelaxation * equilibriumDistributionComponent;
}

template <typename T>
__device__   T calculateEquilibriumDistributionComponent(const T density, const T ex, const T ey, const T velx, const T vely, const T weight)
{
	T eDotProductVel = ex * velx + ey * vely;
	T velocitySquared = velx * velx + vely * vely;

	return weight * density * (1.0f + 3.0f * eDotProductVel + 4.5f * eDotProductVel * eDotProductVel - 1.5f * velocitySquared); // Taylor expansion of the maxwell distribution for low mach numbers for the ith direction component
}

template <typename T>
__device__  T getDensity(const T *zerothDistributionFunctionComponent)
{
	//sum up components for first moment (aka density) of maxwell distribution
	T density = 0;

#pragma unroll //unroll for optimisation

	for (int i = 0; i < NUM_COMPONENTS; i++)
		density += zerothDistributionFunctionComponent[i];

	return density;
}

template <typename T>
__device__  T getXVelocity(const T *zerothDistributionFunctionComponent)
{
	//sum up components with non-zero x direction velocity component
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
	//sum up components with non-zero y direction velocity component
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

	//get initial velocity field
	T xVelocity = xInitialVelocity<T>(y);
	T yVelocity = yInitialVelocity<T>(x);

	//set distribution functions equal to the equilibrium distribution
	calculateEquilibriumDistributionFunctions<T>(initialDensity, &distributionFunctions[NUM_COMPONENTS * idx], xVelocity, yVelocity, x, y);

}


template <typename T>
__global__ void simulateIteration(T *distributionFunctions, T* macroscopicVariables, T *relaxation, T *tempDistributionFunctions, const bool store, const int time)
{
	T distributionFunctionComponents[NUM_COMPONENTS]; //put distribution functions into registers
	T equilibriumDistributionFunctionComponents[NUM_COMPONENTS];

	//locate self within the grid to map to 2d space
	int x = threadIdx.x;
	int y = blockIdx.x;

	int nx = blockDim.x;
	int ny = gridDim.x;

	// Streaming directions (periodic)
	int xMinus = (x == 0) ? nx - 1 : (x - 1);
	int xPlus = (x == nx - 1) ? 0 : (x + 1);

	int yMinus = (y == 0) ? ny - 1 : (y - 1);
	int yPlus = (y == ny - 1) ? 0 : (y + 1);

	//stream in components 
	distributionFunctionComponents[0] = distributionFunctions[NUM_COMPONENTS * (x * ny + y)];
	distributionFunctionComponents[1] = distributionFunctions[NUM_COMPONENTS * (x * ny + y) + 1];
	distributionFunctionComponents[2] = distributionFunctions[NUM_COMPONENTS * (x * ny + y) + 2];
	distributionFunctionComponents[3] = distributionFunctions[NUM_COMPONENTS * (x * ny + y) + 3];
	distributionFunctionComponents[4] = distributionFunctions[NUM_COMPONENTS * (x * ny + y) + 4];
	distributionFunctionComponents[5] = distributionFunctions[NUM_COMPONENTS * (x * ny + y) + 5];
	distributionFunctionComponents[6] = distributionFunctions[NUM_COMPONENTS * (x * ny + y) + 6];
	distributionFunctionComponents[7] = distributionFunctions[NUM_COMPONENTS * (x * ny + y) + 7];
	distributionFunctionComponents[8] = distributionFunctions[NUM_COMPONENTS * (x * ny + y) + 8];

	T density = getDensity(distributionFunctionComponents); // first moment of maxwell distribution

	T xVelocity = getXVelocity(distributionFunctionComponents); //second moment of maxwell distribution for the velocities
	xVelocity /= density;

	T yVelocity = getYVelocity(distributionFunctionComponents);
	yVelocity /= density;

	if (store) // only write to global memory if needed, i.e. for a vorticity calculation and download to host
	{
		macroscopicVariables[NUM_MACROVARS * (x * ny + y)] = density;
		macroscopicVariables[NUM_MACROVARS * (x * ny + y) + 1] = xVelocity;
		macroscopicVariables[NUM_MACROVARS * (x * ny + y) + 2] = yVelocity;
	}

	calculateEquilibriumDistributionFunctions(density, equilibriumDistributionFunctionComponents, xVelocity, yVelocity, x, y);

	float relaxation1 = 1.f;

	//collide and stream out 
	tempDistributionFunctions[NUM_COMPONENTS * (x * ny + y)]			   = collideDistributionComponent<T>(distributionFunctionComponents[0], equilibriumDistributionFunctionComponents[0], relaxation1);
	tempDistributionFunctions[NUM_COMPONENTS * (xPlus * ny + y) + 1]       = collideDistributionComponent<T>(distributionFunctionComponents[1], equilibriumDistributionFunctionComponents[1], relaxation1);
	tempDistributionFunctions[NUM_COMPONENTS * (x * ny + yPlus) + 2]       = collideDistributionComponent<T>(distributionFunctionComponents[2], equilibriumDistributionFunctionComponents[2], relaxation1);
	tempDistributionFunctions[NUM_COMPONENTS * (xMinus * ny + y) + 3]      = collideDistributionComponent<T>(distributionFunctionComponents[3], equilibriumDistributionFunctionComponents[3], relaxation1);
	tempDistributionFunctions[NUM_COMPONENTS * (x * ny + yMinus) + 4]      = collideDistributionComponent<T>(distributionFunctionComponents[4], equilibriumDistributionFunctionComponents[4], relaxation1);
	tempDistributionFunctions[NUM_COMPONENTS * (xPlus * ny + yPlus) + 5]   = collideDistributionComponent<T>(distributionFunctionComponents[5], equilibriumDistributionFunctionComponents[5], relaxation1);
	tempDistributionFunctions[NUM_COMPONENTS * (xMinus * ny + yPlus) + 6]  = collideDistributionComponent<T>(distributionFunctionComponents[6], equilibriumDistributionFunctionComponents[6], relaxation1);
	tempDistributionFunctions[NUM_COMPONENTS * (xMinus * ny + yMinus) + 7] = collideDistributionComponent<T>(distributionFunctionComponents[7], equilibriumDistributionFunctionComponents[7], relaxation1);
	tempDistributionFunctions[NUM_COMPONENTS * (xPlus * ny + yMinus) + 8]  = collideDistributionComponent<T>(distributionFunctionComponents[8], equilibriumDistributionFunctionComponents[8], relaxation1);
	
}

template <typename T>
__global__ void computeVorticity(const T* macroscopicVariables, T* vorticity)
{
	int x = threadIdx.x;
	int y = blockIdx.x;

	int nx = blockDim.x;
	int ny = gridDim.x;

	//Periodic boundary conditions again
	int xMinus = (x == 0) ? nx - 1 : (x - 1);
	int xPlus = (x == nx - 1) ? 0 : (x + 1);

	int yMinus = (y == 0) ? ny - 1 : (y - 1);
	int yPlus = (y == ny - 1) ? 0 : (y + 1);

	//two dimensional vorticity (i.e. the curl on the velocity field) is simply d_xVelocity / dx - d_xVelocity/dy in the z vector direction. However for LB, dx = dy = 1

	T d_xVelocity = macroscopicVariables[NUM_MACROVARS * (x * ny + yPlus) + 1] - macroscopicVariables[NUM_MACROVARS * (x * ny + yMinus) + 1]; 


	T d_yVelocity = macroscopicVariables[NUM_MACROVARS * (xPlus * ny + y) + 2] - macroscopicVariables[NUM_MACROVARS * (xMinus * ny + y) + 2];
	
	vorticity[x * ny + y] = d_yVelocity - d_xVelocity;

}

template <typename T>
__device__  T xInitialVelocity(const int y)
{

	T normalisedY = (T)y / (T)gridDim.x;//y must be between 0 and 1. Cast to avoid zero. 

	if (normalisedY <= 0.5f)
		return tanh(KAPPA * (normalisedY - 0.25f)) * ALPHA;
	else
		return tanh(KAPPA * (0.75f - normalisedY)) * ALPHA;
}

template <typename T>
__device__  T yInitialVelocity(const int x)
{

	T normalisedX = (T)x / (T)blockDim.x; // x needs to be between 0 and 1. Cast to avoid zero. 

	return  sin(2.f * PI * (normalisedX + 0.25f)) * ALPHA * DELTA ;
}

