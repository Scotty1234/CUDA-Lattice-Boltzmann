
#include "LatticeBoltzmannExample.h"

int main(int argc, char **argv)
{
	LatticeBoltzmannExample latticeBoltzmannExample(argc, argv);
	latticeBoltzmannExample.run();
	
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    /*cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }*/

    return EXIT_SUCCESS;
}
