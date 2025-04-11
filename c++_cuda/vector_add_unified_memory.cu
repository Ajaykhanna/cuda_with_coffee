#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>

// Vector addition with CUDA Kernel
__global__ void vecAddUM(int *a, int *b, int *c, int n)
{
	// Get the Global thread ID (tid)
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

	// Sanity Check: Guarding Vector Bounday
	if (tid < n)
	{
		// Each Thread adds a single element
		// Parallelizing the "for" loop with threads
		c[tid] = a[tid] + b[tid];
	}
}

// Using Random Number generator to generate matrix elements of size n
// and integers between 0 to 99
void rand_matrix(int *a, int n)
{
	for (int i = 0; i < n; i++)
	{
		a[i] = rand() % 100;
	}
}

// Verify vector addition results
void verify_results(int *a, int *b, int *c, int n)
{
	for (int i = 0; i < n; i++)
	{
		assert(c[i] == a[i] + b[i]);
	}
}

int main()
{
	// Get GPU Device id
	int id = cudaGetDevice(&id);
	// Vector size of 2^16 (65536 elements)
	int n = 1 << 16;
	// Allocation size for all vectors
	size_t bytes = sizeof(int) * n;

	/* Host Vector Pointers
	//int *h_a, *h_b, *h_c;
	// Device vector pointers
	int *d_a, *d_b, *d_c;


	//Allocate host memory
	h_a = (int*)malloc(bytes);
	h_b = (int*)malloc(bytes);
	h_c = (int*)malloc(bytes);

	// Allocate device memory
	cudaMalloc(&d_a, bytes);
	cudaMalloc(&d_b, bytes);
	cudaMalloc(&d_c, bytes);
	*/
	// Stiping out the host and device pointers

	// <---- Declare unified memory pointers ---->
	int *a, *b, *c;

	// Allocation memory for these new pointers; however
	// <--- using cudaMallocManaged  <-- Instead of cudaMemcpy --->
	// This takes my headace away as I'm a chemist and I don't have
	// to manage memory between cpu & gpu
	cudaMallocManaged(&a, bytes);
	cudaMallocManaged(&b, bytes);
	cudaMallocManaged(&c, bytes);

	// Generate random matrix elements for a and b matrices
	rand_matrix(a, n);
	rand_matrix(b, n);

	/* Copy data from Host to Device
	cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
	*/

	// Threadblock size
	int NUM_THREADS = 256;

	// Grid size
	int NUM_BLOCKS = (int)ceil(n / NUM_THREADS);

	// Lets start loading loading the array onto the GPU
	// in background using "cudaMemPrefetchAsync(array, sizeof, gpu_id)"
	cudaMemPrefetchAsync(a, bytes, id);
	cudaMemPrefetchAsync(b, bytes, id);
	// Launch GPU Kernel/Function on default stream
	vecAddUM<<<NUM_BLOCKS, NUM_THREADS>>>(a, b, c, n);

	/* <--------- Adding cudaDeviceSynchronize() -------------->
	This makes sure all threads done computing before results can be verified
	If we try to access data before this, a raise error will rise and cpu and gpu will
	will fight over who owns the data */
	cudaDeviceSynchronize();

	// Copy sum array c from device to host
	// cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

	// Collecting elements of array c from the device
	cudaMemPrefetchAsync(c, bytes, cudaCpuDeviceId);

	// Check Results for errors
	verify_results(a, b, c, n);

	for (int i = 0; i < int(10); i++)
	{
		printf("The sum of A vector element %d + and B vector element %d is = %d \n", a[i], b[i], c[i]);
	}
	printf("TASK DONE SUCCESSFULLY\n");

	return 0;
}
