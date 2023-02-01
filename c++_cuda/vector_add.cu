#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>

// Vector addition with CUDA Kernel
__global__ void vecAdd(int* a, int* b, int* c, int n) {
	// Get the Global thread ID (tid)
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

	// Sanity Check: Guarding Vector Bounday
	if (tid < n) {
		// Each Thread adds a single element
		// Parallelizing the "for" loop with threads
		c[tid] = a[tid] + b[tid];
	}

}

// Using Random Number generator to generate matrix elements of size n
// and integers between 0 to 99
void rand_matrix(int* a, int n){
	for (int i = 0; i < n; i++){
		a[i] = rand() % 100;
	}
}

// Verify vector addition results
void verify_results(int* a, int* b, int* c, int n){
	for (int i = 0; i < n; i++){
		assert(c[i] == a[i] + b[i]);
	}

}

int main(){
	// Vector size of 2^16 (65536 elements)
	int n = 1 << 16;
	// Host Vector Pointers
	int *h_a, *h_b, *h_c;
	// Device vector pointers
	int *d_a, *d_b, *d_c;

	// Allocation size for all vectors
	size_t bytes = sizeof(int) * n;

	//Allocate host memory
	h_a = (int*)malloc(bytes);
	h_b = (int*)malloc(bytes);
	h_c = (int*)malloc(bytes);

	// Allocate device memory
	cudaMalloc(&d_a, bytes);
	cudaMalloc(&d_b, bytes);
	cudaMalloc(&d_c, bytes);
	
	// Generate random matrix elements for a and b matrices
	rand_matrix(h_a, n);
	rand_matrix(h_b, n);

	// Copy data from Host to Device
	cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

	// Threadblock size
	int NUM_THREADS = 256;
	
	// Grid size
	int NUM_BLOCKS = (int) ceil(n / NUM_THREADS);

	// Launch GPU Kernel/Function on default stream w/o sharedMem
	vecAdd <<<NUM_BLOCKS, NUM_THREADS>>>(d_a, d_b, d_c, n);

	// Copy sum array c from device to host
	cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
	
	// Check Results for errors
	verify_results(h_a, h_b, h_c, n);

	for (int i = 0; i < int(10); i++){
		printf("The sum of A vector element %d + and B vector element %d is = %d \n", h_a[i], h_b[i], h_c[i]);
	}
	printf("TASK DONE SUCCESSFULLY\n");
	
	return 0;

}
