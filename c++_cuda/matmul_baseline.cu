#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>

// Vector addition with CUDA Kernel
__global__ void matMul(int* a, int* b, int* c, int n) {
	// Compute each thread's row
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	// Compute each thread's cols
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	int temp_sum = 0;
	// Sanity Check: Guarding Vector Bounday

	if ((row < n ) && (col < n)){
		// Iterate over rows (l --> r) and cols (u --> d) 
		for (int k = 0; k < n; k++){
			// Store results for a singe element
			temp_sum += a[row * n + k] * b[k * n + col];
		}
		// Assign result
		c[row * n + col] = temp_sum;
	}
}

// Using Random Number generator to generate matrix elements of size n
// and integers between 0 to 99
void rand_matrix(int* a, int n){
	for (int i = 0; i < n; i++){
		for (int j = 0; j < n; j++) {
		a[i * n + j] = rand() % 100;
		}
	}
}

// Verify vector addition results
void verify_results(int* a, int* b, int* c, int n){
	int *verify_c;
	verify_c = (int*)malloc(n * n * sizeof(int));
	int temp_sum;
	for (int i = 0; i < n; i++){
		for (int j = 0; j < n; j++){
			temp_sum = 0; 
			for (int k = 0; k < n; k++){
				temp_sum += a[i * n + k] * b[k * n + j];
			}
			verify_c[i * n + j] = temp_sum;
		}
	}
	for (int i = 0; i < n; i++){
		for (int j = 0; j < n; j++){
			assert(c[i * n + j] == verify_c[i * n + j]);
		}
	}

}

int main(){
	// Vector size of 2^10 (1024 X 1024 elements)
	int n = 1 << 10;
	size_t bytes = n * n * sizeof(int);
	
	// Host Vector Pointers
	int *h_a, *h_b, *h_c;
	
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
	
	// Generate random matrix elements for a and b matrices
	rand_matrix(h_a, n);
	rand_matrix(h_b, n);

	// Copy data from Host to Device
	cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

	// Threads per block
	int BLOCK_SIZE = 16;
	
	// Blocks in each dimension (No padding)
	int GRID_SIZE = (int) ceil(n / BLOCK_SIZE);

	// Use dim3 objects
	dim3 grid(GRID_SIZE, GRID_SIZE);
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

	// Launch GPU Kernel/Function on default stream w/o sharedMem
	matMul<<<grid, threads>>>(d_a, d_b, d_c, n);

	// Copy sum array c from device to host
	cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
	
	// Check Results for errors
	verify_results(h_a, h_b, h_c, n);

//	for (int i = 0; i < int(10); i++){
//		for (int j = 0; j < int(10); j++){
//			printf(h_a[i][j]);
//		}
//		printf("The sum of A vector element %d + and B vector element %d is = %d \n", h_a[i], h_b[i], h_c[i]);
//	}
	printf("TASK DONE SUCCESSFULLY\n");

	// Free host memory
	free(h_a);
	free(h_b);
	free(h_c);
	
	// Free memory on device
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	
	return 0;


}
