#ifndef __MINMAX.CU__
#define __MINMAX.CU__

#define LENA_WIDTH 512
#define LENA_HEIGHT 512
#define LENA_SIZE 262144 //512*512

//two different min and max implementations in order to test divergence
#define MIN y ^ ((x ^ y) & -(x < y))
#define MAX x ^ ((x ^ y) & -(x < y))

//#define MIN (x < y) ? x : y
//#define MAX (x < y) ? y : x

/*
	This kernel finds mins and maxes for each block and stores them in global memory side by side.
	Inspired by parallel reduction v7
	Keeps thread_count*2 elements in shared memory for maxes and mins
	Each thread takes multiple elements by traversing diffrent parts of data
	Loop unrolling contributes to the performance
	When tid is less than 32 there is no need to use __syncthreads() because there is one warp left.
	When tid==0 it takes the min and max to the global memory and exits.
*/
template<unsigned int blockSize>
__global__ void findMinMax(float *g_idata, float *res) {

    extern __shared__ float data[];
    // Dynamically taking shared memory
    // Dividing shared memory in the following way
    float *minData = data;
    float *maxData = &data[blockSize];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize * 2) + tid;
    unsigned int gridSize = blockSize * 2 * gridDim.x;

    minData[tid] = INT32_MAX;
    maxData[tid] = INT32_MIN;

    int x, y, min, max;
    //Traversing all the data by jumping as much as gridSize elements
    while (i < LENA_SIZE) {

        x = g_idata[i];
        y = g_idata[i + blockSize];
        min = MIN;
        max = MAX;
        x = minData[tid];
        y = min;
        minData[tid] = MIN;
        x = maxData[tid];
        y = max;
        maxData[tid] = MAX;

        i += gridSize;
    }
    __syncthreads();

    if (blockSize >= 1024) {
        if (tid < 512) {
            x = minData[tid];
            y = minData[tid + 512];
            minData[tid] = MIN;
            x = maxData[tid];
            y = maxData[tid + 512];
            maxData[tid] = MAX;
        }
        __syncthreads();
    }
    if (blockSize >= 512) {
        if (tid < 256) {
            x = minData[tid];
            y = minData[tid + 256];
            minData[tid] = MIN;
            x = maxData[tid];
            y = maxData[tid + 256];
            maxData[tid] = MAX;
        }
        __syncthreads();
    }
    if (blockSize >= 256) {
        if (tid < 128) {
            x = minData[tid];
            y = minData[tid + 128];
            minData[tid] = MIN;
            x = maxData[tid];
            y = maxData[tid + 128];
            maxData[tid] = MAX;
        }
        __syncthreads();
    }
    if (blockSize >= 128) {
        if (tid < 64) {
            x = minData[tid];
            y = minData[tid + 64];
            minData[tid] = MIN;
            x = maxData[tid];
            y = maxData[tid + 64];
            maxData[tid] = MAX;
        }
        __syncthreads();
    }

    if (tid < 32) {
        if (blockSize >= 64) {
            x = minData[tid];
            y = minData[tid + 32];
            minData[tid] = MIN;
            x = maxData[tid];
            y = maxData[tid + 32];
            maxData[tid] = MAX;
        }
        if (blockSize >= 32) {
            x = minData[tid];
            y = minData[tid + 16];
            minData[tid] = MIN;
            x = maxData[tid];
            y = maxData[tid + 16];
            maxData[tid] = MAX;
        }
        if (blockSize >= 16) {
            x = minData[tid];
            y = minData[tid + 8];
            minData[tid] = MIN;
            x = maxData[tid];
            y = maxData[tid + 8];
            maxData[tid] = MAX;
        }
        if (blockSize >= 8) {
            x = minData[tid];
            y = minData[tid + 4];
            minData[tid] = MIN;
            x = maxData[tid];
            y = maxData[tid + 4];
            maxData[tid] = MAX;
        }
        if (blockSize >= 4) {
            x = minData[tid];
            y = minData[tid + 2];
            minData[tid] = MIN;
            x = maxData[tid];
            y = maxData[tid + 2];
            maxData[tid] = MAX;
        }
        if (blockSize >= 2) {
            x = minData[tid];
            y = minData[tid + 1];
            minData[tid] = MIN;
            x = maxData[tid];
            y = maxData[tid + 1];
            maxData[tid] = MAX;
        }
    }
    // store min&max in res array side by side
    if (tid == 0) {
        res[blockIdx.x] = minData[0];
        res[gridDim.x + blockIdx.x] = maxData[0];
    }
}

/*
This traverses all the image and recalculate the values according to the given min and constant values found before
*/
__global__ void calcPixelVal(float *g_idata, float* constant, float* min)
{
    unsigned int i = blockIdx.x * blockDim.x  + threadIdx.x;

    if(i<LENA_SIZE)g_idata[i]=(g_idata[i]-(*min))*(*constant);

}

/*This is called after min&max value is called
	Only one thread executes this
*/
__device__ void calcConstant(float* constant, float *min, float *max) {
    *constant = 255.0f / (*max - *min);
}

/*	This is nearly same with the findMinMax() kernel.
	After gathering data from first kernel only a few mins and maxes are left
	With one block this kernel finds the min&max and calls the calcConstant()
	The idea is same with the findMinMax() kernel but this just loads 2 elements for each thread
*/
template<unsigned int blockSize>
__global__ void finalMinMax(float *g_idata, float *constant, float *min, float *max) {

    extern __shared__ float data[];
    float *minData = data;
    float *maxData = &data[blockSize];

    unsigned int tid = threadIdx.x;

    int x, y;
    //load 2 elemens for each
    x = g_idata[tid];
    y = g_idata[tid + blockSize];
    minData[tid] = MIN;
    x = g_idata[tid + blockSize * 2];
    y = g_idata[tid + blockSize * 2 + blockSize];
    maxData[tid] = MAX;

    __syncthreads();

    if (blockSize >= 1024) {
        if (tid < 512) {
            x = minData[tid];
            y = minData[tid + 512];
            minData[tid] = MIN;
            x = maxData[tid];
            y = maxData[tid + 512];
            maxData[tid] = MAX;
        }
        __syncthreads();
    }
    if (blockSize >= 512) {
        if (tid < 256) {
            x = minData[tid];
            y = minData[tid + 256];
            minData[tid] = MIN;
            x = maxData[tid];
            y = maxData[tid + 256];
            maxData[tid] = MAX;
        }
        __syncthreads();
    }
    if (blockSize >= 256) {
        if (tid < 128) {
            x = minData[tid];
            y = minData[tid + 128];
            minData[tid] = MIN;
            x = maxData[tid];
            y = maxData[tid + 128];
            maxData[tid] = MAX;
        }
        __syncthreads();
    }
    if (blockSize >= 128) {
        if (tid < 64) {
            x = minData[tid];
            y = minData[tid + 64];
            minData[tid] = MIN;
            x = maxData[tid];
            y = maxData[tid + 64];
            maxData[tid] = MAX;
        }
        __syncthreads();
    }

    if (tid < 32) {
        if (blockSize >= 64) {
            x = minData[tid];
            y = minData[tid + 32];
            minData[tid] = MIN;
            x = maxData[tid];
            y = maxData[tid + 32];
            maxData[tid] = MAX;
        }
        if (blockSize >= 32) {
            x = minData[tid];
            y = minData[tid + 16];
            minData[tid] = MIN;
            x = maxData[tid];
            y = maxData[tid + 16];
            maxData[tid] = MAX;
        }
        if (blockSize >= 16) {
            x = minData[tid];
            y = minData[tid + 8];
            minData[tid] = MIN;
            x = maxData[tid];
            y = maxData[tid + 8];
            maxData[tid] = MAX;
        }
        if (blockSize >= 8) {
            x = minData[tid];
            y = minData[tid + 4];
            minData[tid] = MIN;
            x = maxData[tid];
            y = maxData[tid + 4];
            maxData[tid] = MAX;
        }
        if (blockSize >= 4) {
            x = minData[tid];
            y = minData[tid + 2];
            minData[tid] = MIN;
            x = maxData[tid];
            y = maxData[tid + 2];
            maxData[tid] = MAX;
        }
        if (blockSize >= 2) {
            x = minData[tid];
            y = minData[tid + 1];
            minData[tid] = MIN;
            x = maxData[tid];
            y = maxData[tid + 1];
            maxData[tid] = MAX;
        }
    }

    if (tid == 0) {
        *min = minData[0];
        *max = maxData[0];
        calcConstant(constant, min, max);
    }
}

#endif