#include <iostream>
#include <string.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math.h"
#include "time.h"
#include <fstream>
#include <iomanip>
#include <sstream>
#include <cstdlib>
#include <cstdio>
#include <float.h>
#include <curand_mtgp32_kernel.h>

//Defined in order to test different block and grid dimensions
// THREAD_COUNT and GRID_DIM must be power of 2
#define THREAD_COUNT 1024
#define GRID_DIM 32

using namespace std;

float* LoadPGM(char * sFileName, int & nWidth, int & nHeight, int & nMaxGray);
void WritePGM(char * sFileName, float * pDst_Host, int nWidth, int nHeight, int nMaxGray);


/*
 *  Three kernels are called in the following order:
 *      findMinMax
 *      finalMinMax
 *      calcPixelVal
 *
 *      findMinMax finds mins&maxes for each block
 *      finalMinMax finds 1 min and 1 max from previous output
 *      calcPixelVal calculates new pixels value according to min&max
 */
int main(void)
{
    float *Lena_Original_h, *Lena_d, *Res_MinsMaxs;  // Pointer to host & device arrays

    float *min, *max, *constant; //For keeping min, max and a constant value which holds (255/(max-min))

    size_t vector_size;			//total size of lena

    vector_size = LENA_SIZE * sizeof(float);

    int nWidth, nHeight, nMaxGray;

    Lena_Original_h= LoadPGM("lena_before.pgm", nWidth, nHeight, nMaxGray);

    //allocate space for variables on device
    cudaMalloc((void **)&min, sizeof(float));
    cudaMalloc((void **)&max, sizeof(float));
    cudaMalloc((void **)&constant, sizeof(float));

    cudaMalloc((void **)&Lena_d, vector_size);     // Allocate array on device for Lena

    cudaMemcpy(Lena_d, Lena_Original_h, vector_size, cudaMemcpyHostToDevice);      // copy values to device

    //Block dimension is directly from THREAD_COUNT
    dim3 Block_dim(THREAD_COUNT, 1, 1);

    //Grid dim will start from GRID_DIM
    //Not used dependent values in order to test different dimensions
    dim3 Grid_dim(GRID_DIM, 1, 1);

    //For shared memory size. x2 for keeping minimums and maximums in one array
    int smemSize = sizeof(float) * THREAD_COUNT * 2;

    //For keeping minimums and maximums found in each block
    // x2 for for keeping minimums and maximums in one array
    cudaMalloc((void **)&Res_MinsMaxs, sizeof(float)*Grid_dim.x * 2);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);

    //kernel calls for findings mins&maxes for each block
    switch (THREAD_COUNT)
    {
        case 1024:
            findMinMax<1024> << < Grid_dim, Block_dim, smemSize >> > (Lena_d, Res_MinsMaxs); break;
        case 512:
            findMinMax<512 > << < Grid_dim, Block_dim, smemSize >> > (Lena_d, Res_MinsMaxs); break;
        case 256:
            findMinMax<256 > << < Grid_dim, Block_dim, smemSize >> > (Lena_d, Res_MinsMaxs); break;
        case 128:
            findMinMax<128 > << < Grid_dim, Block_dim, smemSize >> > (Lena_d, Res_MinsMaxs); break;
        case 64:
            findMinMax<64  > << < Grid_dim, Block_dim, smemSize >> > (Lena_d, Res_MinsMaxs); break;
        case 32:
            findMinMax<32  > << < Grid_dim, Block_dim, smemSize >> > (Lena_d, Res_MinsMaxs); break;
        case 16:
            findMinMax<16  > << < Grid_dim, Block_dim, smemSize >> > (Lena_d, Res_MinsMaxs); break;
        case 8:
            findMinMax<8   > << < Grid_dim, Block_dim, smemSize >> > (Lena_d, Res_MinsMaxs); break;
        case 4:
            findMinMax<4   > << < Grid_dim, Block_dim, smemSize >> > (Lena_d, Res_MinsMaxs); break;
        case 2:
            findMinMax<2   > << < Grid_dim, Block_dim, smemSize >> > (Lena_d, Res_MinsMaxs); break;
        case 1:
            findMinMax<1   > <<< Grid_dim, Block_dim, smemSize >> > (Lena_d, Res_MinsMaxs); break;
    }

    //From the kernels above, Grid_dim*2 min and max values will be produced.
    //each block produces a min&max value
    //while taking values from global memory, each thread takes multiple elements
    //so shared memory size for all max&min values => sizeof(float)*(Grid_dim*2)/2
    smemSize = sizeof(float) * Grid_dim.x;
    //because each thread takes two elements, block_dim will be half of Grid_dim
    Block_dim = dim3((Grid_dim.x) / 2, 1, 1);
    //only 1 block in order two get just 1 min&max
    //Also previous grid_dim will not be more than 1024, so there will not be too many elements
    Grid_dim = dim3(1, 1, 1);

    //kernel calls for finding final min&max values
    switch (Block_dim.x)
    {
        case 1024:
            finalMinMax<1024> << < Grid_dim, Block_dim, smemSize >> > (Res_MinsMaxs, constant, min, max); break;
        case 512:
            finalMinMax<512 > << < Grid_dim, Block_dim, smemSize >> > (Res_MinsMaxs, constant, min, max); break;
        case 256:
            finalMinMax<256 > << < Grid_dim, Block_dim, smemSize >> > (Res_MinsMaxs, constant, min, max); break;
        case 128:
            finalMinMax<128 > << < Grid_dim, Block_dim, smemSize >> > (Res_MinsMaxs, constant, min, max); break;
        case 64:
            finalMinMax<64  > << < Grid_dim, Block_dim, smemSize >> > (Res_MinsMaxs, constant, min, max); break;
        case 32:
            finalMinMax<32  > << < Grid_dim, Block_dim, smemSize >> > (Res_MinsMaxs, constant, min, max); break;
        case 16:
            finalMinMax<16  > << < Grid_dim, Block_dim, smemSize >> > (Res_MinsMaxs, constant, min, max); break;
        case 8:
            finalMinMax<8   > << < Grid_dim, Block_dim, smemSize >> > (Res_MinsMaxs, constant, min, max); break;
        case 4:
            finalMinMax<4   > << < Grid_dim, Block_dim, smemSize >> > (Res_MinsMaxs, constant, min, max); break;
        case 2:
            finalMinMax<2   > << < Grid_dim, Block_dim, smemSize >> > (Res_MinsMaxs, constant, min, max); break;
        case 1:
            finalMinMax<1   > << < Grid_dim, Block_dim, smemSize >> > (Res_MinsMaxs, constant, min, max); break;
    }

    //These are basic calls there is one thread for one pixel and each thread substract min and multiply with the constant
    Block_dim = dim3(THREAD_COUNT, 1, 1);
    Grid_dim = dim3(LENA_SIZE / THREAD_COUNT, 1, 1);

    calcPixelVal<< < Grid_dim, Block_dim >> > (Lena_d, constant, min);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float et;
    cudaEventElapsedTime(&et, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);


    // Retrieve result from device and store it in host array
    cudaMemcpy(Lena_Original_h, Lena_d, vector_size, cudaMemcpyDeviceToHost);

    WritePGM("lena_after.pgm", Lena_Original_h, nWidth, nHeight, nMaxGray);

    free(Lena_Original_h);
    cudaFree(Lena_d);
    cudaFree(Res_MinsMaxs);
    cudaFree(min);
    cudaFree(max);
    cudaFree(constant);
    printf("GPU execution time= %f ms\n", et);
}
//For reading .pgm file
float* LoadPGM(char * sFileName, int & nWidth, int & nHeight, int & nMaxGray)
{
    char aLine[256];
    FILE * fInput = fopen(sFileName, "r");
    if (fInput == 0)
    {
        perror("Cannot open file to read");
        exit(EXIT_FAILURE);
    }
    // First line: version
    fgets(aLine, 256, fInput);
    std::cout << "\tVersion: " << aLine;
    // Second line: comment
    fgets(aLine, 256, fInput);
    std::cout << "\tComment: " << aLine;
    fseek(fInput, -1, SEEK_CUR);
    // Third line: size
    fscanf(fInput, "%d", &nWidth);
    std::cout << "\tWidth: " << nWidth;
    fscanf(fInput, "%d", &nHeight);
    std::cout << " Height: " << nHeight << std::endl;
    // Fourth line: max value
    fscanf(fInput, "%d", &nMaxGray);
    std::cout << "\tMax value: " << nMaxGray << std::endl;
    while (getc(fInput) != '\n');
    // Following lines: data
    float * pSrc_Host = new float[nWidth * nHeight];
    for (int i = 0; i < nHeight; ++i)
        for (int j = 0; j < nWidth; ++j)
            pSrc_Host[i*nWidth + j] = fgetc(fInput);
    fclose(fInput);

    return pSrc_Host;
}
//For writing .pgm file
void WritePGM(char * sFileName, float * pDst_Host, int nWidth, int nHeight, int nMaxGray)
{
    FILE * fOutput = fopen(sFileName, "w+");
    if (fOutput == 0)
    {
        perror("Cannot open file to read");
        exit(EXIT_FAILURE);
    }
    char * aComment = "# Created by NPP";
    fprintf(fOutput, "P5\n%s\n%d %d\n%d\n", aComment, nWidth, nHeight, nMaxGray);
    for (int i = 0; i < nHeight; ++i)
        for (int j = 0; j < nWidth; ++j)
            fputc(pDst_Host[i*nWidth + j], fOutput);
    fclose(fOutput);
}