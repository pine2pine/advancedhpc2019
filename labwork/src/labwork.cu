#include <stdio.h>
#include <include/labwork.h>
#include <cuda_runtime_api.h>
#include <omp.h>

#define ACTIVE_THREADS 4

int main(int argc, char **argv) {
    printf("USTH ICT Master 2018, Advanced Programming for HPC.\n");
    if (argc < 2) {
        printf("Usage: labwork <lwNum> <inputImage>\n");
        printf("   lwNum        labwork number\n");
        printf("   inputImage   the input file name, in JPEG format\n");
        return 0;
    }

    int lwNum = atoi(argv[1]);
    std::string inputFilename;

    // pre-initialize CUDA to avoid incorrect profiling
    printf("Warming up...\n");
    char *temp;
    cudaMalloc(&temp, 1024);

    Labwork labwork;
    if (lwNum != 2 ) {
        inputFilename = std::string(argv[2]);
        labwork.loadInputImage(inputFilename);
    }

    printf("Starting labwork %d\n", lwNum);
    Timer timer;
    timer.start();
    switch (lwNum) {
        case 1:
            labwork.labwork1_CPU();
            labwork.saveOutputImage("labwork2-cpu-out.jpg");
            printf("labwork 1 CPU ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            timer.start();
            labwork.labwork1_OpenMP();
            labwork.saveOutputImage("labwork2-openmp-out.jpg");
            break;
        case 2:
            labwork.labwork2_GPU();
            break;
        case 3:
            labwork.labwork3_GPU();
            labwork.saveOutputImage("labwork3-gpu-out.jpg");
            break;
        case 4:
            labwork.labwork4_GPU();
            labwork.saveOutputImage("labwork4-gpu-out.jpg");
            break;
        case 5:
            labwork.labwork5_CPU();
            labwork.saveOutputImage("labwork5-cpu-out.jpg");
            labwork.labwork5_GPU();
            labwork.saveOutputImage("labwork5-gpu-out.jpg");
            break;
        case 6:
            labwork.labwork6_GPU();
            labwork.saveOutputImage("labwork6-gpu-out.jpg");
            break;
        case 7:
            labwork.labwork7_GPU();
            printf("[ALGO ONLY] labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork7-gpu-out.jpg");
            break;
        case 8:
            labwork.labwork8_GPU();
            printf("[ALGO ONLY] labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork8-gpu-out.jpg");
            break;
        case 9:
            labwork.labwork9_GPU();
            printf("[ALGO ONLY] labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork9-gpu-out.jpg");
            break;
        case 10:
            labwork.labwork10_GPU();
            printf("[ALGO ONLY] labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork10-gpu-out.jpg");
            break;
    }
    printf("labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
}

void Labwork::loadInputImage(std::string inputFileName) {
    inputImage = jpegLoader.load(inputFileName);
}

void Labwork::saveOutputImage(std::string outputFileName) {
    jpegLoader.save(outputFileName, outputImage, inputImage->width, inputImage->height, 90);
}

void Labwork::labwork1_CPU() {
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = static_cast<char *>(malloc(pixelCount * 3));
    for (int j = 0; j < 100; j++) {     // let's do it 100 times, otherwise it's too fast!
        for (int i = 0; i < pixelCount; i++) {
            outputImage[i * 3] = (char) (((unsigned char) inputImage->buffer[i * 3] + (unsigned char) inputImage->buffer[i * 3 + 1] +
                                          (unsigned char) inputImage->buffer[i * 3 + 2]) / 3);
            outputImage[i * 3 + 1] = outputImage[i * 3];
            outputImage[i * 3 + 2] = outputImage[i * 3];
        }
    }
}

void Labwork::labwork1_OpenMP() {
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = static_cast<char *>(malloc(pixelCount * 3));

    // Parallel processing part (copied from the normal code)

    #pragma omp parallel for schedule(dynamic)

    for (int j = 0; j < 100; j++) {     // let's do it 100 times, otherwise it's too fast!
        for (int i = 0; i < pixelCount; i++) {
            outputImage[i * 3] = (char) (((unsigned char) inputImage->buffer[i * 3] + (unsigned char) inputImage->buffer[i * 3 + 1] +
                                          (unsigned char) inputImage->buffer[i * 3 + 2]) / 3);
            outputImage[i * 3 + 1] = outputImage[i * 3];
            outputImage[i * 3 + 2] = outputImage[i * 3];
        }
    }
}

int getSPcores(cudaDeviceProp devProp) {
    int cores = 0;
    int mp = devProp.multiProcessorCount;
    switch (devProp.major) {
        case 2: // Fermi
            if (devProp.minor == 1) cores = mp * 48;
            else cores = mp * 32;
            break;
        case 3: // Kepler
            cores = mp * 192;
            break;
        case 5: // Maxwell
            cores = mp * 128;
            break;
        case 6: // Pascal
            if (devProp.minor == 1) cores = mp * 128;
            else if (devProp.minor == 0) cores = mp * 64;
            else printf("Unknown device type\n");
            break;
        default:
            printf("Unknown device type\n");
            break;
    }
    return cores;
}

void Labwork::labwork2_GPU() {
    int nDevices = 0;
    // get all devices
    cudaGetDeviceCount(&nDevices);
    printf("Number total of GPU : %d\n\n", nDevices);
    for (int i = 0; i < nDevices; i++){
        // get informations from individual device
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        // extract properties of the GPU from the struct

	printf("GPU #%d:\n", i);
	printf("===========\n");
	printf("      Identifier: %s\n", prop.name);
	printf("      Clock frequency (kHz): %d\n", prop.clockRate);
	printf("      Number of multiprocessors: %d\n", prop.multiProcessorCount);
	printf("      Warp size: %d\n", prop.warpSize);
	printf("      Memory clock frequency (kHz): %d\n", prop.memoryClockRate);
	printf("      Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
	printf("      Peak Memory Bandwidth (GB/s): %f\n\n", 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    }

}

__global__ void rgb2gray(uchar3 *input, uchar3 *output) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    output[tid].x = (input[tid].x + input[tid].y +input[tid].z) / 3;
    output[tid].z = output[tid].y = output[tid].x;
}

__global__ void rgb2gray2D(uchar3 *input, uchar3 *output, int pixelCount) {
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    int ty = threadIdx.y + blockIdx.y * blockDim.y;

    int tid = ty * blockDim.x * gridDim.x + tx;

    // we use more threads than needed (by having more blocks if necessary)
    // so we'll need to check if this particular thread is for a pixel that
    // doesn't exist. If that is the case, we simply return.

    if (tid >= pixelCount) return;

    output[tid].x = (input[tid].x + input[tid].y + input[tid].z) / 3;
    output[tid].z = output[tid].y = output[tid].x;
}

void Labwork::labwork3_GPU() {

    // Calculate number of pixels
    int pixelCount = inputImage->width * inputImage->height;

    // Allocate CUDA memory
    uchar3* devInput;
    uchar3* devGray;
    
    cudaMalloc(&devInput, pixelCount *sizeof(uchar3));
    cudaMalloc(&devGray, pixelCount *sizeof(uchar3));
    
    // Copy from host to device
    cudaMemcpy(devInput, inputImage->buffer, pixelCount * sizeof(uchar3), cudaMemcpyHostToDevice);

    // Processing
    int blockSize = 64;
    int numBlock = pixelCount / blockSize;
    rgb2gray<<<numBlock, blockSize>>>(devInput, devGray);

    // allocate memory on the host to receive output then copy from dev to host
    outputImage = static_cast<char *>(malloc(pixelCount * sizeof(uchar3)));
    cudaMemcpy(outputImage, devGray, pixelCount*sizeof(uchar3), cudaMemcpyDeviceToHost);

    // Cleaning
    cudaFree(devInput);
    cudaFree(devGray);
}

void Labwork::labwork4_GPU() {
    // Calculate number of pixels
    int pixelCount = inputImage->width * inputImage->height;

    // Allocate CUDA memory
    uchar3* devInput;
    uchar3* devGray;
    
    cudaMalloc(&devInput, pixelCount *sizeof(uchar3));
    cudaMalloc(&devGray, pixelCount *sizeof(uchar3));
    
    // Copy from host to device
    cudaMemcpy(devInput, inputImage->buffer, pixelCount * sizeof(uchar3), cudaMemcpyHostToDevice);

    // Define the block and grid dimensions for our kernel

    dim3 dimBlock(16, 8);  // Let's try block size = 4 * warp size
    dim3 dimGrid;

    int numOfBlocks = pixelCount / (dimBlock.x * dimBlock.y);

    if ( (pixelCount % (dimBlock.x * dimBlock.y)) > 0 ) {
	numOfBlocks++;
    }

    dimGrid.x = 8;
    dimGrid.y = numOfBlocks / dimGrid.x;
    if ( (numOfBlocks % dimGrid.x) > 0 ) {
    	dimGrid.y++;
    }

    // printf("Pixels: %d\n", pixelCount);
    // printf("Grid: %d x %d\n", dimGrid.x, dimGrid.y );

    // Launching our kernel
    rgb2gray2D<<<dimGrid, dimBlock>>>(devInput, devGray, pixelCount);

    // allocate memory on the host to receive output then copy from dev to host
    outputImage = static_cast<char *>(malloc(pixelCount * sizeof(uchar3)));
    cudaMemcpy(outputImage, devGray, pixelCount*sizeof(uchar3), cudaMemcpyDeviceToHost);

    // Cleaning
    cudaFree(devInput);
    cudaFree(devGray);
}

void Labwork::labwork5_CPU() {
    const unsigned char gaussianKernel[7][7] =
    {
        {  0,  0,  1,  2,  1,  0,  0},
        {  0,  3, 13, 22, 13,  3,  0},
        {  1, 13, 59, 97, 59, 13,  1},
        {  2, 22, 97,159, 97, 22,  2},
        {  1, 13, 59, 97, 59, 13,  1},
        {  0,  3, 13, 22, 13,  3,  0},
        {  0,  0,  1,  2,  1,  0,  0}
    };

    int pixelCount = inputImage->width * inputImage->height;
    outputImage = static_cast<char *>(malloc(pixelCount * sizeof(uchar3)));

    for (int y = 0; y < inputImage->height; y++) {
        for (int x = 0; x < inputImage->width; x++) {
            int rTotal = 0;
            int gTotal = 0;
            int bTotal = 0;
            for ( int dx = -3; dx < 4; dx++) {
                for (int dy = -3; dy < 4; dy++) {
                    if ( ((x+dx) >= 0) && ((x+dx) < inputImage->width) && ((y+dy) >= 0) && ((y+dy) < inputImage->height) )
                    {
                        int pixelIdx = (y+dy) * inputImage->width + (x+dx);
                        rTotal += (inputImage->buffer[pixelIdx*3] * gaussianKernel[dx+3][dy+3]);
                        gTotal += (inputImage->buffer[pixelIdx*3 + 1] * gaussianKernel[dx+3][dy+3]);
                        bTotal += (inputImage->buffer[pixelIdx*3 + 2] * gaussianKernel[dx+3][dy+3]);
                    } 
                }
            }
            int outputPixelIdx = inputImage->width * y + x;
            outputImage[outputPixelIdx*3] = rTotal / 1003;
            outputImage[outputPixelIdx*3 + 1] = gTotal / 1003;
            outputImage[outputPixelIdx*3 + 2] = bTotal / 1003;
        }
    }
}

void Labwork::labwork5_GPU() {
}

void Labwork::labwork6_GPU() {
}

void Labwork::labwork7_GPU() {
}

void Labwork::labwork8_GPU() {
}

void Labwork::labwork9_GPU() {

}

void Labwork::labwork10_GPU(){
}


























