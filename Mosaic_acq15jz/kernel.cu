/**
* Program name: Mosaic filter with average color calculation
* Compiling envrionment: VS2013 + CUDA7.0
*
* @Author: Jincao Zhang
* @Version 1.0 (22 Sep 2016)
*/



#ifndef __CUDACC__
#define __CUDACC__
#endif

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector_types.h>
#include <vector_functions.h>
#include "cuda_texture_types.h"
#include "texture_fetch_functions.hpp"
#include <conio.h>
#include <math.h>
#include <time.h>
#include <omp.h>


#define FAILURE 0
#define SUCCESS !FAILURE

#define USER_NAME "acq15jz"		



void print_help();//print command line help 
int process_command_line(int argc, char *argv[]);
int getImageSize(char* filename);// get input image dimentions
void output_image_file(char* filename, uchar4* image);//output image as ppm 
void input_image_file(char* filename, uchar4* image);//input image as ppm
void checkCUDAError(const char *msg);

//texture<uchar4, cudaTextureType2D, cudaReadModeElementType> sample2D;

typedef enum MODE { CPU, OPENMP, CUDA, ALL } MODE; //define program mode 

int IMAGE_DIM;

int c; // mosaic cell size
MODE execution_mode = CPU; 
char* input_FileName;
char* output_FileName;
unsigned int ppm_format = 0;

/**
* CUDA kernel of mosaic filter
*
* @param image input
*		 image_output output
*		 Tile_size  Mosaic cell size
*		 IMAGE_DIM  size of input images
*/
__global__ void Mosaic(uchar4 *image, uchar4 *image_output, int Tile_size, int IMAGE_DIM) {
	// map from threadIdx/BlockIdx to pixel position
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int i = x + y * blockDim.x * gridDim.x;

	//Mosaic index offset
	int Mosaic_IndX = (x / Tile_size)*Tile_size;
	int Mosaic_IndY = (y / Tile_size)*Tile_size;

	int redsum;         // sum of red
	int greensum;       // sum of green
	int bluesum;        // sum of blue
	int count;          // pixel count in a mosaic block
	uchar4* pixel_temp;
	float4 average = make_float4(0, 0, 0, 0);

	redsum = greensum = bluesum = count = 0;

	for (int i = 0; i <= Tile_size; i++){
		for (int j = 0; j <= Tile_size; j++){
			int x_offset = Mosaic_IndX + i;
			int y_offset = Mosaic_IndY + j;
			//handle boundry condition
			if (x_offset >= IMAGE_DIM)
				x_offset = IMAGE_DIM-1;

			if (y_offset >= IMAGE_DIM)
				y_offset = IMAGE_DIM-1;
			int offset = x_offset + y_offset * blockDim.x * gridDim.x;
			pixel_temp = &image[offset];
			
			//pixel_temp = tex2D(sample2D, x_offset, y_offset);

			//sum values
			redsum += pixel_temp->x;
			greensum += pixel_temp->y;
			bluesum += pixel_temp->z;
			count++;
		}
	}

	//calculate average
	average.x = redsum / count;
	average.y = greensum / count;
	average.z = bluesum / count;

	//output image
	image_output[i].x = (unsigned char)average.x;
	image_output[i].y = (unsigned char)average.y;
	image_output[i].z = (unsigned char)average.z;
	image_output[i].w = 255;

}

int main(int argc, char *argv[]) {

	if (process_command_line(argc, argv) == FAILURE)
		return 1;

	unsigned int image_size;

	uchar4 *d_image, *d_image_output;
	uchar4 *h_image, *h_output;

	cudaEvent_t start, stop;
	clock_t begin, end;
	double seconds;//cpu and openmp timer
	float ms;//cuda timer

	IMAGE_DIM = getImageSize(input_FileName);

	image_size = IMAGE_DIM*IMAGE_DIM*sizeof(uchar4);

	//TODO: read input image file (either binary or plain text PPM) 
	h_image = (uchar4*)malloc(image_size);
	input_image_file(input_FileName, h_image);

	//TODO: execute the mosaic filter based on the mode
	switch (execution_mode){
	case (CPU) : {
					 
					 h_output = (uchar4*)malloc(image_size);
					 int redsum;         // sum of red
					 int greensum;       // sum of green
					 int bluesum;        // sum of blue
					 int count;          // pixel count in a mosaic block
					 int x, y, tx, ty;
					 uchar4* pixel_temp;       
					 int r, g, b; r = 0; g = 0; b = 0;


					 //TODO: starting timing here
					 begin = clock();
					 //-----------------------------------------------//
					 for (y = 0; y < IMAGE_DIM; y++){
						 for (x = 0; x < IMAGE_DIM; x++){
							 int block_indX = (x / c)*c;
							 int block_indY = (y / c)*c;
							 redsum = greensum = bluesum = count = 0;
							 for (ty = 0; ty < c; ty++){
								 for (tx = 0; tx < c; tx++){
									 int x_offset = block_indX + tx;
									 int y_offset = block_indY + ty;
									 if (x_offset >= IMAGE_DIM)
										 x_offset = IMAGE_DIM-1;

									 if (y_offset >= IMAGE_DIM)
										 y_offset = IMAGE_DIM-1;
									 int offset = x_offset + y_offset * IMAGE_DIM;
									 pixel_temp = &h_image[offset];
									 //sum values
									 redsum += pixel_temp->x;
									 greensum += pixel_temp->y;
									 bluesum += pixel_temp->z;
									 count++;

								 }
							 }
							 r = redsum / count;
							 g = greensum / count;
							 b = bluesum / count;
							 for (ty = 0; ty < c; ty++){
								 for (tx = 0; tx < c; tx++){
									 int x_offset = block_indX + tx;
									 int y_offset = block_indY + ty;
									 if (x_offset >= IMAGE_DIM)
										 x_offset = IMAGE_DIM-1;

									 if (y_offset >= IMAGE_DIM)
										 y_offset = IMAGE_DIM-1;
									 int offset = x_offset + y_offset * IMAGE_DIM;
									 h_output[offset].x = (unsigned char)r;
									 h_output[offset].y = (unsigned char)g;
									 h_output[offset].z = (unsigned char)b;
									 h_output[offset].w = 255;
								 }
							 }
						 }
					 }

					 //TODO:calculate the average colour value
					 r = 0; g = 0; b = 0;
					 for (int j = 0; j < IMAGE_DIM; j += c){
						 for (int i = 0; i < IMAGE_DIM; i += c){
							 int offset = i + j*IMAGE_DIM;
							 pixel_temp = &h_output[offset];
							 redsum += pixel_temp->x;
							 greensum += pixel_temp->y;
							 bluesum += pixel_temp->z;
							 count++;

						 }
					 }
					 r = redsum / count;
					 g = greensum / count;
					 b = bluesum / count;
					 //-----------------------------------------------//
					 end = clock();
					 seconds = (end - begin) / (double)CLOCKS_PER_SEC;
					 output_image_file(output_FileName, h_output);
					 free(h_output);
					 // Output the average colour value for the image
					 printf("Serial CPU Average image colour red = %d, green = %d, blue = %d \n", r, g, b);

					 //TODO: implement part 3 of the assignment

					 //TODO: end timing here
					 printf("Serial CPU mode execution time took %f s \n", seconds);

					 break;
	}
	case (OPENMP) : {
						h_output = (uchar4*)malloc(image_size);
						int x, y;
						//TODO: starting timing here
						begin = clock();

						omp_set_nested(1);
#pragma omp parallel for private(y,x)   
						for (y = 0; y < IMAGE_DIM; y++){

							for (x = 0; x < IMAGE_DIM; x++){
								int block_indX = (x / c)*c;
								int block_indY = (y / c)*c;
								int redsum = 0;
								int greensum = 0;
								int bluesum = 0;
								int count = 0;
								int r, g, b; r = 0; g = 0; b = 0;
								int tx, ty;
								for (ty = 0; ty < c; ty++){

									for (tx = 0; tx < c; tx++){
										int x_offset = block_indX + tx;
										int y_offset = block_indY + ty;

										if (x_offset >= IMAGE_DIM)
											x_offset = IMAGE_DIM-1;

										if (y_offset >= IMAGE_DIM)
											y_offset = IMAGE_DIM-1;
										int offset = x_offset + y_offset * IMAGE_DIM;
										uchar4* pixel_temp = &h_image[offset];

										redsum += pixel_temp->x;

										greensum += pixel_temp->y;

										bluesum += pixel_temp->z;

										count++;


									}
								}

								r = redsum / count;
								g = greensum / count;
								b = bluesum / count;

								for (ty = 0; ty < c; ty++){

									for (tx = 0; tx < c; tx++){
										int x_offset = block_indX + tx;
										int y_offset = block_indY + ty;
										if (x_offset >= IMAGE_DIM)
											x_offset = IMAGE_DIM-1;

										if (y_offset >= IMAGE_DIM)
											y_offset = IMAGE_DIM-1;
										int offset = x_offset + y_offset * IMAGE_DIM;
										h_output[offset].x = (unsigned char)r;
										h_output[offset].y = (unsigned char)g;
										h_output[offset].z = (unsigned char)b;
										h_output[offset].w = 255;
									}
								}


							}
						}

						int r = 0; int g = 0; int b = 0;
						int redsum = 0, greensum = 0, bluesum = 0, count = 0;
#pragma omp parallel for reduction(+:redsum,greensum,bluesum,count)
						for (int j = 0; j < IMAGE_DIM; j += c){
							for (int i = 0; i < IMAGE_DIM; i += c){
								int offset = i + j*IMAGE_DIM;
								uchar4* pixel_temp = &h_output[offset];
								redsum += pixel_temp->x;
								greensum += pixel_temp->y;
								bluesum += pixel_temp->z;
								count++;

							}
						}
						r = redsum / count;
						g = greensum / count;
						b = bluesum / count;
						end = clock();
						//TODO: starting timing here

						seconds = (end - begin) / (double)CLOCKS_PER_SEC;
						//TODO: calculate the average colour value
						output_image_file(output_FileName,h_output);
						// Output the average colour value for the image
						printf("OpenMP CPU Average image colour red = %d, green = %d, blue = %d \n", r, g, b);
						printf("OpenMP CPU mode execution time took %f s \n", seconds);
					
						free(h_output);
						break;
	}
	case (CUDA) : {

					  h_output = (uchar4*)malloc(image_size);
					  // create timers
					  cudaEventCreate(&start);
					  cudaEventCreate(&stop);

					  // allocate memory on the GPU for the output image
					  cudaMalloc((void**)&d_image, image_size);
					  cudaMalloc((void**)&d_image_output, image_size);
					  checkCUDAError("CUDA malloc");

					  // copy image to device memory
					  cudaMemcpy(d_image, h_image, image_size, cudaMemcpyHostToDevice);
					  checkCUDAError("CUDA memcpy to device");

					  //cuda layout and execution
					  dim3    blocksPerGrid(IMAGE_DIM / 16, IMAGE_DIM / 16);
					  dim3    threadsPerBlock(16, 16);


					  //cudaBindTexture(0, sample1D, d_image, image_size);
					  //cudaBindTexture2D(0, sample2D, d_image, desc, IMAGE_DIM, IMAGE_DIM, IMAGE_DIM*sizeof(uchar4));
					  cudaEventRecord(start, 0);
					  Mosaic << <blocksPerGrid, threadsPerBlock >> >(d_image, d_image_output, c, IMAGE_DIM);
					  cudaEventRecord(stop, 0);
					  cudaEventSynchronize(stop);
					  cudaEventElapsedTime(&ms, start, stop);
					  //cudaUnbindTexture(sample1D);
					  //cudaUnbindTexture(sample2D);
					  checkCUDAError("kernel");

					  cudaMemcpy(h_output, d_image_output, image_size, cudaMemcpyDeviceToHost);
					  checkCUDAError("CUDA memcpy from device");

					  output_image_file(output_FileName, h_output);

					  //cleanup
					  cudaEventDestroy(start);
					  cudaEventDestroy(stop);
					  cudaFree(d_image);
					  cudaFree(d_image_output);

					
					  //TODO: calculate the average colour value

					  // Output the average colour value for the image
					  printf("CUDA Average image colour red = ???, green = ???, blue = ??? \n");


					  printf("CUDA mode execution time took %f s \n", ms/1000);
					  break;
	}
	case (ALL) : {
					 //ToDo
					 break;
	}
	}

	
	free(h_image);
	return 0;
}

void print_help(){
	printf("mosaic_%s C M -i input_file -o output_file [options]\n", USER_NAME);

	printf("where:\n");
	printf("\tC              Is the mosaic cell size in range 0->32\n");
	printf("\tM              Is the mode with a value of either CPU, OPENMP, CUDA or\n"
		"\t               ALL. The mode specifies which version of the simulation\n"
		"\t               code should execute. ALL should execute each mode in\n"
		"\t               turn.\n");
	printf("\t-i input_file  Specifies an input image file\n");
	printf("\t-o output_file Specifies an output image file which will be used\n"
		"\t               to write the mosaic image\n");
	printf("[options]:\n");
	printf("\t-f ppm_format  PPM image output format either PPM_BINARY (default) or \n"
		"\t               PPM_PLAIN_TEXT\n "
		"\t -part3        Any options for part 3 of the assignment\n");
}

int process_command_line(int argc, char *argv[]){
	if (argc < 7){
		fprintf(stderr, "Error: Missing program arguments. Correct usage is...\n");
		print_help();
		return FAILURE;
	}

	//first argument is always the executable name
	if (strcmp(argv[0],"Mosaic_acq15jz.exe")!=0){
		fprintf(stderr, "Error: Wrong program name arguments. Correct usage is...\n");
		print_help();
		return FAILURE;
	}


	//read in the non optional command line arguments
	c = (unsigned int)atoi(argv[1]);
	printf("cell size %d\n",c);

	//TODO: read in the mode
	if (strcmp(argv[2], "CPU") == 0)execution_mode = CPU;
	else if (strcmp(argv[2], "OPENMP") == 0)execution_mode = OPENMP;
	else if (strcmp(argv[2], "CUDA") == 0)execution_mode = CUDA;
	else{
		fprintf(stderr, "Error: Wrong Mode arguments. Correct usage is...\n");
		print_help();
		return FAILURE;
	}


	//TODO: read in the input image name
	if (strcmp(argv[3], "-i") == 0)input_FileName = argv[4]; 
	else{
		fprintf(stderr, "Error: Wrong program arguments. Correct usage is...\n");
		print_help();
		return FAILURE;
	}printf("filenameinput %s \n", input_FileName);

	//TODO: read in the output image name
	if (strcmp(argv[5], "-o") == 0)output_FileName = argv[6];
	else{
		fprintf(stderr, "Error: Wrong program arguments. Correct usage is...\n");
		print_help();
		return FAILURE;
	}printf("filenameoutput %s \n", output_FileName);


	return SUCCESS;
}

void output_image_file(char* filename,uchar4* image)
{
	FILE *f; //output file handle

	//open the output file and write header info for PPM filetype
	f = fopen(filename, "wb");
	if (f == NULL){
		fprintf(stderr, "Error opening %s output file\n", output_FileName);
		exit(1);
	}
	fprintf(f, "P6\n");
	fprintf(f, "%d \n%d\n%d\n", IMAGE_DIM, IMAGE_DIM, 255);
	for (int x = 0; x < IMAGE_DIM; x++){
		for (int y = 0; y < IMAGE_DIM; y++){
			int i = x + y*IMAGE_DIM;
			fwrite(&image[i], sizeof(unsigned char), 3, f); //only write rgb (ignoring a)
		}
	}

	fclose(f);
}

void input_image_file(char* filename, uchar4* image)
{
	FILE *f; //input file handle
	char temp[256];
	int h, w, s;

	//open the input file and write header info for PPM filetype
	f = fopen(filename, "rb");
	if (f == NULL){
		fprintf(stderr, "Error opening %s input file\n",input_FileName);
		exit(1);
	}
	fscanf(f, "%s\n", &temp);
	fscanf(f, "%d\n", &h); 
	fscanf(f, "%d\n", &w);
	fscanf(f, "%d\n", &s);
	if (h != w){
		fprintf(stderr, "Error: Input image file has wrong dimensions\n");
		exit(1);
	}

	for (int x = 0; x < h; x++){
		for (int y = 0; y < w; y++){
			int i = x + y*w;
			fread(&image[i], sizeof(unsigned char), 3, f); //only read rgb
			//image[i].w = 255;
		}
	}

	fclose(f);
}

int getImageSize(char* filename)
{
	FILE *f; //input file handle
	char temp[256];
	int x, y, s;

	//open the input file and write header info for PPM filetype
	f = fopen(filename, "rb");
	if (f == NULL){
		fprintf(stderr, "Error opening %s input file for getting image size\n", input_FileName);
		exit(1);
	}
	fscanf(f, "%s\n", &temp);
	fscanf(f, "%d\n %d\n", &x, &y); printf("img dim is:  %d\n", x); 
	fscanf(f, "%d\n", &s);
	if (x != y){
		fprintf(stderr, "Error: Input image file has wrong dimensions\n");
		exit(1);
	}


	fclose(f);
	return x;

}

void checkCUDAError(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		fprintf(stderr, "CUDA ERR1OR: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

////launching parameters
//dim3    blocksPerGrid((IMAGE_DIM + Mosaic_size - 1) / 32, (IMAGE_DIM + Mosaic_size - 1) / 32);
//
//dim3    threadsPerBlock(32, 32);
//__global__ void _Mosaic(uchar4 *image, uchar4 *image_output, int Tile_size) {
//	//	 map from threadIdx/BlockIdx to pixel position
//	int x, y, i;
//	__shared__ uchar4 sdata[32][32];
//	if (threadIdx.x < Tile_size&&threadIdx.y < Tile_size)
//	{
//		if (threadIdx.x + Tile_size * blockIdx.x < IMAGE_DIM&&threadIdx.y + Tile_size * blockIdx.y < IMAGE_DIM){
//			x = threadIdx.x + Tile_size * blockIdx.x;
//			y = threadIdx.y + Tile_size * blockIdx.y;
//			i = x + y * blockDim.x * gridDim.x;
//			sdata[threadIdx.x][threadIdx.y] = image[i];
//		}
//
//	}
//
//	__syncthreads();
//
//	int m, n;
//	int redsum;
//	int greensum;
//	int bluesum;
//	int count;
//	uchar4 pixel_temp;
//	float4 average = make_float4(0, 0, 0, 0);
//
//	redsum = greensum = bluesum = count = 0;
//	if (threadIdx.x < Tile_size&&threadIdx.y < Tile_size)
//	{
//		if (threadIdx.x + Tile_size * blockIdx.x < IMAGE_DIM&&threadIdx.y + Tile_size * blockIdx.y < IMAGE_DIM){
//			for (m = 0; m < Tile_size; m++){
//				for (n = 0; n < Tile_size; n++){
//
//					//pixel_temp = sdata[m][n];
//					pixel_temp = image[i];
//
//					//sum values
//					redsum += pixel_temp.x;
//					greensum += pixel_temp.y;
//					bluesum += pixel_temp.z;
//					count++;
//
//				}
//
//			}
//
//			//calculate average
//			average.x = redsum / count;
//			average.y = greensum / count;
//			average.z = bluesum / count;
//
//			image_output[i].x = (unsigned char)average.x;
//			image_output[i].y = (unsigned char)average.y;
//			image_output[i].z = (unsigned char)average.z;
//			image_output[i].w = 255;
//		}
//
//	}
//
//}