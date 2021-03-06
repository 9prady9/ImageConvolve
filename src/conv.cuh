#ifndef __CONV_H__
#define __CONV_H__

#include <cstdio>
#include <curand_kernel.h>

typedef unsigned char uchar;
typedef unsigned int  uint;

const int TILE_WIDTH = 32;
const int TILE_HEIGHT = 32;

#define USE_CUDA_TEX_OBJECT 1

void HandleError( cudaError_t err, const char *file, int line );

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

void CheckError(void);

#define CHECK_CUDA_ERRORS() (CheckError())

int ceil(int numer, int denom);

class MemObject {
public:
	// Methods
	MemObject();
	void cleanMemory();
	~MemObject();

	// Attributes
	uint		mKernelSize;
	uint		mImageWidth;
	uint		mImageHeight;
	uchar*		dev_SourceImage;
	uchar*		dev_ConvolvedImage;
	cudaArray*	cuImgArray;

	cudaChannelFormatDesc	channelDesc;
	cudaResourceDesc		resDesc;
	cudaTextureDesc			texDesc;
	cudaTextureObject_t		texObj;
};

void initMemObject(void);

void setKernelOnDevice(float const * elements, const int count);

void setImageOnDevice(const uchar * image_data, const int image_width, const int image_height);

void convolve(const int kernel_radius);

void memCpyImageDeviceToHost(uchar* host_ptr);

#endif //__CONV_H__
