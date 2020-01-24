#include "conv.cuh"

#include <iostream>


void HandleError( cudaError_t err, const char *file, int line )
{
    if (err != cudaSuccess)
	{
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
		getchar();
        exit( EXIT_FAILURE );
    }
}

void CheckError(void)
{
#ifdef _DEBUG_PRINTS_
	cudaDeviceSynchronize();
	HANDLE_ERROR( cudaPeekAtLastError() );
#endif
}

__constant__ float d_kernel[81];

__inline__ __device__ uchar4 getRGBA(const uchar* fSource,
						const int fImageWidth,
						const int fImageHeight,
						const int row,
						const int col)
{
	uchar4 retVal;
	int ii	= row<0 ? 0 : row;
	int jj	= col>=fImageHeight ? fImageHeight-1 : col;
	int indx= 4*(jj*fImageWidth + ii);
	retVal.x= fSource[indx+2];
	retVal.y= fSource[indx+1];
	retVal.z= fSource[indx+0];
	retVal.w= fSource[indx+3];
	return retVal;
}

__inline__ __device__ void setRGBA(uchar* fDestination,
						const int fImageWidth,
						const int fImageHeight,
						const int row,
						const int col,
						float4 value)
{
	int ii	= row<0 ? 0 : row;
	int jj	= col>=fImageHeight ? fImageHeight-1 : col;
	int indx= 4*(jj*fImageWidth + ii);
	fDestination[indx+2] = value.x;
	fDestination[indx+1] = value.y;
	fDestination[indx+0] = value.z;
	fDestination[indx+3] = value.w;
}

__global__ void convolveKernel(const uchar* fSource, int fImageWidth, int fImageHeight, uchar* fDestination, int fKernelSize)
{
	extern __shared__ uchar shared[];
	const int PADDING = 2*fKernelSize;

	int slen	= blockDim.x+PADDING;
	int klen	= 2*fKernelSize+1;
	int gx		= threadIdx.x + blockDim.x * blockIdx.x;
	int gy		= threadIdx.y + blockDim.y * blockIdx.y;
	int sidx	= 4*(threadIdx.y*slen+threadIdx.x);

	uchar4 pxl	= getRGBA(fSource,fImageWidth,fImageHeight,
						gx-fKernelSize,gy-fKernelSize);

	shared[sidx+0] = pxl.x;
	shared[sidx+1] = pxl.y;
	shared[sidx+2] = pxl.z;
	shared[sidx+3] = pxl.w;

	int ti	= threadIdx.x + fKernelSize;
	int tj	= threadIdx.y + fKernelSize;
	int lx2	= threadIdx.x + blockDim.x;
	int ly2	= threadIdx.y + blockDim.y;
	int gx2	= gx + blockDim.x;
	int gy2	= gy + blockDim.y;

	if( threadIdx.x < PADDING ) {
		pxl	= getRGBA(fSource,fImageWidth,fImageHeight,
						gx2-fKernelSize,gy-fKernelSize);
		sidx= 4*(threadIdx.y*slen+lx2);

		shared[sidx+0] = pxl.x;
		shared[sidx+1] = pxl.y;
		shared[sidx+2] = pxl.z;
		shared[sidx+3] = pxl.w;
	}

	if( threadIdx.y < PADDING ) {
		pxl	= getRGBA(fSource,fImageWidth,fImageHeight,
						gx-fKernelSize,gy2-fKernelSize);
		sidx= 4*(ly2*slen+threadIdx.x);

		shared[sidx+0] = pxl.x;
		shared[sidx+1] = pxl.y;
		shared[sidx+2] = pxl.z;
		shared[sidx+3] = pxl.w;
	}

	if( threadIdx.x < PADDING && threadIdx.y < PADDING ) {
		pxl	= getRGBA(fSource,fImageWidth,fImageHeight,
						gx2-fKernelSize,gy2-fKernelSize);
		sidx= 4*(ly2*slen+lx2);

		shared[sidx+0] = pxl.x;
		shared[sidx+1] = pxl.y;
		shared[sidx+2] = pxl.z;
		shared[sidx+3] = pxl.w;
	}

	__syncthreads();

	// Now that the image has been read
	// into shared memory completely.
	// Check for image bounds and exit
	if( gx >= fImageWidth || gy >= fImageHeight )
		return;

	sidx			= 4*(tj*slen+ti);
	uchar* ptr		= shared + sidx;
	float4 accum	= {0.0f,0.0f,0.0f,0.0f};

	for( int jj=-fKernelSize; jj<=fKernelSize; jj++ )
	{
		for( int ii=-fKernelSize; ii<=fKernelSize; ii++ )
		{
			int tmpidx	= 4*(jj*slen+ii);
			float weight= d_kernel[(fKernelSize+jj)*klen+(fKernelSize+ii)];
			accum.x		+= weight*ptr[tmpidx+0];
			accum.y		+= weight*ptr[tmpidx+1];
			accum.z		+= weight*ptr[tmpidx+2];
		}
	}
	accum.w	= shared[sidx+3];

	setRGBA(fDestination,fImageWidth,fImageHeight,gx,gy,accum);
}

__global__ void convolveKernel(cudaTextureObject_t fSource, int fImageWidth, int fImageHeight, uchar* fDestination, int fKernelSize)
{
	extern __shared__ uchar shared[];
	const int PADDING = 2*fKernelSize;

	int slen	= blockDim.x + PADDING;
	int klen	= PADDING + 1;
	int gx		= threadIdx.x + blockDim.x * blockIdx.x;
	int gy		= threadIdx.y + blockDim.y * blockIdx.y;
	int sidx	= 4*(threadIdx.y*slen+threadIdx.x);

	uchar4 pxl	= tex2D<uchar4>(fSource,gx-fKernelSize,gy-fKernelSize);

	shared[sidx+0] = pxl.x;
	shared[sidx+1] = pxl.y;
	shared[sidx+2] = pxl.z;
	shared[sidx+3] = pxl.w;

	int ti	= threadIdx.x + fKernelSize;
	int tj	= threadIdx.y + fKernelSize;
	int lx2	= threadIdx.x + blockDim.x;
	int ly2	= threadIdx.y + blockDim.y;
	int gx2	= gx + blockDim.x;
	int gy2	= gy + blockDim.y;

	if( threadIdx.x < PADDING ) {
		pxl	= tex2D<uchar4>(fSource,gx2-fKernelSize,gy-fKernelSize);
		sidx= 4*(threadIdx.y*slen+lx2);

		shared[sidx+0] = pxl.x;
		shared[sidx+1] = pxl.y;
		shared[sidx+2] = pxl.z;
		shared[sidx+3] = pxl.w;
	}

	if( threadIdx.y < PADDING ) {
		pxl	= tex2D<uchar4>(fSource,gx-fKernelSize,gy2-fKernelSize);
		sidx= 4*(ly2*slen+threadIdx.x);

		shared[sidx+0] = pxl.x;
		shared[sidx+1] = pxl.y;
		shared[sidx+2] = pxl.z;
		shared[sidx+3] = pxl.w;
	}

	if( threadIdx.x < PADDING && threadIdx.y < PADDING ) {
		pxl	= tex2D<uchar4>(fSource,gx2-fKernelSize,gy2-fKernelSize);
		sidx= 4*(ly2*slen+lx2);

		shared[sidx+0] = pxl.x;
		shared[sidx+1] = pxl.y;
		shared[sidx+2] = pxl.z;
		shared[sidx+3] = pxl.w;
	}

	__syncthreads();

	// Now that the image has been read
	// into shared memory completely.
	// Check for image bounds and exit
	if( gx >= fImageWidth || gy >= fImageHeight )
		return;

	sidx			= 4*(tj*slen+ti);
	uchar* ptr		= shared + sidx;
	float4 accum	= {0.0f,0.0f,0.0f,0.0f};

	for( int jj=-fKernelSize; jj<=fKernelSize; jj++ )
	{
		for( int ii=-fKernelSize; ii<=fKernelSize; ii++ )
		{
			int tmpidx	= 4*(jj*slen+ii);
			float weight= d_kernel[(fKernelSize+jj)*klen+(fKernelSize+ii)];
			accum.x		+= weight*ptr[tmpidx+0];
			accum.y		+= weight*ptr[tmpidx+1];
			accum.z		+= weight*ptr[tmpidx+2];
		}
	}
	accum.w	= shared[sidx+3];

	setRGBA(fDestination,fImageWidth,fImageHeight,gx,gy,accum);
}


int ceil(int numer, int denom)
{
	return (numer/denom + (numer % denom != 0));
}

MemObject::MemObject()
{
	dev_SourceImage		= 0;
	dev_ConvolvedImage	= 0;
    cuImgArray          = 0;

	// CUDA texture specification
	memset(&resDesc,0,sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	// CUDA texture object parameters
	memset(&texDesc,0,sizeof(texDesc));
	texDesc.addressMode[0]	= cudaAddressModeWrap;
	texDesc.addressMode[1]	= cudaAddressModeWrap;
	texDesc.filterMode		= cudaFilterModePoint;
	texDesc.readMode		= cudaReadModeElementType;
	texDesc.normalizedCoords= 0;

    texObj = 0;
}

void MemObject::cleanMemory()
{
#if USE_CUDA_TEX_OBJECT
    if (texObj) {
        cudaDestroyTextureObject(texObj);
    }
    if (cuImgArray) {
        cudaFreeArray(cuImgArray);
    }
    cuImgArray = 0;
	texObj     = 0;
#else
	if (dev_SourceImage) {
        cudaFree(dev_SourceImage);
    }
#endif
    if (dev_ConvolvedImage) {
        cudaFree(dev_ConvolvedImage);
    }
	dev_SourceImage		= 0;
	dev_ConvolvedImage	= 0;
}

MemObject::~MemObject()
{
	cleanMemory();
}

MemObject* getMemObject(void)
{
	static MemObject* handle = 0;
	if( handle == 0 ) {
		handle = new MemObject();
	}
	return handle;
}

void initMemObject(void)
{
	getMemObject();
}

void setKernelOnDevice(float const * elements, const int count)
{
	HANDLE_ERROR( cudaMemcpyToSymbol(d_kernel, elements, count*sizeof(float)) );
	CHECK_CUDA_ERRORS();
}

void setImageOnDevice(const uchar * image_data, const int image_width, const int image_height)
{
	MemObject* handle = getMemObject();
	handle->cleanMemory();

	handle->mImageWidth		= image_width;
	handle->mImageHeight	= image_height;

#if USE_CUDA_TEX_OBJECT
	handle->channelDesc = cudaCreateChannelDesc<uchar4>();
	HANDLE_ERROR( cudaMallocArray(&(handle->cuImgArray), &(handle->channelDesc), image_width, image_height) );
	HANDLE_ERROR( cudaMemcpyToArray(handle->cuImgArray, 0,0, image_data,
							image_width*image_height*4*sizeof(uchar),
							cudaMemcpyHostToDevice) );
	handle->resDesc.res.array.array = handle->cuImgArray;
	cudaCreateTextureObject(&(handle->texObj),&(handle->resDesc),&(handle->texDesc),NULL);

#else
	/* Allocate memory on device to hold image data */
	HANDLE_ERROR( cudaMalloc((void**)&handle->dev_SourceImage,
							 image_width*image_height*4*sizeof(uchar)) );
	CHECK_CUDA_ERRORS();

	/* Copy this data to device memory for kernel computation */
	HANDLE_ERROR( cudaMemcpy( handle->dev_SourceImage, image_data,
							  image_width*image_height*4*sizeof(uchar),
							  cudaMemcpyHostToDevice) );
	CHECK_CUDA_ERRORS();

#endif

	/* Allocate memory for output image on GPU device memory */
	HANDLE_ERROR( cudaMalloc((void**)&handle->dev_ConvolvedImage,
							 image_width*image_height*4*sizeof(uchar)) );
	CHECK_CUDA_ERRORS();
}

void convolve(const int kernel_radius)
{
	static dim3	mThreadsPerBlock(TILE_WIDTH,TILE_HEIGHT);

	MemObject* handle = getMemObject();
	int image_width = handle->mImageWidth;
	int image_height = handle->mImageHeight;

	dim3	mGrid(ceil(image_width, mThreadsPerBlock.x),
				  ceil(image_height, mThreadsPerBlock.y));

	int sharedMemSize = (mThreadsPerBlock.y+2*kernel_radius)*
						(mThreadsPerBlock.x+2*kernel_radius)*
						4*sizeof(uchar);

#ifdef _DEBUG_PRINTS_
	std::cout<<"Threads per block "<<mThreadsPerBlock.x<<","<<mThreadsPerBlock.y<<std::endl;
	std::cout<<"Blocks per grid "<<mGrid.x<<","<<mGrid.y<<std::endl;;
	std::cout<<"Shared memory usage : "<<sharedMemSize<<" Bytes"<<std::endl;;
#endif

#if USE_CUDA_TEX_OBJECT
	convolveKernel<<<mGrid,mThreadsPerBlock,sharedMemSize>>>(handle->texObj,
															 image_width,
															 image_height,
															 handle->dev_ConvolvedImage,
															 kernel_radius);
#else
	convolveKernel<<<mGrid,mThreadsPerBlock,sharedMemSize>>>(handle->dev_SourceImage,
															 image_width, image_height,
															 handle->dev_ConvolvedImage,
															 kernel_radius);
#endif

	CHECK_CUDA_ERRORS();
	cudaDeviceSynchronize();
}

void memCpyImageDeviceToHost(uchar* host_ptr)
{
	MemObject* handle = getMemObject();
	HANDLE_ERROR( cudaMemcpy(host_ptr, handle->dev_ConvolvedImage,
							 handle->mImageWidth*handle->mImageHeight*4*sizeof(uchar),
							 cudaMemcpyDeviceToHost) );

	CHECK_CUDA_ERRORS();
}
