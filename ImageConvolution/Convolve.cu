#include "Convolve.cuh"
#include <malloc.h>
#include <qimage.h>
#include <qdebug.h>

__constant__	int	d_kernel[81];

static int	mKernelSize;
uchar*		dev_SourceImage;
uchar*		dev_ConvolvedImage;

int ceil(int numer, int denom)
{
	return (numer/denom + (numer % denom != 0));
}

Kernel::Kernel()
{
	krValues = 0;
	kr = -1;
}

void Kernel::setKernelRadius(int fKernelRadius)
{
	if(krValues) free(krValues);
	kr = fKernelRadius;
	int temp = 2*fKernelRadius+1;
	krValues = (int*)malloc(temp*temp*sizeof(int));
}

void Kernel::setCellValue(int fRow, int fCol, int fValue)
{
	int temp = 2*kr+1;
	krValues[fRow*temp+fCol] = fValue;
}


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
						uint4 value)
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

	int slen	= blockDim.x+2*fKernelSize;
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

	if( threadIdx.x < fKernelSize ) {
		pxl	= getRGBA(fSource,fImageWidth,fImageHeight,
						gx2-fKernelSize,gy-fKernelSize);
		sidx= 4*(threadIdx.y*slen+lx2);

		shared[sidx+0] = pxl.x;
		shared[sidx+1] = pxl.y;
		shared[sidx+2] = pxl.z;
		shared[sidx+3] = pxl.w;
	}

	if( threadIdx.y < fKernelSize ) {
		pxl	= getRGBA(fSource,fImageWidth,fImageHeight,
						gx-fKernelSize,gy2-fKernelSize);
		sidx= 4*(ly2*slen+threadIdx.x);

		shared[sidx+0] = pxl.x;
		shared[sidx+1] = pxl.y;
		shared[sidx+2] = pxl.z;
		shared[sidx+3] = pxl.w;
	}

	if( threadIdx.x < fKernelSize && threadIdx.y < fKernelSize ) {
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
		
	sidx		= 4*(tj*slen+ti);
	uchar* ptr	= shared + sidx;	
	uint4 accum = {0,0,0,0};

	for( int jj=-fKernelSize; jj<=fKernelSize; jj++ )
	{
		for( int ii=-fKernelSize; ii<=fKernelSize; ii++ )
		{
			int tmpidx	= 4*(jj*slen+ii);
			int weight	= d_kernel[(fKernelSize+jj)*klen+(fKernelSize+ii)];
			accum.x		+= weight*ptr[tmpidx+0];
			accum.y		+= weight*ptr[tmpidx+1];
			accum.z		+= weight*ptr[tmpidx+2];
		}
	}
	accum.w	= shared[sidx+3];

	setRGBA(fDestination,fImageWidth,fImageHeight,gx,gy,accum);
}

Convolve::Convolve()
{
	mThreadsPerBlock.x	= 16;
	mThreadsPerBlock.y	= 16;
	mIsKernelSet		= false;
	mIsImageSet			= false;
	mIsCUDAInit			= false;
	dev_SourceImage		= 0;
	dev_ConvolvedImage	= 0;
	host_ConvolvedImage = 0;
}

void Convolve::setKernelData(const Kernel &fKernel)
{
	int temp			= fKernel.kr*2 + 1;
	HANDLE_ERROR( cudaMemcpyToSymbol(d_kernel, fKernel.krValues, temp*temp*sizeof(int)) );
	mKernelSize			= fKernel.kr;
	mIsKernelSet		= true;
}

void Convolve::setImageData(const uchar* fImageData, int fImageWidth, int fImageHeight)
{	
	free(host_ConvolvedImage);
	cleanMemory();

	mImageWidth		= fImageWidth;
	mImageHeight	= fImageHeight;
	/* Allocate memory on device to hold image data */
	HANDLE_ERROR( cudaMalloc((void**)&dev_SourceImage,mImageWidth*mImageHeight*4*sizeof(uchar)) );
	/* Copy this data to device memory for kernel computation */
	HANDLE_ERROR( cudaMemcpy( dev_SourceImage, fImageData, mImageWidth*mImageHeight*4*sizeof(uchar), cudaMemcpyHostToDevice) );
	mIsImageSet		= true;
	/* Allocate memory for output data on host */
	host_ConvolvedImage = (uchar*)malloc(mImageWidth*mImageHeight*4*sizeof(uchar));
	/* Allocate memory for output image on GPU device memory */
	HANDLE_ERROR( cudaMalloc((void**)&dev_ConvolvedImage, mImageWidth*mImageHeight*4*sizeof(uchar)) );
}

uchar* Convolve::getConvolvedImage()
{
	return host_ConvolvedImage;
}

void Convolve::cudaConvolve()
{
	initCUDA();
	if(mIsKernelSet && mIsImageSet && mIsCUDAInit)
	{
		int sharedMemSize = (mThreadsPerBlock.y+2*mKernelSize)*(mThreadsPerBlock.x+2*mKernelSize)*4*sizeof(uchar);
		cudaEventRecord(mStart,0);
		convolveKernel<<<mGrid,mThreadsPerBlock,sharedMemSize>>>( dev_SourceImage, mImageWidth, mImageHeight, dev_ConvolvedImage, mKernelSize );
		HANDLE_ERROR( cudaPeekAtLastError() );
		cudaDeviceSynchronize();
		HANDLE_ERROR( cudaPeekAtLastError() );
		cudaEventRecord(mStop,0);
		HANDLE_ERROR( cudaPeekAtLastError() );
		float computeTime = 0.0f;
		cudaEventSynchronize(mStop);
		HANDLE_ERROR( cudaPeekAtLastError() );
		cudaEventElapsedTime(&computeTime,mStart,mStop);
		qDebug()<<"computation time on device: "<<computeTime<<" ms\n";
		/* Copy back convolved image to host and show it */		
		HANDLE_ERROR( cudaMemcpy( host_ConvolvedImage, dev_ConvolvedImage, mImageWidth*mImageHeight*4*sizeof(uchar), cudaMemcpyDeviceToHost) );
		QImage output(host_ConvolvedImage,mImageWidth,mImageHeight, QImage::Format_ARGB32);
	} else {
		if(!mIsKernelSet)
			printf("Prerequisite: Kernel not setup. \n");
		if(!mIsImageSet)
			printf("Prerequisite: Image not setup. \n");
		if(!mIsCUDAInit)
			printf("Prerequisite: CUDA not initialized. \n");
	}
	destroyCUDA();
}

Convolve::~Convolve()
{
	cleanMemory();
	free(host_ConvolvedImage);
}

void Convolve::initCUDA()
{
	HANDLE_ERROR( cudaEventCreate( &mStart ) );
	HANDLE_ERROR( cudaEventCreate( &mStop ) );

	mThreadsPerBlock.x = TILE_WIDTH;
	mThreadsPerBlock.y = TILE_HEIGHT;
	mGrid.x = ceil( mImageWidth, mThreadsPerBlock.x );
	mGrid.y = ceil( mImageHeight, mThreadsPerBlock.y );

	qDebug()<<"Threads per block "<<mThreadsPerBlock.x<<","<<mThreadsPerBlock.y;
	qDebug()<<"Blocks per grid "<<mGrid.x<<","<<mGrid.y;
	int share  = (mThreadsPerBlock.y+2*mKernelSize)*(mThreadsPerBlock.x+2*mKernelSize)*4*sizeof(uchar);
	qDebug()<<"Shared memory usage : "<<share<<" Bytes";
	mIsCUDAInit= true;
}

void Convolve::cleanMemory()
{
	cudaFree(dev_SourceImage);
	cudaFree(dev_ConvolvedImage);
	dev_SourceImage		= 0;
	dev_ConvolvedImage	= 0;
}

void Convolve::destroyCUDA()
{
	HANDLE_ERROR( cudaEventDestroy(mStart) );
	HANDLE_ERROR( cudaEventDestroy(mStop) );
}
