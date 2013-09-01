#include "Convolve.cuh"
#include <malloc.h>
#include <qimage.h>
#include <qdebug.h>

__constant__	int	dev_ConstMemory_Kernel[81];

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

__global__ void convolveKernel(const uchar* fSource, int fImageWidth, int fImageHeight, uchar* fDestination, int fKernelSize)
{
	extern __shared__ uchar canvasCache[];
	
	int gx = threadIdx.x + blockDim.x * blockIdx.x;
	int gy = threadIdx.y + blockDim.y * blockIdx.y;

	if( gx >= fImageWidth || gy >= fImageHeight )
		return;

	int linearIdx	= gy*fImageWidth + gx;
	int ridx		= linearIdx+2;
	int gidx		= linearIdx+1;
	int bidx		= linearIdx;
	int aidx		= linearIdx+3;
	fDestination[ridx] = 0; //fSource[ridx];
	fDestination[gidx] = 0; //fSource[gidx];
	fDestination[bidx] = gx; //fSource[bidx];
	fDestination[aidx] = gy; //fSource[aidx];
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
	HANDLE_ERROR( cudaMemcpyToSymbol(dev_ConstMemory_Kernel, fKernel.krValues, temp*temp*sizeof(int)) );
	mKernelSize			= fKernel.kr;
	mIsKernelSet		= true;
}

void Convolve::setImageData(const uchar* fImageData, int fImageWidth, int fImageHeight)
{	
	free(host_ConvolvedImage);
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
		int sharedMemSize = ( mThreadsPerBlock.x*mThreadsPerBlock.y + 2*mKernelSize )*4*sizeof(uchar);
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
		output.save("C:\\Users\\prady\\Downloads\\output.png");
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
	/*cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop,0);
	int MAX_SHARED_MEM = prop.sharedMemPerBlock/1024;
	int PIXEL_LOAD_PER_DIM;		
	for(PIXEL_LOAD_PER_DIM=7; ;++PIXEL_LOAD_PER_DIM)
	{
	int currentSize = (PIXEL_LOAD_PER_DIM*mThreadsPerBlock.x+2*mKernelSize)/16;
	if(currentSize*currentSize > MAX_SHARED_MEM)
	{
	PIXEL_LOAD_PER_DIM--;
	mPerThreadLoad.x = PIXEL_LOAD_PER_DIM;
	mPerThreadLoad.y = PIXEL_LOAD_PER_DIM;
	break;
	}
	}*/
	mThreadsPerBlock.x = TILE_WIDTH;
	mThreadsPerBlock.y = TILE_HEIGHT;
	mGrid.x = ceil( mImageWidth, mThreadsPerBlock.x );
	mGrid.y = ceil( mImageHeight, mThreadsPerBlock.y );

	qDebug()<<"Threads per block "<<mThreadsPerBlock.x<<","<<mThreadsPerBlock.y;
	//qDebug()<<"Load per Thread "<<mPerThreadLoad.x<<","<<mPerThreadLoad.y;
	qDebug()<<"Blocks per grid "<<mGrid.x<<","<<mGrid.y;
	//qDebug()<<"Shared memory usage : "<<((mPerThreadLoad.x*mThreadsPerBlock.x+2*mKernelSize)*(mPerThreadLoad.x*mThreadsPerBlock.x+2*mKernelSize))/256<<" KB";
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
	cleanMemory();
}
