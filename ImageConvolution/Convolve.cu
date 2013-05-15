#include "Convolve.cuh"
#include <malloc.h>
#include <qimage.h>
#include <qdebug.h>

__constant__	int	dev_ConstMemory_Kernel[81];

static int	mKernelSize;
uchar*		dev_ARGB_Channel;
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

__global__ void convolveKernel(uchar* fCanvas, int fCanvasWidth, int fCanvasHeight, uchar* dev_fConvolvedImage, int fImgWidth, int fImgHeight, int fKernelSize, dim3 fMyLoad )
{
	extern __shared__ uchar canvasCache[];
	
	int length = (fMyLoad.x*blockDim.x+2*fKernelSize);
	int sharedMemSize = length*length*4*sizeof(uchar);

	/* Thread 0 in each block copies canvas data to shared cache */
	if(threadIdx.x==0)
	{
		int blockBeginX = blockIdx.x * blockDim.x * fMyLoad.x;
		int blockBeginY = blockIdx.y * blockDim.y * fMyLoad.y;
		for(int i=0;i<length; ++i)
		{	/* Row by Row */
			for(int j=0;j<length; ++j)
			{	/* Col by Col */
				int index = 4*(j + length*i);
				int row = blockBeginY+i;
				int col = blockBeginX+j;
				if(row<fCanvasHeight && col<fCanvasWidth)
				{
					int originalIdx = 4*(col + fCanvasWidth*row);
					canvasCache[index] = fCanvas[originalIdx];
					canvasCache[index+1] = fCanvas[originalIdx+1];
					canvasCache[index+2] = fCanvas[originalIdx+2];
					canvasCache[index+3] = fCanvas[originalIdx+3];
				} else {
					canvasCache[index] = 0;
					canvasCache[index+1] = 0;
					canvasCache[index+2] = 0;
					canvasCache[index+3] = 255;
				}
			}
		}
	}
	__syncthreads();
	/**
	 * fMyLoad defines the size of 2D set of pixels each thread processes
	 * Carry out convolve operation on this 2d set of pixels using the
	 * the kernel stored in constant memory of GPU device
	 */
	int kernelLength = 2*fKernelSize + 1;
	int numKernelElements = kernelLength*kernelLength;
	for(int i=0;i<fMyLoad.y; ++i)
	{	/* Row by Row */
		for(int j=0;j<fMyLoad.x; ++j)
		{	/* Col by Col */
			/* Compute convole result using canvas cache and kernel in constant memory */
			int B, G, R, A;
			B = G = R = A = 0;
			for(int kerRow=0; kerRow<kernelLength; ++kerRow)
			{
				for(int kerCol=0; kerCol<kernelLength; ++kerCol)
				{
					int kerIdx	= kerCol + kernelLength*kerRow;
					int factor	= dev_ConstMemory_Kernel[kerIdx];

					int cacheX	= i + kerCol;
					int cacheY	= j + kerRow;
					int pixelIdx= 4*(cacheX + length*cacheY); 

					B += factor*canvasCache[pixelIdx];
					G += factor*canvasCache[pixelIdx+1];
					R += factor*canvasCache[pixelIdx+2];
					A += factor*canvasCache[pixelIdx+3];
				}
			}
			B /= numKernelElements;
			G /= numKernelElements;
			R /= numKernelElements;
			A /= numKernelElements;

			/* Store result in dev_fConvolvedImage at location pointed by pixelRow and pixelCol */			
			int pixelCol	= j + threadIdx.x*fMyLoad.x + blockIdx.x*blockDim.x*fMyLoad.x;
			int pixelRow	= i + threadIdx.y*fMyLoad.y + blockIdx.y*blockDim.y*fMyLoad.y;
			if(pixelCol<fImgWidth && pixelRow<fImgHeight)
			{
				int pxIdx		= 4*(pixelCol + fImgWidth*pixelRow);
				dev_fConvolvedImage[pxIdx] = B;
				dev_fConvolvedImage[pxIdx+1] = G;
				dev_fConvolvedImage[pxIdx+2] = R;
				dev_fConvolvedImage[pxIdx+3] = A;
			}
		}
	}
}

Convolve::Convolve()
{
	mThreadsPerBlock.x	= 8;
	mThreadsPerBlock.y	= 8;
	mIsKernelSet		= false;
	mIsImageSet			= false;
	mIsCUDAInit			= false;
	mIsComputeDone		= false;
	host_ARGB_channel	= 0;
	dev_ARGB_Channel	= 0;
}

void Convolve::setKernelData(const Kernel &fKernel)
{
	int temp			= fKernel.kr*2 + 1;
	HANDLE_ERROR( cudaMemcpyToSymbol(dev_ConstMemory_Kernel, fKernel.krValues, temp*temp*sizeof(int)) );
	mKernelSize			= fKernel.kr;
	mIsKernelSet		= true;
}

/**
 * Below method expects image data as a unsigned char array whose size = width*height*4
 * where width and height are dimensions of the image and 4 indicates number of color
 * channels including alpha channel.
 * The image data is expected to be a single dimension array of size width*height with
 * each pixel represented by a tuple of 4 bytes. For example, image with (width,height) = (2,2)
 * { Blue_00, Green_00, Red_00, Alpha_00, Blue_01, Green_01, Red_01, Alpha_01,
 *   Blue_10, Green_10, Red_10, Alpha_10, Blue_11, Green_11, Red_11, Alpha_11 }
 * Thus, Pixel Index = 4*(scanLine_of_Pixel*width + coloumn_of_the_pixel)
 * Once, pixel index is known, all four channels of the pixel can be addressed by
 * simply incrementing the base index by 0, 1, 2, 3
 * 
 * NOTE: The order of channels Red, blue, green and alpha is not mandatory to be RGBA for convolve to work
 */
void Convolve::setImageData(const uchar* fImageData, int fImageWidth, int fImageHeight)
{
	if(mIsKernelSet)
	{
		mImageWidth		= fImageWidth;
		mImageHeight	= fImageHeight;
		mCanvasWidth	= mImageWidth + 2*mKernelSize;
		mCanvasHeight	= mImageHeight + 2*mKernelSize;
		host_ARGB_channel = (uchar*)malloc(mCanvasWidth*mCanvasHeight*4*sizeof(uchar));
		/* Prepare canvas with kernel size padding along border of the image */
		int imgXThresholdOnCanvas = mImageWidth + mKernelSize;
		int imgYThresholdOnCanvas = mImageHeight + mKernelSize;
		for(int row=0; row<mCanvasHeight; ++row)		
		{
			for(int col=0; col<mCanvasWidth; ++col)	
			{
				int index = 4*(col + mCanvasWidth*row);
				if( (col>=mKernelSize && row>=mKernelSize) &&
					(col<imgXThresholdOnCanvas && row<imgYThresholdOnCanvas) )
				{
					int originalIdx = 4*((col-mKernelSize) + mImageWidth*(row-mKernelSize));
					host_ARGB_channel[index+0] = fImageData[originalIdx+0];
					host_ARGB_channel[index+1] = fImageData[originalIdx+1];
					host_ARGB_channel[index+2] = fImageData[originalIdx+2];
					host_ARGB_channel[index+3] = fImageData[originalIdx+3];
				} else {
					/**
					 * Copy pixel value along the border to the additional padding
					 * pixels along  the line perpendicular to the border edge
					 */
					if(col<mKernelSize && row<mKernelSize) {
						host_ARGB_channel[index+0] = fImageData[0];
						host_ARGB_channel[index+1] = fImageData[1];
						host_ARGB_channel[index+2] = fImageData[2];
						host_ARGB_channel[index+3] = fImageData[3];
					} else if(col>=imgXThresholdOnCanvas && row>=imgYThresholdOnCanvas) {
						int originalIdx = 4*(mImageWidth*mImageHeight-1);
						host_ARGB_channel[index+0] = fImageData[originalIdx+0];
						host_ARGB_channel[index+1] = fImageData[originalIdx+1];
						host_ARGB_channel[index+2] = fImageData[originalIdx+2];
						host_ARGB_channel[index+3] = fImageData[originalIdx+3];
					} else if(row<mKernelSize && col>=imgXThresholdOnCanvas) {
						int originalIdx = 4*(mImageWidth-1);
						host_ARGB_channel[index+0] = fImageData[originalIdx+0];
						host_ARGB_channel[index+1] = fImageData[originalIdx+1];
						host_ARGB_channel[index+2] = fImageData[originalIdx+2];
						host_ARGB_channel[index+3] = fImageData[originalIdx+3];
					} else if(col<mKernelSize && row>=imgYThresholdOnCanvas) {
						int originalIdx = 4*(mImageWidth*(mImageHeight-1));
						host_ARGB_channel[index+0] = fImageData[originalIdx+0];
						host_ARGB_channel[index+1] = fImageData[originalIdx+1];
						host_ARGB_channel[index+2] = fImageData[originalIdx+2];
						host_ARGB_channel[index+3] = fImageData[originalIdx+3];
					} else if(col<mKernelSize && (row>=mKernelSize && row<imgYThresholdOnCanvas)) {
						int originalIdx = 4*(mImageWidth*(row-mKernelSize));
						host_ARGB_channel[index+0] = fImageData[originalIdx+0];
						host_ARGB_channel[index+1] = fImageData[originalIdx+1];
						host_ARGB_channel[index+2] = fImageData[originalIdx+2];
						host_ARGB_channel[index+3] = fImageData[originalIdx+3];
					} else if(col>=imgXThresholdOnCanvas && (row>=mKernelSize && row<imgYThresholdOnCanvas)) {
						int originalIdx = 4*(mImageWidth*((row-mKernelSize)+1)-1);
						host_ARGB_channel[index+0] = fImageData[originalIdx+0];
						host_ARGB_channel[index+1] = fImageData[originalIdx+1];
						host_ARGB_channel[index+2] = fImageData[originalIdx+2];
						host_ARGB_channel[index+3] = fImageData[originalIdx+3];
					} else if(row<mKernelSize && (col>=mKernelSize && col<imgXThresholdOnCanvas)) {
						int originalIdx = 4*(col-mKernelSize);
						host_ARGB_channel[index+0] = fImageData[originalIdx+0];
						host_ARGB_channel[index+1] = fImageData[originalIdx+1];
						host_ARGB_channel[index+2] = fImageData[originalIdx+2];
						host_ARGB_channel[index+3] = fImageData[originalIdx+3];
					} else {
						int originalIdx = 4*((col-mKernelSize) + mImageWidth*(mImageHeight-1));
						host_ARGB_channel[index+0] = fImageData[originalIdx+0];
						host_ARGB_channel[index+1] = fImageData[originalIdx+1];
						host_ARGB_channel[index+2] = fImageData[originalIdx+2];
						host_ARGB_channel[index+3] = fImageData[originalIdx+3];
					}
				}
			}
		}
		/* Copy this data to device memory for kernel computation */
		HANDLE_ERROR( cudaMalloc((void**)&dev_ARGB_Channel,mCanvasWidth*mCanvasHeight*4*sizeof(uchar)) );
		HANDLE_ERROR( cudaMemcpy( dev_ARGB_Channel, host_ARGB_channel, mCanvasWidth*mCanvasHeight*4*sizeof(uchar), cudaMemcpyHostToDevice) );
		/* Allocate memory for output image on GPU device memory */
		HANDLE_ERROR( cudaMalloc((void**)&dev_ConvolvedImage, mImageWidth*mImageHeight*4*sizeof(uchar)) );
		mIsImageSet		= true;
		free(host_ARGB_channel);
		host_ARGB_channel = 0;
		initCUDA();
	} else
		printf("Prerequisite: Kernel data not set. \n");
}

uchar* Convolve::getConvolvedImage()
{
	return host_ConvolvedImage;
}

void Convolve::initCUDA()
{
	if(mIsKernelSet && mIsImageSet)
	{
		HANDLE_ERROR( cudaEventCreate( &mStart ) );
		HANDLE_ERROR( cudaEventCreate( &mStop ) );
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop,0);
		int MAX_SHARED_MEM = prop.sharedMemPerBlock/1024;
		int PIXEL_LOAD_PER_DIM;
		/**
		 * Figure out feasible pixel load along one dimension
		 * based on how much shared memory can be allocted at
		 * the most. Use same value along the second dimension
		 * of image to have square block of pixels
		 */
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
		}
		mGrid.x = ceil( mImageWidth, mThreadsPerBlock.x*mPerThreadLoad.x );
		mGrid.y = ceil( mImageHeight, mThreadsPerBlock.y*mPerThreadLoad.y );

		qDebug()<<"Threads per block "<<mThreadsPerBlock.x<<","<<mThreadsPerBlock.y;
		qDebug()<<"Load per Thread "<<mPerThreadLoad.x<<","<<mPerThreadLoad.y;
		qDebug()<<"Blocks per grid "<<mGrid.x<<","<<mGrid.y;
		qDebug()<<"Shared memory usage : "<<((mPerThreadLoad.x*mThreadsPerBlock.x+2*mKernelSize)*(mPerThreadLoad.x*mThreadsPerBlock.x+2*mKernelSize))/256<<" KB";

		mIsCUDAInit= true;
	} else
		printf("Prerequisite: Image data not set. \n");
}

void Convolve::cudaConvolve()
{	
	if(mIsKernelSet && mIsImageSet && mIsCUDAInit)
	{
		int temp = (mPerThreadLoad.x*mThreadsPerBlock.x+2*mKernelSize);
		int sharedMemSize = temp*temp*4*sizeof(uchar);
		cudaEventRecord(mStart,0);
		convolveKernel<<<mGrid,mThreadsPerBlock,sharedMemSize>>>( dev_ARGB_Channel, mCanvasWidth, mCanvasHeight, dev_ConvolvedImage, mImageWidth, mImageHeight, mKernelSize, mPerThreadLoad );
		/* Sync device */
		cudaDeviceSynchronize();
		cudaEventRecord(mStop,0);
		float computeTime = 0.0f;
		cudaEventSynchronize(mStop);
		cudaEventElapsedTime(&computeTime,mStart,mStop);
		qDebug()<<"Time (data transfer+computation on device): "<<computeTime<<" ms\n";
		/* Copy back convolved image to host and show it */		
		host_ConvolvedImage = (uchar*)malloc(mImageWidth*mImageHeight*4*sizeof(uchar));
		HANDLE_ERROR( cudaMemcpy( host_ConvolvedImage, dev_ConvolvedImage, mImageWidth*mImageHeight*4*sizeof(uchar), cudaMemcpyDeviceToHost) );
		QImage output(host_ConvolvedImage,mImageWidth,mImageHeight,QImage::Format_ARGB32);
		output.save("C:\\Users\\prady\\Downloads\\output.png");
		mIsComputeDone = true;
	} else
		printf("Prerequisite: CUDA not initialized. \n");
}

void Convolve::destroyCUDA()
{	
	if(mIsComputeDone)
	{
		HANDLE_ERROR( cudaEventDestroy(mStart) );
		HANDLE_ERROR( cudaEventDestroy(mStop) );
		cudaFree(dev_ARGB_Channel);
		free(host_ARGB_channel);
	} else
		printf("Prerequisite: CUDA computation not finished. \n");
}

Convolve::~Convolve()
{
	cudaFree(dev_ARGB_Channel);
	free(host_ARGB_channel);
}