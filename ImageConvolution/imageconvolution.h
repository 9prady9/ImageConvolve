#ifndef IMAGECONVOLUTION_H
#define IMAGECONVOLUTION_H

#include "Canvas.h"
#include "ui_kernel.h"
#include "Convolve.cuh"
#include "ui_imageconvolution.h"

#include <QtGui/QMainWindow>

const unsigned int KERNEL_MAX_SIZE = 4;
const unsigned int KERNEL_MIN_SIZE = 1;

class ImageConvolution : public QMainWindow
{
	Q_OBJECT

public:
	ImageConvolution(QWidget *parent = 0, Qt::WFlags flags = 0);
	~ImageConvolution();

public slots:
	void loadImage();
	void increaseKernelSize();
	void decreaseKernelSize();
	void saveKernel();
	void applyKernel();

private:
	/* Helper Methods */
	bool validateKernel();

	/* Attributes */
	Ui::ImageConvolutionClass ui;
	Ui::Form		kernelUi;

	QWidget*		kernelFormWidget;
	Canvas*			mRenderArea;

	int				mImageWidth;
	int				mImageHeight;
	unsigned int	mKernelRadius;
	bool			mHaveKernel;
	Kernel			mKernel;
	Convolve		mConvolver;
};

#endif // IMAGECONVOLUTION_H
