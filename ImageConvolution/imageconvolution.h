#ifndef IMAGECONVOLUTION_H
#define IMAGECONVOLUTION_H

#include <QtGui/QMainWindow>
#include "ui_imageconvolution.h"
#include "ui_kernel.h"
#include <qscrollarea.h>
#include <qlabel.h>
#include "Convolve.cuh"

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
	void sendImage();
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
	QImage*			mImageHandle;
	QLabel*			mImageViewer;
	QScrollArea*	mScrollArea;
	unsigned int	mKernelRadius;
	bool			mHaveKernel;
	Kernel			mKernel;
	Convolve		mConvolver;
};

#endif // IMAGECONVOLUTION_H
