#include "imageconvolution.h"
#include <qboxlayout.h>
#include <qfiledialog.h>
#include <qpixmap.h>
#include <qmessagebox.h>
#include <qdebug.h>

ImageConvolution::ImageConvolution(QWidget *parent, Qt::WFlags flags)
	: QMainWindow(parent, flags)
{
	ui.setupUi(this);
	mHaveKernel		= false;
	mKernelRadius	= 1;
	mKernel.setKernelRadius(mKernelRadius);
	kernelFormWidget= new QWidget;
	kernelUi.setupUi(kernelFormWidget);
	setWindowTitle(tr("Image Convolution Viewer"));
	setMinimumSize(160,120);
	mImageViewer	= new QLabel;	
	mScrollArea		= new QScrollArea;
    mImageViewer->setBackgroundRole(QPalette::Base);
    mImageViewer->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
    mImageViewer->setScaledContents(true);
    mScrollArea->setBackgroundRole(QPalette::Dark);
    mScrollArea->setWidget(mImageViewer);
    setCentralWidget(mScrollArea);
	connect( ui.actionOpen, SIGNAL(triggered()),this,SLOT(loadImage()) );
	connect( ui.actionSet_Kernel, SIGNAL(triggered()), kernelFormWidget, SLOT(show()) );
	connect( kernelUi.increaseKernelSizeButton, SIGNAL(clicked()), this, SLOT(increaseKernelSize()) );
	connect( kernelUi.decreaseKernelSizeButton, SIGNAL(clicked()), this, SLOT(decreaseKernelSize()) );
	connect( kernelUi.storeKernelButton, SIGNAL(clicked()), this, SLOT(saveKernel()) );
	connect( kernelUi.storeKernelButton, SIGNAL(clicked()), kernelFormWidget, SLOT(hide()) );
	connect( ui.actionApply_Kernel, SIGNAL(triggered()), this, SLOT(applyKernel()) );
}

ImageConvolution::~ImageConvolution() { }

void ImageConvolution::loadImage()
{
	QString fileName = QFileDialog::getOpenFileName(this,tr("Open Image"),"",tr("*.png *.jpg *.bmp"));
	if(!fileName.isEmpty())
	{
		QImage mImageHandle(fileName);
        if (mImageHandle.isNull()) {
			QMessageBox::information(this, tr("Image Viewer"), tr("Cannot load %1.").arg(fileName));
            return;
        }
		mImageWidth = mImageHandle.width();
		mImageHeight= mImageHandle.height();
		mImageViewer->setPixmap(QPixmap::fromImage(mImageHandle));
		mImageViewer->adjustSize();
		update();
		QImage ARGB_Img = mImageHandle.convertToFormat(QImage::Format_ARGB32);
		mConvolver.setImageData(ARGB_Img.bits(),ARGB_Img.width(),ARGB_Img.height());
	}
}

void ImageConvolution::increaseKernelSize()
{	
	mKernelRadius++;
	if(mKernelRadius <= KERNEL_MAX_SIZE) {
		kernelUi.kernelLineEdit->setText(tr("%1").arg(mKernelRadius));
		kernelUi.kernelTableWidget->setRowCount(2*mKernelRadius+1);
		kernelUi.kernelTableWidget->setColumnCount(2*mKernelRadius+1);
		mKernel.setKernelRadius(mKernelRadius);
	} else {
		mKernelRadius--;
		QMessageBox::information(this, tr("Image Viewer"), tr("Kernel radius cannot exceed %1.").arg(KERNEL_MAX_SIZE));
	}
}

void ImageConvolution::decreaseKernelSize()
{
	mKernelRadius--;
	if(mKernelRadius >= KERNEL_MIN_SIZE) {
		kernelUi.kernelLineEdit->setText(tr("%1").arg(mKernelRadius));
		kernelUi.kernelTableWidget->setRowCount(2*mKernelRadius+1);
		kernelUi.kernelTableWidget->setColumnCount(2*mKernelRadius+1);
		mKernel.setKernelRadius(mKernelRadius);
	} else {
		mKernelRadius++;
		QMessageBox::information(this, tr("Image Viewer"), tr("Kernel radius cannot fall below %1.").arg(KERNEL_MIN_SIZE));
	}
}

void ImageConvolution::saveKernel()
{
	if(validateKernel())
	{
		/* The following double loop validates the kernel values */
		int dim = 2*mKernelRadius+1;
		for(int i=0; i<dim; ++i)
		{
			for(int j=0; j<dim; ++j)
			{
				QTableWidgetItem* item = kernelUi.kernelTableWidget->item(i,j);
				bool isConvSuccess = true;
				QString cellValue;
				if(item) {
					cellValue = item->text();
					cellValue.toInt(&isConvSuccess);
					if(!isConvSuccess) {
						QMessageBox::information(this, tr("Image Viewer"), tr("Kernel value at (%1,%2) is not numeral").arg(i+1).arg(j+1));
						return;
					}
				} else {
					QMessageBox::information(this, tr("Image Viewer"), tr("Kernel value at (%1,%2) not set").arg(i+1).arg(j+1));
					return;
				}
			}
		}
		/* Now store the kernel into a buffer for CUDA computation */
		for(int i=0; i<dim; ++i)
		{
			for(int j=0; j<dim; ++j)
			{
				QTableWidgetItem* item = kernelUi.kernelTableWidget->item(i,j);
				bool isConvSuccess = true;
				QString cellValue = item->text();
				mKernel.setCellValue(i,j,cellValue.toInt(&isConvSuccess));
			}
		}
		mConvolver.setKernelData(mKernel);
		mHaveKernel = true;
	}
}

void ImageConvolution::applyKernel()
{
	if(mHaveKernel) {
		mConvolver.cudaConvolve();
		QImage output(mConvolver.getConvolvedImage(),mImageWidth,mImageHeight, QImage::Format_ARGB32);
		mImageViewer->setPixmap(QPixmap::fromImage(output));
		mImageViewer->adjustSize();
		update();
	}else {
		QMessageBox::information(this, tr("Image Viewer"), tr("Kernel not set"));
	}
}

bool ImageConvolution::validateKernel()
{
	return true;
}

