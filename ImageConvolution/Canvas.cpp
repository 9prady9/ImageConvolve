#include <Canvas.h>

Canvas::Canvas(QWidget* parent,Qt::WindowFlags flags)
: QWidget(parent, flags)
{
	this->setFocusPolicy(Qt::StrongFocus);
	mPaintImage = QImage(640,480,QImage::Format_ARGB32);
}

Canvas::~Canvas()
{
}

void Canvas::updateImage(const QImage &fImage)
{
	mPaintImage	= fImage.convertToFormat(QImage::Format_ARGB32_Premultiplied);
	update();
}

void Canvas::paintEvent(QPaintEvent* fEvent)
{	
	QPainter lPainter(this);
	QRect target(0, 0, this->width(), this->height());
	QRect source(0, 0, mPaintImage.width(), mPaintImage.height());
	lPainter.drawImage(target,mPaintImage,source);
	lPainter.end();
}
	