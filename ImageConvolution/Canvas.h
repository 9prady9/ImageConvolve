#ifndef CANVAS_H
#define CANVAS_H

#include <QImage>
#include <QPainter>
#include <QtGui/QWidget>

class Canvas: public QWidget
{
	Q_OBJECT
public:
	Canvas(QWidget * parent = 0, Qt::WindowFlags flags = 0);
	~Canvas();
	void updateImage(const QImage &);
	
protected:
	void paintEvent(QPaintEvent* fEvent);

private:
	QImage		mPaintImage;
};

#endif //CANVAS_H