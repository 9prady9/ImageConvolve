#ifndef CANVAS_H
#define CANVAS_H

#include <QImage>
#include <QPainter>
#include <QWidget>

class Canvas: public QWidget
{
	Q_OBJECT
public:
	Canvas(QWidget * parent = 0, Qt::WindowFlags flags = 0);
	~Canvas();
	void updateImage(const QImage &);

public slots:
	void saveImage(void);

protected:
	void paintEvent(QPaintEvent* fEvent);

private:
	QImage		mPaintImage;
};

#endif //CANVAS_H
