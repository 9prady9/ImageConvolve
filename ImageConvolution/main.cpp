#include "imageconvolution.h"
#include <qapplication.h>

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);
	ImageConvolution w;
	w.show();
	return a.exec();
}
