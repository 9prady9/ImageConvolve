/********************************************************************************
** Form generated from reading UI file 'imageconvolution.ui'
**
** Created: Sat Aug 31 20:28:40 2013
**      by: Qt User Interface Compiler version 4.8.4
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_IMAGECONVOLUTION_H
#define UI_IMAGECONVOLUTION_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QHeaderView>
#include <QtGui/QMainWindow>
#include <QtGui/QMenu>
#include <QtGui/QMenuBar>
#include <QtGui/QStatusBar>
#include <QtGui/QToolBar>
#include <QtGui/QWidget>

QT_BEGIN_NAMESPACE

class Ui_ImageConvolutionClass
{
public:
    QAction *actionOpen;
    QAction *actionExit;
    QAction *actionSave;
    QAction *actionSet_Kernel;
    QAction *actionApply_Kernel;
    QAction *actionSend_Image_Data;
    QWidget *centralWidget;
    QMenuBar *menuBar;
    QMenu *menuFile;
    QMenu *menuConvolve;
    QToolBar *mainToolBar;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *ImageConvolutionClass)
    {
        if (ImageConvolutionClass->objectName().isEmpty())
            ImageConvolutionClass->setObjectName(QString::fromUtf8("ImageConvolutionClass"));
        ImageConvolutionClass->resize(600, 400);
        actionOpen = new QAction(ImageConvolutionClass);
        actionOpen->setObjectName(QString::fromUtf8("actionOpen"));
        actionExit = new QAction(ImageConvolutionClass);
        actionExit->setObjectName(QString::fromUtf8("actionExit"));
        actionSave = new QAction(ImageConvolutionClass);
        actionSave->setObjectName(QString::fromUtf8("actionSave"));
        actionSet_Kernel = new QAction(ImageConvolutionClass);
        actionSet_Kernel->setObjectName(QString::fromUtf8("actionSet_Kernel"));
        actionApply_Kernel = new QAction(ImageConvolutionClass);
        actionApply_Kernel->setObjectName(QString::fromUtf8("actionApply_Kernel"));
        actionSend_Image_Data = new QAction(ImageConvolutionClass);
        actionSend_Image_Data->setObjectName(QString::fromUtf8("actionSend_Image_Data"));
        centralWidget = new QWidget(ImageConvolutionClass);
        centralWidget->setObjectName(QString::fromUtf8("centralWidget"));
        ImageConvolutionClass->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(ImageConvolutionClass);
        menuBar->setObjectName(QString::fromUtf8("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 600, 21));
        menuFile = new QMenu(menuBar);
        menuFile->setObjectName(QString::fromUtf8("menuFile"));
        menuConvolve = new QMenu(menuBar);
        menuConvolve->setObjectName(QString::fromUtf8("menuConvolve"));
        ImageConvolutionClass->setMenuBar(menuBar);
        mainToolBar = new QToolBar(ImageConvolutionClass);
        mainToolBar->setObjectName(QString::fromUtf8("mainToolBar"));
        ImageConvolutionClass->addToolBar(Qt::TopToolBarArea, mainToolBar);
        statusBar = new QStatusBar(ImageConvolutionClass);
        statusBar->setObjectName(QString::fromUtf8("statusBar"));
        ImageConvolutionClass->setStatusBar(statusBar);

        menuBar->addAction(menuFile->menuAction());
        menuBar->addAction(menuConvolve->menuAction());
        menuFile->addAction(actionOpen);
        menuFile->addAction(actionSave);
        menuFile->addAction(actionExit);
        menuConvolve->addAction(actionSet_Kernel);
        menuConvolve->addAction(actionApply_Kernel);

        retranslateUi(ImageConvolutionClass);
        QObject::connect(actionExit, SIGNAL(triggered()), ImageConvolutionClass, SLOT(close()));

        QMetaObject::connectSlotsByName(ImageConvolutionClass);
    } // setupUi

    void retranslateUi(QMainWindow *ImageConvolutionClass)
    {
        ImageConvolutionClass->setWindowTitle(QApplication::translate("ImageConvolutionClass", "ImageConvolution", 0, QApplication::UnicodeUTF8));
        actionOpen->setText(QApplication::translate("ImageConvolutionClass", "Open", 0, QApplication::UnicodeUTF8));
        actionExit->setText(QApplication::translate("ImageConvolutionClass", "Exit", 0, QApplication::UnicodeUTF8));
        actionSave->setText(QApplication::translate("ImageConvolutionClass", "Save", 0, QApplication::UnicodeUTF8));
        actionSet_Kernel->setText(QApplication::translate("ImageConvolutionClass", "Set Kernel", 0, QApplication::UnicodeUTF8));
        actionApply_Kernel->setText(QApplication::translate("ImageConvolutionClass", "Apply Kernel", 0, QApplication::UnicodeUTF8));
        actionSend_Image_Data->setText(QApplication::translate("ImageConvolutionClass", "Pad Image", 0, QApplication::UnicodeUTF8));
        menuFile->setTitle(QApplication::translate("ImageConvolutionClass", "File", 0, QApplication::UnicodeUTF8));
        menuConvolve->setTitle(QApplication::translate("ImageConvolutionClass", "Convolve", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class ImageConvolutionClass: public Ui_ImageConvolutionClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_IMAGECONVOLUTION_H
