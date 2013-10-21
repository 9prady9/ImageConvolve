/********************************************************************************
** Form generated from reading UI file 'kernel.ui'
**
** Created by: Qt User Interface Compiler version 4.8.5
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_KERNEL_H
#define UI_KERNEL_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QGridLayout>
#include <QtGui/QHBoxLayout>
#include <QtGui/QHeaderView>
#include <QtGui/QLabel>
#include <QtGui/QLineEdit>
#include <QtGui/QPushButton>
#include <QtGui/QSpacerItem>
#include <QtGui/QTableWidget>
#include <QtGui/QVBoxLayout>
#include <QtGui/QWidget>

QT_BEGIN_NAMESPACE

class Ui_Form
{
public:
    QWidget *layoutWidget;
    QVBoxLayout *verticalLayout_2;
    QVBoxLayout *verticalLayout;
    QTableWidget *kernelTableWidget;
    QGridLayout *gridLayout;
    QLineEdit *kernelLineEdit;
    QPushButton *increaseKernelSizeButton;
    QSpacerItem *horizontalSpacer;
    QSpacerItem *horizontalSpacer_2;
    QPushButton *decreaseKernelSizeButton;
    QLabel *kernelSizeLabel;
    QHBoxLayout *horizontalLayout;
    QSpacerItem *horizontalSpacer_3;
    QPushButton *storeKernelButton;
    QSpacerItem *horizontalSpacer_4;

    void setupUi(QWidget *Form)
    {
        if (Form->objectName().isEmpty())
            Form->setObjectName(QString::fromUtf8("Form"));
        Form->setWindowModality(Qt::ApplicationModal);
        Form->resize(281, 368);
        Form->setMinimumSize(QSize(281, 368));
        Form->setMaximumSize(QSize(281, 368));
        layoutWidget = new QWidget(Form);
        layoutWidget->setObjectName(QString::fromUtf8("layoutWidget"));
        layoutWidget->setGeometry(QRect(20, 20, 240, 327));
        verticalLayout_2 = new QVBoxLayout(layoutWidget);
        verticalLayout_2->setObjectName(QString::fromUtf8("verticalLayout_2"));
        verticalLayout_2->setContentsMargins(0, 0, 0, 0);
        verticalLayout = new QVBoxLayout();
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        kernelTableWidget = new QTableWidget(layoutWidget);
        if (kernelTableWidget->columnCount() < 3)
            kernelTableWidget->setColumnCount(3);
        if (kernelTableWidget->rowCount() < 3)
            kernelTableWidget->setRowCount(3);
        QTableWidgetItem *__qtablewidgetitem = new QTableWidgetItem();
        kernelTableWidget->setItem(0, 0, __qtablewidgetitem);
        QTableWidgetItem *__qtablewidgetitem1 = new QTableWidgetItem();
        kernelTableWidget->setItem(0, 1, __qtablewidgetitem1);
        QTableWidgetItem *__qtablewidgetitem2 = new QTableWidgetItem();
        kernelTableWidget->setItem(0, 2, __qtablewidgetitem2);
        QTableWidgetItem *__qtablewidgetitem3 = new QTableWidgetItem();
        kernelTableWidget->setItem(1, 0, __qtablewidgetitem3);
        QTableWidgetItem *__qtablewidgetitem4 = new QTableWidgetItem();
        kernelTableWidget->setItem(1, 1, __qtablewidgetitem4);
        QTableWidgetItem *__qtablewidgetitem5 = new QTableWidgetItem();
        kernelTableWidget->setItem(1, 2, __qtablewidgetitem5);
        QTableWidgetItem *__qtablewidgetitem6 = new QTableWidgetItem();
        kernelTableWidget->setItem(2, 0, __qtablewidgetitem6);
        QTableWidgetItem *__qtablewidgetitem7 = new QTableWidgetItem();
        kernelTableWidget->setItem(2, 1, __qtablewidgetitem7);
        QTableWidgetItem *__qtablewidgetitem8 = new QTableWidgetItem();
        kernelTableWidget->setItem(2, 2, __qtablewidgetitem8);
        kernelTableWidget->setObjectName(QString::fromUtf8("kernelTableWidget"));
        QSizePolicy sizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(kernelTableWidget->sizePolicy().hasHeightForWidth());
        kernelTableWidget->setSizePolicy(sizePolicy);
        kernelTableWidget->setMinimumSize(QSize(236, 236));
        kernelTableWidget->setMaximumSize(QSize(236, 236));
        kernelTableWidget->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
        kernelTableWidget->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
        kernelTableWidget->setAutoScroll(false);
        kernelTableWidget->setAlternatingRowColors(true);
        kernelTableWidget->setShowGrid(true);
        kernelTableWidget->setCornerButtonEnabled(true);
        kernelTableWidget->setRowCount(3);
        kernelTableWidget->setColumnCount(3);
        kernelTableWidget->horizontalHeader()->setVisible(false);
        kernelTableWidget->horizontalHeader()->setDefaultSectionSize(26);
        kernelTableWidget->horizontalHeader()->setHighlightSections(false);
        kernelTableWidget->horizontalHeader()->setMinimumSectionSize(26);
        kernelTableWidget->verticalHeader()->setVisible(false);
        kernelTableWidget->verticalHeader()->setDefaultSectionSize(26);
        kernelTableWidget->verticalHeader()->setHighlightSections(false);
        kernelTableWidget->verticalHeader()->setMinimumSectionSize(26);

        verticalLayout->addWidget(kernelTableWidget);

        gridLayout = new QGridLayout();
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        kernelLineEdit = new QLineEdit(layoutWidget);
        kernelLineEdit->setObjectName(QString::fromUtf8("kernelLineEdit"));
        kernelLineEdit->setMinimumSize(QSize(31, 21));
        kernelLineEdit->setMaximumSize(QSize(31, 21));
        kernelLineEdit->setReadOnly(true);

        gridLayout->addWidget(kernelLineEdit, 1, 2, 1, 1);

        increaseKernelSizeButton = new QPushButton(layoutWidget);
        increaseKernelSizeButton->setObjectName(QString::fromUtf8("increaseKernelSizeButton"));
        increaseKernelSizeButton->setMinimumSize(QSize(21, 21));
        increaseKernelSizeButton->setMaximumSize(QSize(21, 21));

        gridLayout->addWidget(increaseKernelSizeButton, 1, 3, 1, 1);

        horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        gridLayout->addItem(horizontalSpacer, 0, 0, 1, 1);

        horizontalSpacer_2 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        gridLayout->addItem(horizontalSpacer_2, 0, 4, 1, 1);

        decreaseKernelSizeButton = new QPushButton(layoutWidget);
        decreaseKernelSizeButton->setObjectName(QString::fromUtf8("decreaseKernelSizeButton"));
        decreaseKernelSizeButton->setMinimumSize(QSize(21, 21));
        decreaseKernelSizeButton->setMaximumSize(QSize(21, 21));

        gridLayout->addWidget(decreaseKernelSizeButton, 1, 1, 1, 1);

        kernelSizeLabel = new QLabel(layoutWidget);
        kernelSizeLabel->setObjectName(QString::fromUtf8("kernelSizeLabel"));
        kernelSizeLabel->setMinimumSize(QSize(91, 21));
        kernelSizeLabel->setMaximumSize(QSize(91, 21));

        gridLayout->addWidget(kernelSizeLabel, 0, 1, 1, 3);


        verticalLayout->addLayout(gridLayout);


        verticalLayout_2->addLayout(verticalLayout);

        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        horizontalSpacer_3 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout->addItem(horizontalSpacer_3);

        storeKernelButton = new QPushButton(layoutWidget);
        storeKernelButton->setObjectName(QString::fromUtf8("storeKernelButton"));
        storeKernelButton->setMinimumSize(QSize(81, 23));
        storeKernelButton->setMaximumSize(QSize(81, 23));

        horizontalLayout->addWidget(storeKernelButton);

        horizontalSpacer_4 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout->addItem(horizontalSpacer_4);


        verticalLayout_2->addLayout(horizontalLayout);


        retranslateUi(Form);

        QMetaObject::connectSlotsByName(Form);
    } // setupUi

    void retranslateUi(QWidget *Form)
    {
        Form->setWindowTitle(QApplication::translate("Form", "Form", 0, QApplication::UnicodeUTF8));

        const bool __sortingEnabled = kernelTableWidget->isSortingEnabled();
        kernelTableWidget->setSortingEnabled(false);
        QTableWidgetItem *___qtablewidgetitem = kernelTableWidget->item(0, 0);
        ___qtablewidgetitem->setText(QApplication::translate("Form", "-1", 0, QApplication::UnicodeUTF8));
        QTableWidgetItem *___qtablewidgetitem1 = kernelTableWidget->item(0, 1);
        ___qtablewidgetitem1->setText(QApplication::translate("Form", "-1", 0, QApplication::UnicodeUTF8));
        QTableWidgetItem *___qtablewidgetitem2 = kernelTableWidget->item(0, 2);
        ___qtablewidgetitem2->setText(QApplication::translate("Form", "-1", 0, QApplication::UnicodeUTF8));
        QTableWidgetItem *___qtablewidgetitem3 = kernelTableWidget->item(1, 0);
        ___qtablewidgetitem3->setText(QApplication::translate("Form", "-1", 0, QApplication::UnicodeUTF8));
        QTableWidgetItem *___qtablewidgetitem4 = kernelTableWidget->item(1, 1);
        ___qtablewidgetitem4->setText(QApplication::translate("Form", "8", 0, QApplication::UnicodeUTF8));
        QTableWidgetItem *___qtablewidgetitem5 = kernelTableWidget->item(1, 2);
        ___qtablewidgetitem5->setText(QApplication::translate("Form", "-1", 0, QApplication::UnicodeUTF8));
        QTableWidgetItem *___qtablewidgetitem6 = kernelTableWidget->item(2, 0);
        ___qtablewidgetitem6->setText(QApplication::translate("Form", "-1", 0, QApplication::UnicodeUTF8));
        QTableWidgetItem *___qtablewidgetitem7 = kernelTableWidget->item(2, 1);
        ___qtablewidgetitem7->setText(QApplication::translate("Form", "-1", 0, QApplication::UnicodeUTF8));
        QTableWidgetItem *___qtablewidgetitem8 = kernelTableWidget->item(2, 2);
        ___qtablewidgetitem8->setText(QApplication::translate("Form", "-1", 0, QApplication::UnicodeUTF8));
        kernelTableWidget->setSortingEnabled(__sortingEnabled);

        kernelLineEdit->setText(QApplication::translate("Form", "1", 0, QApplication::UnicodeUTF8));
        increaseKernelSizeButton->setText(QApplication::translate("Form", "+", 0, QApplication::UnicodeUTF8));
        decreaseKernelSizeButton->setText(QApplication::translate("Form", "-", 0, QApplication::UnicodeUTF8));
        kernelSizeLabel->setText(QApplication::translate("Form", "      Kernel Size", 0, QApplication::UnicodeUTF8));
        storeKernelButton->setText(QApplication::translate("Form", "Save kernel", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class Form: public Ui_Form {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_KERNEL_H
