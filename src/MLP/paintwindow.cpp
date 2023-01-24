#include "paintwindow.h"

#include "ui_paintwindow.h"

PaintWindow::PaintWindow(QWidget *parent) : QDialog(parent),
                                            ui(new Ui::PaintWindow),
                                            image_(512, 512, QImage::Format_RGB16) {
    ui->setupUi(this);
    this->setMouseTracking(true);
    this->setWindowTitle("Painter");
    image_.fill(QColor(255, 255, 255));
    update();
}

void PaintWindow::paintEvent(QPaintEvent *) {
    QPainter painter(this);
    painter.drawImage(QPoint(0, 0), image_);
}

void PaintWindow::mouseMoveEvent(QMouseEvent *event) {
    if (is_left_button_pressed_) {
        QPainter painter(&image_);
        painter.setPen(QPen(Qt::black, pen_width_, Qt::SolidLine));
        QPoint newPoint = event->pos();
        painter.drawLine(last_point_.x(), last_point_.y(), newPoint.x(), newPoint.y());
        last_point_ = newPoint;
        update();
    }
}

void PaintWindow::mousePressEvent(QMouseEvent *event) {
    if (event->button() == Qt::LeftButton) {
        is_left_button_pressed_ = true;
        last_point_ = event->pos();
    } else if (event->button() == Qt::RightButton) {
        image_.fill(QColor(255, 255, 255));
        update();
    }
}

void PaintWindow::mouseReleaseEvent(QMouseEvent *event) {
    if (event->button() == Qt::LeftButton) {
        is_left_button_pressed_ = false;
    }
}

QImage& PaintWindow::GetImage() {
    return image_;
}

PaintWindow::~PaintWindow() {
    delete ui;
}
