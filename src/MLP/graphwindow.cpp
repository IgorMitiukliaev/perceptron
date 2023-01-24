#include "graphwindow.h"

#include "ui_graphwindow.h"

GraphWindow::GraphWindow(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::GraphWindow),
    image_(512, 512, QImage::Format_RGB16) {
    ui->setupUi(this);
    this->setWindowTitle("Graph");
    image_.fill(QColor(255, 255, 255));
    update();
}

void GraphWindow::paintEvent(QPaintEvent *) {
    QPainter painter(this);
    painter.drawImage(QPoint(0, 0), image_);
}

auto GraphWindow::DrawGraph(const std::vector<double> &values) -> void {
    QString procentage[] = {"90%", "80%", "70%", "60%", "50%", "40%", "30%", "20%", "10%", "0%"};
    QPainter painter(&image_);
    painter.setPen(QPen(Qt::black, pen_width_for_axis_, Qt::SolidLine));

    for (int i = 1; i < 10; ++i) {
        painter.drawLine(0, height_ * 0.1 * i, width_, height_ * 0.1 * i);
        painter.drawText(2, height_ * 0.1 * i - 5, procentage[i - 1]);
    }
    painter.setPen(QPen(Qt::black, pen_width_for_line_, Qt::SolidLine));
    QPoint oldPoint(0, height_ * (1.0 - values[0]));
    const size_t vectorSize = values.size();
    const int dx = width_ / (vectorSize-1);
    for (int i = 1; i < vectorSize; ++i) {
        QPoint newPoint(dx * (i + 1), height_ * (1.0 - values[i]));
        painter.drawLine(oldPoint.rx(), oldPoint.ry(), newPoint.rx(), newPoint.ry());
        oldPoint = newPoint;
    }
    update();
}

GraphWindow::~GraphWindow() {
    delete ui;
}
