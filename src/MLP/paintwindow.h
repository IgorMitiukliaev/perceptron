#ifndef SRC_MLP_PAINTWINDOW_H_
#define SRC_MLP_PAINTWINDOW_H_

#include <QDialog>
#include <QImage>
#include <QMouseEvent>
#include <QPaintEvent>
#include <QPainter>
#include <vector>

namespace Ui {
class PaintWindow;
}

class PaintWindow : public QDialog {
    Q_OBJECT

 public:
    explicit PaintWindow(QWidget *parent = nullptr);
    ~PaintWindow();

    QImage &GetImage();

 protected:
    void paintEvent(QPaintEvent *p);
    void mouseMoveEvent(QMouseEvent *event);
    void mousePressEvent(QMouseEvent *event);
    void mouseReleaseEvent(QMouseEvent *event);

 private:
    Ui::PaintWindow *ui;
    QPoint last_point_;
    QImage image_;
    bool is_left_button_pressed_ = false;
    const int pen_width_ = 70;
};

#endif  // SRC_MLP_PAINTWINDOW_H_
