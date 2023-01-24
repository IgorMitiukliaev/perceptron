#ifndef SRC_MLP_GRAPHWINDOW_H_
#define SRC_MLP_GRAPHWINDOW_H_

#include <QDialog>
#include <QImage>
#include <QPaintEvent>
#include <QPainter>
#include <vector>

namespace Ui {
class GraphWindow;
}

class GraphWindow : public QDialog {
  Q_OBJECT

 public:
  explicit GraphWindow(QWidget *parent = nullptr);
  ~GraphWindow();
  auto DrawGraph(const std::vector<double> &values) -> void;

 protected:
  void paintEvent(QPaintEvent *p);

 private:
  Ui::GraphWindow *ui;
  QImage image_;
  const int width_ = 512;
  const int height_ = 512;
  const int pen_width_for_axis_ = 1;
  const int pen_width_for_line_ = 2;
};

#endif  // SRC_MLP_GRAPHWINDOW_H_
