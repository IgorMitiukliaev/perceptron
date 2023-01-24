#ifndef SRC_MLP_MAINWINDOW_H_
#define SRC_MLP_MAINWINDOW_H_
#include <QDebug>
#include <QFileDialog>
#include <QLabel>
#include <QMainWindow>
#include <QPixmap>
#include <QProgressBar>
#include <QRegularExpression>
#include <iostream>

#include "controller.h"
#include "graphwindow.h"
#include "paintwindow.h"

QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}  // namespace Ui
QT_END_NAMESPACE

class MainWindow : public QMainWindow {
  Q_OBJECT

 private:
  Ui::MainWindow *ui;
  PaintWindow *paint_window_;
  GraphWindow *graph_window_;
  s21::Controller *controller_;
  long num_images_ = 0, num_curr_image_ = 0;
  std::vector<double> vector_pixels_;
  QImage graphics_view_image_;
  const int count_neurons_ = 28;

  void DrawPreview(int img_num = 0);
  void UpdatePreviewLabel();
  void UpdateBatchLabel();
  void UpdateMLPState();
  bool EnableButtons();
  void UpdateConfigurationView();
  void UpdateAnswerLabel();
  void CreateVectorPixels(QImage &image);
  void GraphicsViewUpdate(QImage &image);
  QString GetDatasetFileName();
  QPixmap GetPreviewPicture(int img_num);
  void DrawTestPreview(int img_num = 0);
  void UpdateTestPreviewLabel();
  void UpdateTestSheet();
  auto ResearchTestingTime(const int count) -> double;

 public:
  explicit MainWindow(s21::Controller *c, QWidget *parent = nullptr);
  ~MainWindow();

 private slots:
  void on_btnSaveNetworkConfiguration_clicked();
  void on_btnLoadNetworkConfiguration_clicked();
  void on_btnLoadImage_clicked();
  void on_btnLoadDataset_clicked();
  void on_btnImgUp_clicked();
  void on_pushButton_draw_clicked();
  void on_btnProceed_clicked();
  void on_btnInit_clicked();
  void on_btnStartLearn_clicked();
  void on_valEpochNum_valueChanged(int arg1);
  void on_valBatchNum_valueChanged(int arg1);
  void on_tabWidget_tabBarClicked(int index);
  void on_progressChanged_(int value, int value2);
  void on_progressTestChanged_(int value, int value2);
  void on_CreateGraph_clicked();
  void on_btnLoadDatasetTest_clicked();
  void on_btnImgUpTest_clicked();
  void on_btnStartTest_clicked();
  void on_MainWindow_destroyed();
  void on_rbtnGraph_clicked();
  void on_rbtnMatrix_clicked();
  void on_num_layers_hidden_valueChanged(int arg1);
  void on_num_neurons_hidden_valueChanged(int arg1);
  void on_pushButtonResearch_clicked();
};
#endif  // SRC_MLP_MAINWINDOW_H_
