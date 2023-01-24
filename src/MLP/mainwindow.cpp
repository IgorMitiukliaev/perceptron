#include "mainwindow.h"

#include <QDebug>

#include "./ui_mainwindow.h"
#include "paintwindow.h"

using s21::Controller;

MainWindow::MainWindow(s21::Controller *controller, QWidget *parent)
    : controller_(controller),
      QMainWindow(parent),
      ui(new Ui::MainWindow),
      paint_window_(new PaintWindow),
      graph_window_(new GraphWindow) {
  ui->setupUi(this);
  ui->barLearnProgress->setRange(0, 100);
  QObject::connect(qobject_cast<QObject *>(controller_),
                   SIGNAL(progressChanged_(int, int)), this,
                   SLOT(on_progressChanged_(int, int)));
  QObject::connect(qobject_cast<QObject *>(controller_),
                   SIGNAL(progressTestChanged_(int, int)), this,
                   SLOT(on_progressTestChanged_(int, int)));
}

MainWindow::~MainWindow() {
  delete graph_window_;
  delete paint_window_;
  delete ui;
}

void MainWindow::on_btnLoadImage_clicked() {
  QFileDialog dialog(this);
  dialog.setFileMode(QFileDialog::ExistingFile);
  QString fileName = dialog.getOpenFileName(this, tr("Open File"), "/home",
                                            tr("Images (*.bmp)"));
  if (!fileName.isEmpty()) {
    QImage image(fileName);
    GraphicsViewUpdate(image);
  }
}

void MainWindow::on_btnLoadDataset_clicked() {
  QString file_name = GetDatasetFileName();
  if (!file_name.isEmpty()) {
    controller_->LoadDataset(file_name.toStdString());
    num_images_ = controller_->GetCountOfElements();
    num_curr_image_ = 0;
    DrawPreview();
    UpdatePreviewLabel();
    UpdateAnswerLabel();
    UpdateMLPState();
  }
  EnableButtons();
}

void MainWindow::DrawPreview(int img_num) {
  QLabel *wg = (QLabel *)ui->lblPreview;
  QPixmap pixmap = GetPreviewPicture(img_num);
  wg->setPixmap(pixmap);
  wg->show();
}

void MainWindow::UpdatePreviewLabel() {
  QString lbl = " of " + QString::number(num_images_);
  ui->lblTotalImgs->setText(lbl);
  ui->inpNumCurrImg->setText(QString::number(num_curr_image_));
  int num_letter = controller_->GetCorrectValue() + 65;
  ui->lblLetter->setText(QString(QChar::fromLatin1(num_letter)));
  UpdateBatchLabel();
}

void MainWindow::UpdateMLPState() {
  std::vector<double> out = controller_->GetOutValues();
  QString text;
  for (int i{0}; i < ui->gridMLP->rowCount(); ++i) {
    for (int j{0}; j < ui->gridMLP->columnCount(); ++j) {
      auto *label =
          static_cast<QLabel *>(ui->gridMLP->itemAtPosition(i, j)->widget());
      text = QString(QChar::fromLatin1(j * 13 + i + 65));
      text += ": " + QString::number(out[j * 13 + i] * 100, 'f', 2) + "%";
      label->setText(text);
    }
  }
  char buf[100];
  std::strftime(buf, sizeof buf, "%T",
                std::gmtime(&controller_->GetErr().time_lap));
  text = "Total count " + QString::number(controller_->GetErr().count, 'f', 0) +
         "\n";
  text += "Success count " +
          QString::number(controller_->GetErr().count_success, 'f', 0) + "\n";
  text += "Accuracy " +
          QString::number(controller_->GetErr().accuracy * 100, 'f', 2) + "%\n";
  text += "Precision " +
          QString::number(controller_->GetErr().precision * 100, 'f', 2) +
          "%\n";
  text += "Recall " +
          QString::number(controller_->GetErr().recall * 100, 'f', 2) + "%\n";
  text += "f-measure " +
          QString::number(controller_->GetErr().f_measure * 100, 'f', 2) +
          "%\n";
  text += "Time spent ";
  text.append(buf);
  ui->lblError->setText(text);
}

void MainWindow::on_btnImgUp_clicked() {
  for (int i = 0; i < ui->valEpochNum->text().toInt(); i++) {
    for (int j = 0; j < ui->valEpochNum->text().toInt(); j++) {
      num_curr_image_++;
      controller_->LoadNextDataset();
      DrawPreview();
      UpdateAnswerLabel();
      UpdatePreviewLabel();
      UpdateMLPState();
    }
    UpdateMLPState();
  }
}

void MainWindow::on_btnInit_clicked() {
  s21::InitConfig config;
  config.is_graph = ui->rbtnGraph->isChecked();
  config.num_layers_hidden = ui->num_layers_hidden->value();
  config.num_neurons_hidden = ui->num_neurons_hidden->value();
  config.num_neurons_input = pow(ui->num_neurons_input->text().toInt(), 2);
  config.num_neurons_out = ui->num_neurons_out->text().toInt();
  controller_->InitNetwork(config);
  EnableButtons();
}

void MainWindow::on_pushButton_draw_clicked() { paint_window_->show(); }

void MainWindow::GraphicsViewUpdate(QImage &image) {
  if (image.width() <= 512 && image.height() <= 512) {
    graphics_view_image_ = image;
    QGraphicsScene *scene = new QGraphicsScene();
    scene->addPixmap(QPixmap::fromImage(graphics_view_image_));
    scene->setSceneRect(0, 0, graphics_view_image_.width(),
                        graphics_view_image_.height());
    ui->graphicsView->setScene(scene);
  }
}

void MainWindow::on_btnProceed_clicked() {
  if (paint_window_->isVisible()) {
    GraphicsViewUpdate(paint_window_->GetImage());
  }
  CreateVectorPixels(graphics_view_image_);
  if (!vector_pixels_.empty() && controller_->CheckModelState() > 0) {
    controller_->SetVectorPixelsOfImage(vector_pixels_);
    UpdateMLPState();
    DrawPreview();
    UpdateAnswerLabel();
  }
}

void MainWindow::CreateVectorPixels(QImage &image) {
  if (!image.isNull()) {
    if (!vector_pixels_.empty()) {
      vector_pixels_.clear();
    }
    QImage smallImage(image.scaled(count_neurons_, count_neurons_));
    for (int i = 0; i < count_neurons_; ++i) {
      for (int j = 0; j < count_neurons_; ++j) {
        vector_pixels_.push_back(smallImage.pixelColor(i, j).blackF());
      }
    }
  }
}

void MainWindow::on_btnStartLearn_clicked() {
  if (controller_->stop_) {
    controller_->StopTeachLoop(false);
    s21::LearnConfig learn_config;
    ui->btnStartLearn->setText("Stop");
    learn_config.num_batches = ui->valBatchNum->text().toInt();
    learn_config.num_epochs = ui->valEpochNum->text().toInt();
    controller_->TeachNetwork(learn_config);
    ui->btnStartLearn->setText("Start");
  } else {
    controller_->StopTeachLoop(true);
    ui->btnStartLearn->setText("Start");
  }
}

void MainWindow::on_valEpochNum_valueChanged(int arg1) {
  if (arg1 == 1) {
    ui->valBatchNum->setEnabled(true);
    UpdateBatchLabel();
  } else {
    ui->valBatchNum->setEnabled(false);
    ui->lblBatchLen->setText("");
  }
}

void MainWindow::UpdateBatchLabel() {
  if (num_images_ > 0) {
    unsigned int batch_len = num_images_ / ui->valBatchNum->text().toInt();
    QString lbl = QString::number(batch_len) + " images / batch";
    ui->lblBatchLen->setText(lbl);
  }
}

void MainWindow::on_valBatchNum_valueChanged(int arg1) {
  if (arg1 == 1) {
    ui->valEpochNum->setEnabled(true);
    UpdateBatchLabel();
  } else {
    ui->valEpochNum->setEnabled(false);
  }
  UpdateBatchLabel();
}

void MainWindow::on_tabWidget_tabBarClicked(int index) { EnableButtons(); }

bool MainWindow::EnableButtons() {
  s21::ModelState state = controller_->CheckModelState();
  ui->btnImgUp->setEnabled(state);
  ui->tabInit->setEnabled(controller_->stop_);
  ui->tabResearch->setEnabled(controller_->stop_ && state != s21::Empty);
  ui->tabTest->setEnabled(controller_->stop_ && state != s21::Empty);
  if (state > s21::Initialized) {
    ui->btnStartLearn->setEnabled(true);
  } else {
    ui->btnStartLearn->setEnabled(false);
  }
  if (state > s21::Initialized && !controller_->GetErrVector().empty()) {
    ui->CreateGraph->setEnabled(true);
  } else {
    ui->CreateGraph->setEnabled(false);
  }
  if (state != s21::Empty) {
    ui->tabLearn->setEnabled(true);
    ui->btnProceed->setEnabled(true);
    ui->valEpochNum->setEnabled(true);
    ui->valBatchNum->setEnabled(true);
  } else {
    ui->tabLearn->setEnabled(false);
    ui->btnProceed->setEnabled(false);
  }
  if (state > s21::Initialized && !controller_->GetInputValues().empty()) {
    ui->pushButtonResearch->setEnabled(true);
  } else {
    ui->pushButtonResearch->setEnabled(false);
  }
  if (state == s21::Learned) {
    ui->btnSaveNetworkConfiguration->setEnabled(true);
  }
  return state;
}

void MainWindow::on_progressChanged_(int i, int percentage) {
  ui->barLearnProgress->setValue(percentage);
  num_curr_image_ =
      (num_curr_image_ + i >= num_images_ ? num_curr_image_ + i - num_images_
                                          : num_curr_image_ + i);
  DrawPreview();
  UpdatePreviewLabel();
  UpdateAnswerLabel();
  UpdateMLPState();
  EnableButtons();
  QCoreApplication::processEvents();
}

void MainWindow::on_progressTestChanged_(int i, int percentage) {
  ui->barTestProgress->setValue(percentage);
  num_curr_image_ =
      (num_curr_image_ + i >= num_images_ ? num_curr_image_ + i - num_images_
                                          : num_curr_image_ + i);
  UpdateTestSheet();
  if (controller_->stop_) {
    ui->btnStartLearn->setEnabled(true);
    ui->btnStartTest->setText("Start");
  }
  QCoreApplication::processEvents();
}

void MainWindow::on_btnSaveNetworkConfiguration_clicked() {
  QFileDialog dialog;
  QRegularExpression rx("\\..+$");
  QString filters("Conf files (*.bin)");
  dialog.setDefaultSuffix(".bin");
  QString q_filename = dialog.getSaveFileName(this, "Save configuration", ".",
                                              filters, &filters);
  if (!rx.match(q_filename).hasMatch()) q_filename += ".bin";
  if (!q_filename.isEmpty()) {
    controller_->SaveConfiguration((q_filename).toStdString());
  }
}

void MainWindow::on_btnLoadNetworkConfiguration_clicked() {
  QString filters("Conf files (*.bin);;All files (*)");
  QString q_filename = QFileDialog::getOpenFileName(
      this, tr("Load configuration"), ".", filters);
  if (!q_filename.isEmpty()) {
    controller_->LoadConfiguration(q_filename.toStdString(),
                                   ui->rbtnGraph->isChecked());
    UpdateConfigurationView();
  }
}

void MainWindow::UpdateConfigurationView() {
  s21::InitConfig config = controller_->GetConfiguration();
  ui->num_layers_hidden->setValue(config.num_layers_hidden);
  ui->num_neurons_hidden->setValue(config.num_neurons_hidden);
}

void MainWindow::UpdateAnswerLabel() {
  std::vector<double> out = controller_->GetOutValues();
  int maxElementIndex = std::max_element(out.begin(), out.end()) - out.begin();
  ui->lblAnswer->setText(QString(QChar::fromLatin1(maxElementIndex + 65)));
}

void MainWindow::on_CreateGraph_clicked() {
  graph_window_->show();
  std::vector<double> v;
  std::vector<s21::ErrorData> err = controller_->GetErrVector();
  std::for_each(err.begin(), err.end(),
                [&v](s21::ErrorData el) { v.push_back(el.accuracy); });
  graph_window_->DrawGraph(v);
}

QString MainWindow::GetDatasetFileName() {
  QString file_name = "";
  QFileDialog *fileDialog = new QFileDialog(this);
  fileDialog->setWindowTitle(tr("Open dataset"));
  fileDialog->setDirectory(".");
  fileDialog->setFileMode(QFileDialog::ExistingFile);
  fileDialog->setViewMode(QFileDialog::Detail);
  if (fileDialog->exec()) {
    file_name = fileDialog->selectedFiles()[0];
  }
  return file_name;
}

void MainWindow::on_btnLoadDatasetTest_clicked() {
  QString file_name = GetDatasetFileName();
  if (!file_name.isEmpty()) {
    controller_->LoadDataset(file_name.toStdString());
    controller_->EvaluateErr();
    num_images_ = controller_->GetCountOfElements();
    num_curr_image_ = 0;
    UpdateTestSheet();
    EnableButtons();
  }
}

QPixmap MainWindow::GetPreviewPicture(int img_num) {
  QByteArray pData;
  std::vector<double> input = controller_->GetInputValues(img_num);
  std::for_each(input.begin(), input.end(), [&pData](double const &value) {
    pData.insert(0, ~0);
    pData.insert(0, (1 - value) * 255);
    pData.insert(0, (1 - value) * 255);
    pData.insert(0, (1 - value) * 255);
  });
  const unsigned char *imageData =
      reinterpret_cast<const unsigned char *>(pData.constData());
  QImage qim = QImage(imageData, 28, 28, QImage::Format_ARGB32_Premultiplied);
  qim = qim.transformed(QTransform().rotate(90)).scaled(280, 280);
  qim = qim.mirrored(false, true);
  QPixmap pixmap = QPixmap::fromImage(qim);
  return pixmap;
}

void MainWindow::DrawTestPreview(int img_num) {
  QPixmap pixmap = GetPreviewPicture(img_num);
  QImage image = pixmap.toImage();
  GraphicsViewUpdate(image);
}

void MainWindow::UpdateTestPreviewLabel() {
  QString lbl = " of " + QString::number(num_images_);
  ui->lblTotalImgsTest->setText(lbl);
  ui->inpNumCurrImgTest->setText(QString::number(num_curr_image_));
  int num_letter = controller_->GetCorrectValue() + 65;
  ui->lblLetterTest->setText(QString(QChar::fromLatin1(num_letter)));
}

void MainWindow::on_btnImgUpTest_clicked() {
  num_curr_image_++;
  controller_->LoadNextDataset();
  controller_->EvaluateErr();
  UpdateTestSheet();
}

void MainWindow::UpdateTestSheet() {
  DrawTestPreview();
  UpdateTestPreviewLabel();
  UpdateAnswerLabel();
  UpdateMLPState();
  EnableButtons();
}

void MainWindow::on_btnStartTest_clicked() {
  if (controller_->stop_) {
    controller_->StopTeachLoop(false);
    ui->btnStartTest->setText("Stop");
    controller_->TestNetwork(ui->valTestPercentage->value());
    ui->btnStartTest->setText("Start");
  } else {
    controller_->StopTeachLoop(true);
    ui->btnStartTest->setText("Start");
  }
  ui->btnStartLearn->setEnabled(controller_->stop_);
}

void MainWindow::on_MainWindow_destroyed() { controller_->StopTeachLoop(true); }

void MainWindow::on_rbtnGraph_clicked() {
  controller_->ResetNetworkConfiguration();
}

void MainWindow::on_rbtnMatrix_clicked() {
  controller_->ResetNetworkConfiguration();
}

void MainWindow::on_num_layers_hidden_valueChanged(int arg1) {
  controller_->ResetNetworkConfiguration();
}

void MainWindow::on_num_neurons_hidden_valueChanged(int arg1) {
  controller_->ResetNetworkConfiguration();
}

auto MainWindow::ResearchTestingTime(const int count) -> double {
  clock_t t1, t2;
  t1 = std::clock();
  for (int i = 0; i < count; ++i) {
    controller_->TestNetwork(ui->valTestPercentage->value());
  }
  t2 = std::clock();
  return ((double)(t2 - t1) / CLOCKS_PER_SEC);
}

void MainWindow::on_pushButtonResearch_clicked() {
  ui->tabWidget->setCurrentIndex(3);
  ui->tabResearch->setEnabled(true);
  controller_->SaveConfiguration(
      QDir::currentPath().append("/test.bin").toStdString());
  for (int i = 1; i <= 2; ++i) {
    if (i == 1) {
      controller_->LoadConfiguration(
          QDir::currentPath().append("/test.bin").toStdString(), false);
    } else {
      controller_->LoadConfiguration(
          QDir::currentPath().append("/test.bin").toStdString(), true);
    }
    double averageTime(0.0);
    for (int j = 1; j <= 3; ++j) {
      double time(0.0);
      time = ResearchTestingTime(pow(10, j));
      averageTime += time;
      ui->tableWidget->setItem(
          i, j,
          new QTableWidgetItem(QString(QString::number(time).append(" sec"))));
    }
    averageTime /= 1110;
    ui->tableWidget->setItem(i, 4,
                             new QTableWidgetItem(QString(
                                 QString::number(averageTime).append(" sec"))));
  }
}
