#include "model.h"

#include "graphneuralnetwork.h"
#include "matrixneuralnetwork.h"

using s21::Model;

s21::Model::Model() {
  network_ = nullptr;
  err_.confusion_matrix = nullptr;
  num_images_ = 0;
  err_.accuracy = 0;
  err_.count = 0;
  err_.count_success = 0;
  err_.f_measure = 0;
  err_.precision = 0;
  err_.recall = 0;
}

s21::Model::~Model() {
  delete network_;
  delete err_.confusion_matrix;
}

void Model::InitNetwork(const s21::InitConfig &config) {
  num_layers_hidden_ = config.num_layers_hidden;
  num_neurons_hidden_ = config.num_neurons_hidden,
  num_neurons_input_ = config.num_neurons_input,
  num_neurons_out_ = config.num_neurons_out;
  if (network_) delete network_;
  if (config.is_graph) {
    network_ = new GraphNeuralNetwork();
  } else {
    network_ = new MatrixNeuralNetwork();
  }
  network_->InitNetwork(&config);
  err_ = {0};
  err_.confusion_matrix = new Matrix(num_neurons_out_, num_neurons_out_);
  input_.clear();
}

void Model::LoadDataset(string const &path) {
  fileloader_.SetFileStream(path);
  num_images_ = fileloader_.GetCountOfElements();
  fileloader_.ReadElement();
  input_ = fileloader_.GetInputValues();
  correct_ = fileloader_.GetOutputValues();
  input_value_ = fileloader_.GetOutputValues();
  NormalizeInput();
  ResetErr();
}

void Model::LoadNextDataset() {
  bool check = fileloader_.ReadElement();
  if (!check) {
    fileloader_.StartReadElements();
    check = fileloader_.ReadElement();
  }
  if (!check) throw std::runtime_error("Error reading file: rewind failed");
  input_ = fileloader_.GetInputValues();
  NormalizeInput();
  correct_ = fileloader_.GetOutputValues();
}

void Model::NormalizeInput() {
  double max = *max_element(input_.begin(), input_.end());
  double min = *min_element(input_.begin(), input_.end());
  if (max > min) {
    std::for_each(input_.begin(), input_.end(), [min, max](double &value) {
      value = (value - min) / (max - min);
    });
  } else {
    std::for_each(input_.begin(), input_.end(),
                  [](double &value) { value = 0; });
  }
}

void Model::Activate(const std::vector<double> &input_) {
  network_->Activate(input_);
  out_ = network_->GetOutput();
  UpdateErrData();
}

void Model::TeachNetwork() { network_->TeachNetwork(correct_); }

void Model::TeachNetwork(const LearnConfig &learn_config) {
  num_epochs_ = learn_config.num_epochs,
  num_batches_ = learn_config.num_batches;
  for (unsigned int i = 0, j = 0; i < num_epochs_; i++) {
    while (j++ < num_images_) {
      network_->TeachNetwork(correct_);
      LoadNextDataset();
    }
    fileloader_.StartReadElements();
  }
}

void Model::SaveConfiguration(const std::string &filename) {
  network_->SaveConfiguration(filename);
}

void Model::LoadConfiguration(const std::string &filename, bool is_graph) {
  if (network_) delete network_;
  if (is_graph) {
    network_ = new GraphNeuralNetwork();
  } else {
    network_ = new MatrixNeuralNetwork();
  }
  network_->LoadConfiguration(filename);
  InitConfig config = network_->GetConfiguration();
  num_layers_hidden_ = config.num_layers_hidden;
  num_neurons_hidden_ = config.num_neurons_hidden;
  num_neurons_input_ = config.num_neurons_input;
  num_neurons_out_ = config.num_neurons_out;
  ResetErr();
}

s21::InitConfig Model::GetConfiguration() {
  return network_->GetConfiguration();
}

void Model::ResetErr() {
  if (err_.confusion_matrix) delete err_.confusion_matrix;
  err_ = {0};
  int size = network_->GetConfiguration().num_neurons_out;
  err_.confusion_matrix = new s21::Matrix(size, size);
  err_.time_reset = std::time(nullptr);
}

void Model::UpdateErrData() {
  if (correct_.size() > 0) {
    err_.count++;
    int correctLetterIndex =
        std::max_element(correct_.begin(), correct_.end()) - correct_.begin();
    int answerLetterIndex =
        std::max_element(out_.begin(), out_.end()) - out_.begin();
    if (correctLetterIndex == answerLetterIndex) err_.count_success++;
    ((*err_.confusion_matrix)(answerLetterIndex, correctLetterIndex))++;
  }
}

void Model::EvaluateErr() {
  // here come some definitions

  // TP = True positives (given A predicted as A)
  // FP = False positives (predicted as A but not A)
  // FN = False negatives (given A not predicted as A)
  // Accuracy = Success count / Total count
  // Precision = TP / (TP + FP)
  // Recall = TP / (TP + FN)
  // f-measure = 2*(Precision * Recall)/(Precision + Recall)
  // source:
  // https://towardsdatascience.com/precision-recall-and-f1-score-of-multiclass-classification-learn-in-depth-6c194b217629

  err_.precision = err_.recall = 0;
  long count_p = 0, count_r = 0;
  for (int i = 0; i < num_neurons_out_; i++) {
    int __sum = (*err_.confusion_matrix).SumColumn(i);
    if (__sum > 0)
      err_.precision += (*err_.confusion_matrix)(i, i) / __sum;
    else
      count_p++;
    __sum = (*err_.confusion_matrix).SumRow(i);
    if (__sum > 0)
      err_.recall += (*err_.confusion_matrix)(i, i) / __sum;
    else
      count_r++;
  }
  err_.precision /= (num_neurons_out_ - count_p);
  err_.recall /= (num_neurons_out_ - count_r);
  err_.accuracy = (double)err_.count_success / err_.count;
  err_.f_measure =
      2 * (err_.precision * err_.recall) / (err_.precision + err_.recall);
  err_.time_lap = std::time(nullptr) - err_.time_reset;
}
s21::ModelState Model::CheckModelState() {
  ModelState res = Empty;
  if (network_ != nullptr) res = Initialized;
  if (!input_.empty()) res = DatasetReady;
  if (err_.accuracy > 0.5) res = Learned;
  return res;
}
