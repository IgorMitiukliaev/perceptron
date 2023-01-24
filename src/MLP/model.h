#ifndef SRC_MLP_MODEL_H_
#define SRC_MLP_MODEL_H_
#include <QDebug>
#include <cmath>
#include <string>

#include "fileloader.h"
#include "neuralnetwork.h"

using std::string;

namespace s21 {
struct ErrorData {
  std::time_t time_reset, time_lap;
  long count;
  long count_success;
  double precision, recall, f_measure, accuracy;
  s21::Matrix *confusion_matrix;
};

enum ModelState {
  Empty = 0,
  Initialized = 1,
  DatasetReady = 2,
  Learned = 3,
};

class Model {
 private:
  s21::FileLoader fileloader_;
  s21::NeuralNetwork *network_;
  std::vector<double> input_, out_, correct_, sum_err_;
  ErrorData err_;
  std::vector<double> input_value_;
  long num_images_;
  unsigned int num_layers_hidden_ = 2;
  unsigned int num_neurons_hidden_ = 100;
  unsigned int num_neurons_input_ = 784;
  unsigned int num_neurons_out_ = 26;
  unsigned int num_epochs_ = 0;
  unsigned int num_batches_ = 0;
  void NormalizeInput();

 public:
  Model();
  ~Model();
  void InitNetwork(const InitConfig &config);
  void LoadDataset(string const &path);
  void Activate(const std::vector<double> &input);
  void LoadNextDataset();
  void TeachNetwork();
  void TeachNetwork(const LearnConfig &learn_config);
  void UpdateErrData();
  void EvaluateErr();

  // simple functions
  auto CheckModelState() -> s21::ModelState;
  auto GetInputValues(int img_num = 0) -> std::vector<double> { return input_; }
  auto GetOutValues() -> std::vector<double> { return out_; }
  auto GetCorrectValue(int img_num) -> std::vector<double> { return correct_; }
  auto GetCountOfElements() -> long { return num_images_; }
  auto ResetErr() -> void;
  auto GetErr() -> s21::ErrorData & { return err_; }
  auto ResetNetworkConfiguration() -> void {
    if (network_) delete network_;
    network_ = nullptr;
  };

  auto SetVectorPixelsOfImage(const std::vector<double> &vector_pixels)
      -> void {
    input_ = vector_pixels;
  }
  void SaveConfiguration(const std::string &filename);
  void LoadConfiguration(const std::string &filename, bool is_graph);
  InitConfig GetConfiguration();
};
}  // namespace s21
#endif  // SRC_MLP_MODEL_H_
