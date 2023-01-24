#ifndef SRC_MLP_CONTROLLER_H_
#define SRC_MLP_CONTROLLER_H_
#include <QApplication>
#include <string>

#include "model.h"

namespace s21 {
class Controller : public QObject {
  Q_OBJECT

 private:
  s21::Model *model_;
  std::vector<ErrorData> error_data_vector_;

 public:
  explicit Controller(s21::Model *model) : model_(model) {}
  bool stop_ = true;
  bool load_ = false;
  void LoadDataset(string const &path);
  void LoadNextDataset();
  void InitNetwork(const InitConfig &config);
  std::vector<double> GetOutValues();
  auto GetCorrectValue() -> unsigned;
  void TeachNetwork(LearnConfig const &learn_config);
  void TestNetwork(unsigned int percent);
  void ResetNetworkConfiguration();

  // simple functions
  auto GetCountOfElements() -> long { return model_->GetCountOfElements(); }
  auto GetInputValues(int img_num = 0) -> std::vector<double>;
  auto CheckModelState() -> s21::ModelState { return model_->CheckModelState(); };
  auto StopTeachLoop(bool val) -> void { stop_ = val; };
  auto SaveConfiguration(const std::string &filename) -> void {
    model_->SaveConfiguration(filename);
  };
  auto LoadConfiguration(const std::string &filename, bool is_graph) -> void {
    model_->LoadConfiguration(filename, is_graph);
    load_ = true;
  };
  auto GetConfiguration() -> s21::InitConfig { return model_->GetConfiguration(); };
  auto GetErr() -> s21::ErrorData & { return model_->GetErr(); }
  auto GetErrVector() -> const std::vector<s21::ErrorData> & {
    return error_data_vector_;
  }
  auto EvaluateErr() -> void { model_->EvaluateErr(); };

  auto SetVectorPixelsOfImage(const std::vector<double> &vector_pixels)
      -> void {
    model_->SetVectorPixelsOfImage(vector_pixels);
    model_->Activate(vector_pixels);
  }

 signals:
  void progressChanged_(int value, int value2);
  void progressTestChanged_(int value, int value2);
};
}  // namespace s21
#endif  // SRC_MLP_CONTROLLER_H_
