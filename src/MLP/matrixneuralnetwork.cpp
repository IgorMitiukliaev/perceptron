#include "matrixneuralnetwork.h"

void s21::MatrixNeuralNetwork::InitNetwork(const s21::InitConfig* config) {
  num_layers_hidden_ = config->num_layers_hidden;
  num_neurons_hidden_ = config->num_neurons_hidden;
  num_neurons_input_ = config->num_neurons_input;
  num_neurons_out_ = config->num_neurons_out;
  InitWeights();
  InitNeuronsValues();
  InitNeuronsErrors();
}

void s21::MatrixNeuralNetwork::InitWeights() {
  weights_.resize(num_layers_hidden_ + 1);
  weights_[0].InitRand(num_neurons_hidden_, num_neurons_input_);
  for (auto i = 1; i <= num_layers_hidden_ - 1; i++) {
    weights_[i].InitRand(num_neurons_hidden_, num_neurons_hidden_);
  }
  weights_[num_layers_hidden_].InitRand(num_neurons_out_, num_neurons_hidden_);
}

void s21::MatrixNeuralNetwork::InitNeuronsValues() {
  neurons_val_.resize(num_layers_hidden_ + 2);
  neurons_val_[0].resize(num_neurons_input_);
  for (auto i = 1; i <= num_layers_hidden_; i++)
    neurons_val_[i].resize(num_neurons_hidden_);
  neurons_val_[num_layers_hidden_ + 1].resize(num_neurons_out_);
}

void s21::MatrixNeuralNetwork::InitNeuronsErrors() {
  neurons_err_.resize(num_layers_hidden_ + 2);
  neurons_err_[0].resize(num_neurons_input_);
  for (auto i = 1; i <= num_layers_hidden_; i++) {
    neurons_err_[i].resize(num_neurons_hidden_);
  }
  neurons_err_[num_layers_hidden_ + 1].resize(num_neurons_out_);
}

void s21::MatrixNeuralNetwork::Sigmoid(std::vector<double>& a, int n) {
  for (auto i = 0; i < n; i++) a[i] = 1 / (1 + exp(-a[i]));
}

double s21::MatrixNeuralNetwork::DerSigmoid(double a) {
  double res = a * (1 - a);
  return res;
}

void s21::MatrixNeuralNetwork::Activate(const std::vector<double>& input) {
  for (auto i = 0; i < num_neurons_input_; i++) {
    neurons_val_[0][i] = input[i];
  }
  for (int i = 1; i <= num_layers_hidden_; i++) {
    Matrix::Mult(weights_[i - 1], neurons_val_[i - 1], neurons_val_[i]);
    Sigmoid(neurons_val_[i], num_neurons_hidden_);
  }
  auto i = num_layers_hidden_ + 1;
  Matrix::Mult(weights_[i - 1], neurons_val_[i - 1], neurons_val_[i]);
  Sigmoid(neurons_val_[i], num_neurons_out_);
}

std::vector<double> s21::MatrixNeuralNetwork::GetOutput() {
  std::vector<double> out;
  for (auto i = 0; i < num_neurons_out_; i++)
    out.push_back(neurons_val_[num_layers_hidden_ + 1][i]);
  return out;
}

void s21::MatrixNeuralNetwork::BackPropagationSignal(
    const std::vector<double>& correct) {
  auto k = num_layers_hidden_ + 1;
  for (auto i = 0; i < num_neurons_out_; i++) {
    neurons_err_[k][i] =
        (correct[i] - neurons_val_[k][i]) * DerSigmoid(neurons_val_[k][i]);
  }
  for (auto i = num_layers_hidden_; i > 1; i--) {
    Matrix::TransposeMult(weights_[i], neurons_err_[i + 1], neurons_err_[i]);
    for (auto j = 0; j < num_neurons_hidden_; j++) {
      neurons_err_[i][j] *= DerSigmoid(neurons_val_[i][j]);
    }
  }
  k = 0;
  Matrix::TransposeMult(weights_[k], neurons_err_[k + 1], neurons_err_[k]);
  for (auto j = 0; j < num_neurons_input_; j++) {
    neurons_err_[k][j] *= DerSigmoid(neurons_val_[k][j]);
  }
}

void s21::MatrixNeuralNetwork::CalcWeights(double learning_rate) {
  auto i2 = 0;
  for (auto j = 0; j < num_neurons_hidden_; j++) {
    for (auto k = 0; k < num_neurons_input_; k++) {
      weights_[i2](j, k) +=
          neurons_val_[i2][k] * neurons_err_[i2 + 1][j] * learning_rate;
    }
  }
  for (auto i = 1; i < num_layers_hidden_; i++) {
    for (auto j = 0; j < num_neurons_hidden_; j++) {
      for (auto k = 0; k < num_neurons_hidden_; k++) {
        weights_[i](j, k) +=
            neurons_val_[i][k] * neurons_err_[i + 1][j] * learning_rate;
      }
    }
  }
  i2 = num_layers_hidden_;
  for (auto j = 0; j < num_neurons_out_; j++) {
    for (auto k = 0; k < num_neurons_hidden_; k++) {
      weights_[i2](j, k) +=
          neurons_val_[i2][k] * neurons_err_[i2 + 1][j] * learning_rate;
    }
  }
}

void s21::MatrixNeuralNetwork::TeachNetwork(
    const std::vector<double>& correct) {
  BackPropagationSignal(correct);
  CalcWeights(0.1);
}

void s21::MatrixNeuralNetwork::SaveConfiguration(const std::string& filename) {
  std::ofstream out(filename, std::ios::binary | std::ios::out);
  s21::InitConfig config = GetConfiguration();
  out.write((char*)&config, sizeof(config));

  for (auto i = 0; i < num_layers_hidden_ + 1; i++) {
    weights_[i].Save(out);
  }

  out.close();
}

void s21::MatrixNeuralNetwork::LoadConfiguration(const std::string& filename) {
  std::ifstream in(filename, std::ios::binary | std::ios::in);
  s21::InitConfig config;
  in.read((char*)&config, sizeof(config));
  InitNetwork(&config);
  for (auto i = 0; i < num_layers_hidden_ + 1; i++) {
    weights_[i].Load(in);
  }
  in.close();
}

s21::InitConfig s21::MatrixNeuralNetwork::GetConfiguration() {
  s21::InitConfig config = {.num_neurons_input = num_neurons_input_,
                            .num_layers_hidden = num_layers_hidden_,
                            .num_neurons_hidden = num_neurons_hidden_,
                            .num_neurons_out = num_neurons_out_,
                            .is_graph = false};
  return config;
}
