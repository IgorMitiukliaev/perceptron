#include "graphneuralnetwork.h"

using s21::GraphNeuralNetwork;
using s21::NeuralNetwork;

double GraphNeuralNetwork::Neuron::Sigmoid(double x) {
  return 1 / (1 + exp(-x));
}
double GraphNeuralNetwork::Neuron::DSigmoid(double x) {
  return Sigmoid(x) * (1 - Sigmoid(x));
}

void GraphNeuralNetwork::Neuron::Activate(const double input = 1) {
  sum = 0;
  if (n[0] == nullptr) {
    sum = w[0] * input;
  } else {
    for (int i = 0; i < n.size(); i++) {
      sum += w[i] * n[i]->GetResponse();
    }
  }
  out = Sigmoid(sum);
  dout = DSigmoid(sum);
}

void GraphNeuralNetwork::Activate(const std::vector<double> &input_data) {
  for (unsigned int i = 0; i < num_neurons_input_; i++) {
    input_layer_[i].Activate(input_data[i]);
  }
  for (unsigned int i = 0; i < num_layers_hidden_; i++) {
    for (unsigned int j = 0; j < num_neurons_hidden_; j++) {
      hidden_layer_[i][j].Activate();
    }
  }
  for (unsigned int i = 0; i < num_neurons_out_; i++) {
    out_layer_[i].Activate();
  }
}

void GraphNeuralNetwork::InitNetwork(const InitConfig *config) {
  num_layers_hidden_ = config->num_layers_hidden;
  num_neurons_hidden_ = config->num_neurons_hidden;
  num_neurons_input_ = config->num_neurons_input;
  num_neurons_out_ = config->num_neurons_out;
  input_layer_ = std::vector<Neuron>(config->num_neurons_input);

  for (unsigned int i = 0; i < config->num_layers_hidden; i++) {
    for (unsigned int j = 0; j < num_neurons_hidden_; j++) {
      if (i == 0) {
        hidden_layer_[i].push_back(Neuron(&input_layer_));
      } else {
        hidden_layer_[i].push_back(Neuron(&hidden_layer_[i - 1]));
      }
    }
  }
  for (unsigned int i = 0; i < config->num_neurons_out; i++) {
    out_layer_.push_back(Neuron(&hidden_layer_[num_layers_hidden_ - 1]));
  }

  for (unsigned int i = 0; i < num_neurons_input_; i++) {
    input_layer_[i].p.clear();
    for (unsigned int j = 0; j < num_neurons_hidden_; j++) {
      input_layer_[i].p.push_back(&hidden_layer_[0][j]);
    }
  }

  for (unsigned int i = 0; i < num_neurons_hidden_; i++) {
    hidden_layer_[num_layers_hidden_ - 1][i].p.clear();
    for (unsigned int j = 0; j < num_neurons_out_; j++) {
      hidden_layer_[num_layers_hidden_ - 1][i].p.push_back(&out_layer_[j]);
    }
  }

  for (unsigned int i = 0; i < num_layers_hidden_ - 1; i++) {
    for (unsigned int j = 0; j < num_neurons_hidden_; j++) {
      hidden_layer_[i][j].p.clear();
      for (unsigned int k = 0; k < num_neurons_hidden_; k++) {
        hidden_layer_[i][j].p.push_back(&hidden_layer_[i + 1][k]);
      }
    }
  }
}

GraphNeuralNetwork::Neuron::Neuron()
    : w(std::vector<double>(1)),
      dw(std::vector<double>(1)),
      n(std::vector<Neuron *>(1)),
      p(std::vector<Neuron *>(1)) {
  n[0] = nullptr;
  w[0] = 1;
  p[0] = nullptr;
}

GraphNeuralNetwork::Neuron::Neuron(std::vector<Neuron> *layer) : Neuron() {
  n = std::vector<Neuron *>(layer->size());
  w = std::vector<double>(layer->size());
  dw = std::vector<double>(layer->size());
  p = std::vector<Neuron *>(1);
  for (int i = 0; i < n.size(); i++) n[i] = &(*layer)[i];
  std::random_device rd;
  std::default_random_engine eng(rd());
  std::uniform_real_distribution<double> distr(-1, 1);
  std::for_each(w.begin(), w.end(),
                [&distr, &eng](double &el) { el = distr(eng); });
}

std::vector<double> GraphNeuralNetwork::GetOutput() {
  std::vector<double> res;
  for_each(out_layer_.begin(), out_layer_.end(),
           [&res](Neuron &el) { res.push_back(el.GetResponse()); });
  return res;
}

void GraphNeuralNetwork::Neuron::EvaluateErr(unsigned int num_pos = 0,
                                             double correct = 0) {
  delta_ = 0;
  if (p[0] == nullptr) {
    delta_ = (correct - out) * dout;
  } else {
    std::for_each(p.begin(), p.end(), [&](Neuron *el) {
      delta_ += el->GetWeight(num_pos) * el->GetDelta();
    });
    delta_ *= dout;
  }
}

void GraphNeuralNetwork::Neuron::RefreshWeight(double const &a_,
                                               double const &g_) {
  for (unsigned int i = 0; i < dw.size(); i++) {
    dw[i] *= g_;
    dw[i] += a_ * delta_ * n[i]->GetResponse();
    w[i] += dw[i];
  }
}

void GraphNeuralNetwork::TeachNetwork(const std::vector<double> &correct) {
  for (unsigned int i = 0; i < num_neurons_out_; i++) {
    out_layer_[i].EvaluateErr(i, correct[i]);
  }
  for (int i = num_layers_hidden_ - 1; i >= 0; i--) {
    for (unsigned int j = 0; j < num_neurons_hidden_; j++) {
      hidden_layer_[i][j].EvaluateErr(j, 0);
    }
  }
  for (unsigned int i = 0; i < num_neurons_input_; i++) {
    input_layer_[i].EvaluateErr(i, 0);
  }
  for (unsigned int i = 0; i < num_neurons_out_; i++) {
    out_layer_[i].RefreshWeight(a_, g_);
  }
  for (int i = num_layers_hidden_ - 1; i >= 0; i--) {
    for (unsigned int j = 0; j < num_neurons_hidden_; j++) {
      hidden_layer_[i][j].RefreshWeight(a_, g_);
    }
  }
}

bool GraphNeuralNetwork::CheckNetworkReady() { return true; }

void s21::GraphNeuralNetwork::SaveConfiguration(const std::string &filename) {
  std::ofstream out(filename, std::ios::binary | std::ios::out);
  s21::InitConfig config = GetConfiguration();
  out.write((char *)&config, sizeof(config));

  for (unsigned i = 0; i < num_layers_hidden_; i++) {
    std::for_each(hidden_layer_[i].begin(), hidden_layer_[i].end(),
                  [&](Neuron &el) { SaveWeight(out, el.w); });
  }

  std::for_each(out_layer_.begin(), out_layer_.end(),
                [&](Neuron &el) { SaveWeight(out, el.w); });

  out.close();
}

void s21::GraphNeuralNetwork::LoadConfiguration(const std::string &filename) {
  std::ifstream in(filename, std::ios::binary | std::ios::in);
  s21::InitConfig config;
  in.read((char *)&config, sizeof(config));
  InitNetwork(&config);
  for (unsigned i = 0; i < num_layers_hidden_; i++) {
    std::for_each(hidden_layer_[i].begin(), hidden_layer_[i].end(),
                  [&](Neuron &el) { LoadWeight(in, el.w); });
  }

  std::for_each(out_layer_.begin(), out_layer_.end(),
                [&](Neuron &el) { LoadWeight(in, el.w); });

  in.close();
}

s21::InitConfig s21::GraphNeuralNetwork::GetConfiguration() {
  s21::InitConfig config = {.num_neurons_input = num_neurons_input_,
                            .num_layers_hidden = num_layers_hidden_,
                            .num_neurons_hidden = num_neurons_hidden_,
                            .num_neurons_out = num_neurons_out_,
                            .is_graph = false};
  return config;
}

void s21::GraphNeuralNetwork::SaveWeight(std::ofstream &out,
                                         std::vector<double> &weight) {
  std::for_each(weight.begin(), weight.end(),
                [&out](double &w) { out.write((char *)&(w), sizeof(double)); });
}

void s21::GraphNeuralNetwork::LoadWeight(std::ifstream &in,
                                         std::vector<double> &weight) {
  std::for_each(weight.begin(), weight.end(),
                [&in](double &w) { in.read((char *)&(w), sizeof(double)); });
}
