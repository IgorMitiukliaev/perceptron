#ifndef SRC_MLP_NEURALNETWORK_H_
#define SRC_MLP_NEURALNETWORK_H_

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

#include "matrix.h"

namespace s21 {
struct InitConfig {
    unsigned int num_neurons_input;
    unsigned int num_layers_hidden;
    unsigned int num_neurons_hidden;
    unsigned int num_neurons_out;
    bool is_graph;
};

struct LearnConfig {
    unsigned int num_batches;
    unsigned int num_epochs;
};

class NeuralNetwork {
 public:
    NeuralNetwork() {}
    virtual ~NeuralNetwork() {}
    virtual void InitNetwork(InitConfig const *config) {}  // инициализация из конфига
    virtual void SaveConfiguration(const std::string &filename) {}
    virtual void LoadConfiguration(const std::string &filename) {}
    virtual InitConfig GetConfiguration() {
        return InitConfig();
    }
    virtual void Activate(const std::vector<double> &input) {}
    virtual std::vector<double> GetOutput() { return std::vector<double>(1); }
    virtual bool CheckNetworkReady() { return true; }
    virtual void TeachNetwork(const std::vector<double> &e) {}

 protected:
    unsigned int num_layers_hidden_ = 2;
    unsigned int num_neurons_hidden_ = 100;
    unsigned int num_neurons_input_ = 784;
    unsigned int num_neurons_out_ = 26;
};

}  // namespace s21

#endif  // SRC_MLP_NEURALNETWORK_H_
