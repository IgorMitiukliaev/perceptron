#ifndef SRC_MLP_GRAPHNEURALNETWORK_H_
#define SRC_MLP_GRAPHNEURALNETWORK_H_

#include "neuralnetwork.h"

namespace s21 {
class GraphNeuralNetwork : public NeuralNetwork {
 private:
    class Neuron {
     public:
        std::vector<double> w;
        std::vector<double> dw;
        std::vector<Neuron *> n;
        double sum = 0, out = 0, dout = 0, err_ = 0, delta_ = 0;
        double Sigmoid(double x);
        double DSigmoid(double x);

     public:
        Neuron();
        ~Neuron() {}
        std::vector<Neuron *> p;
        explicit Neuron(std::vector<Neuron> *input_layer);
        void Activate(const double input);
        double GetResponse() { return out; }
        double GetDResponse() { return dout; }
        double GetInput() { return sum; }
        void EvaluateErr(unsigned int i, double correct);
        void RefreshWeight(double const &a_, double const &g_);
        double GetDelta() { return delta_; }
        std::vector<double> GetWeights();
        double GetWeight(int i) {
            if (i < 0 || i >= w.size()) throw std::runtime_error("index out of bounds");
            return w[i];
        }
    };
    double const a_ = 0.07, g_ = 0;
    std::vector<Neuron> input_layer_;
    std::vector<Neuron> out_layer_;
    std::vector<Neuron> hidden_layer_[5];

 public:
    GraphNeuralNetwork() {}
    ~GraphNeuralNetwork() {}
    auto InitNetwork(const InitConfig *config) -> void override;
    auto CheckNetworkReady() -> bool override;
    auto Activate(const std::vector<double> &input) -> void override;
    auto GetOutput() -> std::vector<double> override;
    auto TeachNetwork(const std::vector<double> &correct) -> void override;
    auto SaveConfiguration(const std::string &filename) -> void override;
    auto LoadConfiguration(const std::string &filename) -> void override;
    auto GetConfiguration() -> s21::InitConfig override;

 private:
    auto SaveWeight(std::ofstream &out, std::vector<double> &weight) -> void;
    auto LoadWeight(std::ifstream &in, std::vector<double> &weight) -> void;
};

};      // namespace s21
#endif  // SRC_MLP_GRAPHNEURALNETWORK_H_
