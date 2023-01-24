#ifndef SRC_MLP_MATRIXNEURALNETWORK_H_
#define SRC_MLP_MATRIXNEURALNETWORK_H_

#include "matrix.h"
#include "neuralnetwork.h"

namespace s21 {
class MatrixNeuralNetwork : public NeuralNetwork {
 public:
    MatrixNeuralNetwork() {}
    ~MatrixNeuralNetwork() {}
    void InitNetwork(const InitConfig*  config) override;       // Инициализация весов
    void Activate(const std::vector<double>& input) override;  //  Прямое распространение сигнала
    std::vector<double> GetOutput() override;
    void TeachNetwork(const std::vector<double>& correct) override;
    void SaveConfiguration(const std::string& filename) override;
    void LoadConfiguration(const std::string& filename) override;
    s21::InitConfig GetConfiguration() override;

 private:
    std::vector<Matrix> weights_;
    std::vector<std::vector<double>> neurons_val_;
    std::vector<std::vector<double>> neurons_err_;
    void InitWeights();
    void InitNeuronsValues();
    void InitNeuronsErrors();
    void Sigmoid(std::vector<double>& a, int n);
    double DerSigmoid(double a);
    void BackPropagationSignal(const std::vector<double>& correct);
    void CalcWeights(double learning_rate);
};
}  // namespace s21
#endif  // SRC_MLP_MATRIXNEURALNETWORK_H_
