#ifndef SRC_MLP_MATRIX_H_
#define SRC_MLP_MATRIX_H_

#include <cmath>
#include <fstream>
#include <vector>

namespace s21 {
class Matrix {
 public:
    Matrix() {}
    Matrix(int rows, int columns);
    ~Matrix() {}
    void InitRand(int rows, int columns);
    static void Mult(const Matrix &m, const std::vector<double> &b, std::vector<double> &c);
    static void TransposeMult(const Matrix &m, const std::vector<double> &b, std::vector<double> &c);
    double &operator()(int i, int j);
    void Save(std::ofstream &out);
    void Load(std::ifstream &in);
    double SumRow(int row);
    double SumColumn(int column);

 private:
    int rows_;
    int columns_;
    std::vector<std::vector<double>> matrix_;
    void Resize();
};

};      // namespace s21
#endif  // SRC_MLP_MATRIX_H_
