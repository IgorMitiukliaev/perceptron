#include "matrix.h"

#include <cstdlib>
#include <ctime>
#include <iostream>

void s21::Matrix::Matrix::InitRand(int rows, int columns) {
    rows_ = rows;
    columns_ = columns;
    std::srand(time(NULL));
    Resize();
    for (auto i = 0; i < rows_; i++) {
        for (auto j = 0; j < columns_; j++) {
            matrix_[i][j] = -1 + 0.01 * (std::rand() % 201);
        }
    }
}

void s21::Matrix::Mult(const Matrix &m, const std::vector<double> &b,
                       std::vector<double> &c) {
    for (auto i = 0; i < m.rows_; i++) {
        c[i] = 0;
        for (auto j = 0; j < m.columns_; j++) {
            c[i] += m.matrix_[i][j] * b[j];
        }
    }
}

void s21::Matrix::TransposeMult(const Matrix &m, const std::vector<double> &b,
                                std::vector<double> &c) {
    for (auto i = 0; i < m.columns_; i++) {
        c[i] = 0;
        for (auto j = 0; j < m.rows_; j++) {
            c[i] += m.matrix_[j][i] * b[j];
        }
    }
}

double &s21::Matrix::operator()(int i, int j) { return matrix_[i][j]; }

void s21::Matrix::Save(std::ofstream &out) {
    for (auto i = 0; i < rows_; i++) {
        for (auto j = 0; j < columns_; j++) {
            out.write((char *)&(matrix_[i][j]), sizeof(double));
        }
    }
}

void s21::Matrix::Load(std::ifstream &in) {
    for (auto i = 0; i < rows_; i++) {
        for (auto j = 0; j < columns_; j++) {
            in.read((char *)&(matrix_[i][j]), sizeof(double));
        }
    }
}

void s21::Matrix::Resize() {
    matrix_.resize(rows_);
    for (auto i = 0; i < rows_; i++) {
        matrix_[i].resize(columns_);
    }
}

s21::Matrix::Matrix(int rows, int columns) : rows_(rows), columns_(columns) {
    Resize();
    for (auto i = 0; i < rows_; i++) {
        for (auto j = 0; j < columns_; j++) {
            matrix_[i][j] = 0;
        }
    }
}

double s21::Matrix::SumRow(int row) {
    double res = 0;
    for (auto i = 0; i < columns_; i++) {
        res += matrix_[row][i];
    }
    return res;
}

double s21::Matrix::SumColumn(int column) {
    double res = 0;
    for (auto i = 0; i < rows_; i++) {
        res += matrix_[i][column];
    }
    return res;
}

