#include "controller.h"

#include <QDebug>

#include "model.h"

using s21::Controller, s21::InitConfig, s21::InitConfig, s21::Model;

void Controller::LoadDataset(string const &path) {
  model_->LoadDataset(path);
  model_->Activate(model_->GetInputValues());
}

std::vector<double> Controller::GetInputValues(int img_num) {
  return model_->GetInputValues(img_num);
}

std::vector<double> Controller::GetOutValues() {
  return model_->GetOutValues();
}

void Controller::LoadNextDataset() {
  model_->LoadNextDataset();
  model_->Activate(model_->GetInputValues());
}

void Controller::InitNetwork(const s21::InitConfig &config) {
  model_->InitNetwork(config);
}

unsigned Controller::GetCorrectValue() {
  std::vector<double> const &correct = model_->GetCorrectValue(0);
  int i = 0;
  for (; correct[i] == 0; i++) {
  }
  return i;
}

void Controller::TeachNetwork(LearnConfig const &learn_config) {
  model_->ResetErr();
  const int p_bar_increment = 100;
  unsigned int num_epochs_ = learn_config.num_epochs;
  unsigned int num_batches_ = learn_config.num_batches;
  int batches_count = 0;
  unsigned int const &num_images_ = model_->GetCountOfElements();
  model_->Activate(model_->GetInputValues());
  if (num_batches_ == 1) {
    error_data_vector_.resize(num_epochs_);
    long max_count = num_epochs_ * num_images_;
    for (unsigned int i = 0; (i < max_count) && !stop_; i++) {
      model_->TeachNetwork();
      LoadNextDataset();
      if (i % p_bar_increment == 0 && i > 0) {
        model_->EvaluateErr();
        emit progressChanged_(p_bar_increment, 100 * i / max_count);
      }
      if (i % num_images_ == 0 && i > 0) {
        model_->EvaluateErr();
        error_data_vector_[batches_count++] = model_->GetErr();
        emit progressChanged_(p_bar_increment, 100 * i / max_count);
        model_->ResetErr();
      }
    }
    error_data_vector_[batches_count++] = model_->GetErr();
  } else {
    bool teach_on = false;
    long teach_count = 0;
    long eval_count = 0;
    long max_count = (1 + num_batches_) * num_images_;
    error_data_vector_.resize(num_batches_);
    for (unsigned int i = 0; (i < max_count) && !stop_; i++) {
      if (teach_on) {
        model_->TeachNetwork();
        teach_count++;
      } else {
        eval_count++;
      }
      if (eval_count >= num_images_ / num_batches_) {
        eval_count = 0;
        teach_on = true;
        model_->EvaluateErr();
        error_data_vector_[batches_count++] = model_->GetErr();
        emit progressChanged_(0, 100 * i / max_count);
        model_->ResetErr();
      }
      if (teach_count >= num_images_) {
        teach_count = 0;
        teach_on = false;
        model_->ResetErr();
      }
      LoadNextDataset();
      if (i % p_bar_increment == 0 && i > 0) {
        model_->EvaluateErr();
        emit progressChanged_(p_bar_increment, 100 * i / max_count);
      }
    }
  }
  stop_ = true;
  emit progressChanged_(p_bar_increment, 100);
}

void Controller::TestNetwork(unsigned int percent) {
  unsigned int const &num_images_ = model_->GetCountOfElements();
  auto num_test_images = num_images_ * percent / 100;
  model_->ResetErr();
  if (!model_->GetInputValues().empty()) {
    model_->Activate(model_->GetInputValues());
    unsigned int i;
    for (i = 1; (i < num_test_images) && !stop_; i++) {
      LoadNextDataset();
      if (i % 100 == 0) {
        model_->EvaluateErr();
        emit progressTestChanged_(100, 100 * i / num_test_images);
      }
    }
    if (i == num_test_images) {
      stop_ = true;
      emit progressTestChanged_(100, 100);
    }
  }
}

void Controller::ResetNetworkConfiguration() {
  if (!load_) model_->ResetNetworkConfiguration();
  load_ = false;
}
