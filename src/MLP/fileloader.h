#ifndef SRC_MLP_FILELOADER_H_
#define SRC_MLP_FILELOADER_H_

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

namespace s21 {
class FileLoader {
 public:
  FileLoader() {}
  ~FileLoader() {}
  bool SetFileStream(std::string filename);
  std::vector<double> GetOutputValues();
  std::vector<double> GetInputValues();
  long GetCountOfElements();
  bool ReadElement();
  void StartReadElements();
  void SetPosition(int pos);  // нумерация строк с нуля

 private:
  std::ifstream filestream_;
  std::vector<double> output_values_;
  std::vector<double> input_values_;
  long count_of_elements_ = 0;
  void SetOutputValues(int value);
  void ClearData();
  std::string GetLine();
  std::string GetToken(std::string const& line, size_t pos);
  size_t FindSeparatorPosition(const std::string& line);
  void EraseToken(std::string& line, size_t pos);
  void AddValueToOutputVector(const std::string& value);
  void ReadCountOfElements();
};
};      // namespace s21
#endif  // SRC_MLP_FILELOADER_H_
