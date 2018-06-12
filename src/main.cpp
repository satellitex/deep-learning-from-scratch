//
// Created by TakumiYamashita on 2018/06/10.
//

#include <Eigen/Core>
#include <iostream>
#include "trainer/trainer.h"

int main() {
  Eigen::MatrixXf A = Eigen::MatrixXf::Zero(2, 2);
  A(0, 0) = 2;
  A(1, 1) = 5;

  std::cout << A << std::endl;

  dpl::ndarray nd;
  std::cout << "boot :: deep learning lib" << std::endl;
}