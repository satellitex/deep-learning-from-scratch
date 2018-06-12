//
// Created by TakumiYamashita on 2018/06/10.
//

#include <opencv2/core/core.hpp>
#include <iostream>
#include "trainer/trainer.h"

int main() {
  dpl::ndarray nd;
  // 64F, channels=10, 3x3 の2次元配列（行列）
  cv::Mat m1(3, 3, CV_64FC(10));
  cv::Mat m2(3, 3, CV_MAKETYPE(CV_64F, 10));

  CV_Assert(m1.type() == m2.type());
  std::cout << "mat1/mat2"<< std::endl;
  std::cout << "  dims: " << m1.dims << ", depth(byte/channel):" << m1.elemSize1() \
	    << ", channels: " << m1.channels() << std::endl;

  std::cout << "boot :: deep learning lib" << std::endl;
}