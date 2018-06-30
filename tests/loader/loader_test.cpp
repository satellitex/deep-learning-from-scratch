//
// Created by TakumiYamashita on 2018/06/25.
//

#include <gtest/gtest.h>
#include <algorithm>
#include <iostream>
#include "../src/config.hpp"
#include "../src/layer/layer.hpp"
#include "../src/loader/mnist.hpp"
#include "../src/primitive/primitive.hpp"

using namespace dpl;

TEST(LOADER_TEST, MNITS) {
  MNISTLoader mnistLoader;
  mnistLoader.download();
}

TEST(LOADER_TEST, MNIST) {
  MNISTLoader mnistLoader;
  mnistLoader.load();
  auto& test_img = mnistLoader.getTestImage();
  auto& train_img = mnistLoader.getTrainImage();
  auto& test_label = mnistLoader.getTestLabel();
  auto& train_label = mnistLoader.getTrainLabel();

  // sampling test img test
  ndarray<float, 28> test_ex_0_0_10;
  test_ex_0_0_10 << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0666667, 0.258824,
      0.054902, 0.262745, 0.262745, 0.262745, 0.231373, 0.0823529, 0.92549,
      0.996078, 0.415686, 0, 0, 0, 0, 0, 0;
  ASSERT_TRUE(nearly(test_ex_0_0_10, test_img->at(0, 0, 10), (float)1e-6));

  // sampling test label test
  ndarray<float, 3, 10> test_label_ex_100_103;
  test_label_ex_100_103 << 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0;
  ASSERT_EQ(test_label_ex_100_103, *(test_label->slice<0, 100, 103, 1>()));

  // sampling train img test
  ndarray<float, 28> train_ex_59999_0_10;
  train_ex_59999_0_10 << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0980392, 0.85098,
      0.94902, 0.360784, 0.0156863, 0, 0, 0, 0, 0.0156863, 0.576471, 0.992157,
      0.941176, 0.909804, 0.360784, 0, 0, 0;
  ASSERT_TRUE(
      nearly(train_ex_59999_0_10, train_img->at(59999, 0, 10), (float)1e-6));

  // sampling train label test
  ndarray<float, 3, 10> train_label_ex_59990_59993;
  train_label_ex_59990_59993 << 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;
  ASSERT_EQ(train_label_ex_59990_59993,
            *(train_label->slice<0, 59990, 59993, 1>()));
}