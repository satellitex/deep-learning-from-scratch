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
}