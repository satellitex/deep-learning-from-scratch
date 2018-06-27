//
// Created by TakumiYamashita on 2018/06/26.
//

#include "../src/network/network.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <iostream>
#include "../src/layer/layer.hpp"
#include "../src/network/builder.hpp"
#include "../src/primitive/primitive.hpp"

using namespace dpl;

TEST(NETWORK_TEST, PRDICT) {
  NetworkBuilder builder;
  auto builder_ = builder.Input<50, 1, 28, 28>()
                      .Convolution<16, 3, 3, 1, 1>()
                      .Relu()
                      .Convolution<16, 3, 3, 1, 1>()
                      .Relu()
                      .Pooling<2, 2, 2>()
                      .Affine<50>()
                      .Relu()
                      .Dropout()
                      .Affine<10>()
                      .Dropout()
                      .SoftmaxWithLoss();
}