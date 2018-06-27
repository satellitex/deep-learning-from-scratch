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
  auto network = NetworkBuilder<2>::Input<1, 28, 28>()
                     .Convolution<16, 3, 3, 1, 1>()
                     .Relu()
                     .Convolution<16, 3, 3, 1, 1>()
                     .Relu()
                     .Pooling<2, 2, 2>()
                     .Affine<50>()
                     .Relu()
                     .Dropout(0.5)
                     .Affine<10>()
                     .Dropout(0.5)
                     .SoftmaxWithLoss()
                     .build();

  ndarray<float, 2, 1, 28, 28> input;
  input.rand();

  ndarray<float, 2, 10> teacher;
  teacher.fill(0);
  teacher.at(0).at(0) = 1;
  teacher.at(1).at(9) = 1;

  float v = network.predict(input, teacher, true);
}