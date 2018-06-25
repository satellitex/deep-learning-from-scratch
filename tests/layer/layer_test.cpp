//
// Created by TakumiYamashita on 2018/06/25.
//

#include "../src/layer/layer.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <iostream>
#include "../src/primitive/primitive.hpp"

using namespace dpl;

TEST(LAYER_TEST, RELU) {
  ndarray<float, 100, 100> in;
  for (int i = 0; i < in.size(); i++) in.linerAt(i) = -50 * 50 + i;

  Relu<float, 100, 100> relu;

  auto out = relu.forward(in);
  for (int i = 0; i < out.size(); i++) {
    ASSERT_FLOAT_EQ(std::max<float>(0, in.linerAt(i)), out.linerAt(i));
  }

  auto dx = relu.backward(out);
  for (int i = 0; i < dx.size(); i++) {
    ASSERT_FLOAT_EQ(out.linerAt(i) > 0.0 ? 1 : 0, dx.linerAt(i));
  }
}
