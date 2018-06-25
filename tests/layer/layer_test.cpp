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
  Relu<float, 100, 100> relu;

  ndarray<float, 100, 100> in;
  for (int i = 0; i < in.size(); i++) in.linerAt(i) = -50 * 50 + i;

  auto out = relu.forward(in);
  for (int i = 0; i < out.size(); i++) {
    ASSERT_FLOAT_EQ(std::max<float>(0, in.linerAt(i)), out.linerAt(i));
  }

  auto dx = relu.backward(out);
  for (int i = 0; i < dx.size(); i++) {
    ASSERT_FLOAT_EQ(out.linerAt(i) > 0.0 ? 1 : 0, dx.linerAt(i));
  }
}

TEST(LAYER_TEST, AFFINE) {
  Affine<float, 100, 40, 60> affine;

  ndarray<float, 100, 40> in;
  auto out = affine.forward(in);
  auto dx = affine.backward(out);
  affine.getDw();
  affine.getDb();
}

TEST(LAYER_TEST, DROPOUT) {
  Dropout<float, 100, 20, 10> dropout(0.5);

  ndarray<float, 100, 20, 10> in;
  auto out = dropout.forward(in);
  auto dx = dropout.backward(out);

  auto out_f = dropout.forward(in, false);
  auto dx_f = dropout.backward(out_f);
}