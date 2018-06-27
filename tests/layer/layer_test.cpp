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
  Affine<float, 100, 40, 20, 10> affine;

  ndarray<float, 100, 20, 10> in;
  ndarray<float, 100, 40> out = affine.forward(in);
  ndarray<float, 100, 20, 10> dx = affine.backward(out);
  ndarray<float, 200, 40> dw = affine.getDw();
  ndarray<float, 40> db = affine.getDb();
}

TEST(LAYER_TEST, DROPOUT) {
  Dropout<float, 100, 20, 10> dropout(0.5);

  ndarray<float, 100, 20, 10> in;
  ndarray<float, 100, 20, 10> out = dropout.forward(in);
  ndarray<float, 100, 20, 10> dx = dropout.backward(out);

  ndarray<float, 100, 20, 10> out_f = dropout.forward(in, false);
  ndarray<float, 100, 20, 10> dx_f = dropout.backward(out_f);
}

TEST(LAYER_TEST, CONVOLUTION) {
  Convolution<float, 2, 3, 28, 28, 16, 3, 3, 1, 1> conv;

  ndarray<float, 2, 3, 28, 28> in;
  constexpr int OUT_H = (28 + 2 * 1 - 3) / 1 + 1;
  constexpr int OUT_W = (28 + 2 * 1 - 3) / 1 + 1;
  ndarray<float, 2, 16, OUT_H, OUT_W> out = conv.forward(in);
  ndarray<float, 2, 3, 28, 28> dx = conv.backward(out);
}

TEST(LAYER_TEST, POOLING) {
  Pooling<float, 2, 3, 28, 28, 2, 2, 2> pooling;

  ndarray<float, 2, 3, 28, 28> in;
  constexpr int OUT_H = (28 - 2) / 2 + 1;
  constexpr int OUT_W = (28 - 2) / 2 + 1;

  ndarray<float, 2, 3, OUT_H, OUT_W> out = pooling.forward(in);
  ndarray<float, 2, 3, 28, 28> dx = pooling.backward(out);
}

TEST(LAYER_TEST, SOFT_MAX) {
  SoftmaxWithLoss<float, 2, 10> last_layer;

  ndarray<float, 2, 10> input;
  ndarray<float, 2, 10> teacher;
  float loss = last_layer.forward(input, teacher);
  ndarray<float, 2, 10> dx = last_layer.backward();
}