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

  auto in = make_ndarray_ptr<float, 100, 100>();
  for (int i = 0; i < in->size(); i++) in->linerAt(i) = -50 * 50 + i;

  auto out = relu.forward(*in);
  for (int i = 0; i < out->size(); i++) {
    ASSERT_FLOAT_EQ(std::max<float>(0, in->linerAt(i)), out->linerAt(i));
  }

  auto dx = relu.backward(*out);
  for (int i = 0; i < dx->size(); i++) {
    ASSERT_FLOAT_EQ(out->linerAt(i) > 0.0 ? 1 : 0, dx->linerAt(i));
  }
}

TEST(LAYER_TEST, AFFINE) {
  Affine<float, 100, 40, 20, 10> affine;

  auto in = make_ndarray_ptr<float, 100, 20, 10>();
  ndarrayPtr<float, 100, 40> out = affine.forward(*in);
  ndarrayPtr<float, 100, 20, 10> dx = affine.backward(*out);
}

TEST(LAYER_TEST, DROPOUT) {
  Dropout<float, 100, 20, 10> dropout;
  dropout.set_dropout_ratio(0.5);

  auto in = make_ndarray_ptr<float, 100, 20, 10>();
  ndarrayPtr<float, 100, 20, 10> out = dropout.forward(*in);
  ndarrayPtr<float, 100, 20, 10> dx = dropout.backward(*out);

  //  ndarrayPtr<float, 100, 20, 10> out_f = dropout.forward(*in, false);
  //  ndarrayPtr<float, 100, 20, 10> dx_f = dropout.backward(*out_f);
}

TEST(LAYER_TEST, CONVOLUTION) {
  Convolution<float, 2, 3, 28, 28, 16, 3, 3, 1, 1> conv;

  auto in = make_ndarray_ptr<float, 2, 3, 28, 28>();
  constexpr int OUT_H = (28 + 2 * 1 - 3) / 1 + 1;
  constexpr int OUT_W = (28 + 2 * 1 - 3) / 1 + 1;
  ndarrayPtr<float, 2, 16, OUT_H, OUT_W> out = conv.forward(*in);
  ndarrayPtr<float, 2, 3, 28, 28> dx = conv.backward(*out);
}

TEST(LAYER_TEST, POOLING) {
  Pooling<float, 2, 3, 28, 28, 2, 2, 2> pooling;

  auto in = make_ndarray_ptr<float, 2, 3, 28, 28>();
  constexpr int OUT_H = (28 - 2) / 2 + 1;
  constexpr int OUT_W = (28 - 2) / 2 + 1;

  ndarrayPtr<float, 2, 3, OUT_H, OUT_W> out = pooling.forward(*in);
  ndarrayPtr<float, 2, 3, 28, 28> dx = pooling.backward(*out);
}

TEST(LAYER_TEST, SOFT_MAX) {
  SoftmaxWithLoss<float, 2, 10> last_layer;

  auto input = make_ndarray_ptr<float, 2, 10>();
  auto teacher = make_ndarray_ptr<float, 2, 10>();
  float loss = last_layer.forward(*input, *teacher);
  ndarrayPtr<float, 2, 10> dx = last_layer.backward();
}