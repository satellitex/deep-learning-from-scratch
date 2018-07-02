//
// Created by TakumiYamashita on 2018/07/01.
//

#include "../src/trainer/trainer.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <iostream>
#include "../src/layer/layer.hpp"
#include "../src/network/builder.hpp"
#include "../src/network/network.hpp"
#include "../src/optimizer/optimizer.hpp"
#include "../src/primitive/primitive.hpp"

using namespace dpl;

TEST(TRAINER_TEST, TRAIN) {
  constexpr int TRAIN_NUM = 20;
  constexpr int TEACH_NUM = 10;
  constexpr int BATCH_NUM = 2;
  constexpr int C = 1;
  constexpr int H = 9;
  constexpr int W = 9;
  constexpr int M = 10;
  auto network = NetworkBuilder<BATCH_NUM>::Input<C, H, W>()
                     .Convolution<8, 3, 3, 1, 1>()
                     .Relu()
                     .Pooling<2, 2, 2>()
                     .Affine<M>()
                     .Dropout(0.5)
                     .SoftmaxWithLoss()
                     .buildPtr();
  auto optimizer = SGD(0.1);
  auto x_train = make_ndarray_ptr<float, TRAIN_NUM, C, H, W>();
  x_train->rand();
  auto x_test = make_ndarray_ptr<float, TRAIN_NUM, M>();
  x_test->rand();

  auto t_train = make_ndarray_ptr<float, TEACH_NUM, C, H, W>();
  t_train->rand();
  auto t_test = make_ndarray_ptr<float, TEACH_NUM, M>();
  t_test->rand();

  auto trainer = Trainer<BATCH_NUM, 2, decltype(network), decltype(optimizer),
                         decltype(x_train), decltype(x_test), decltype(t_train),
                         decltype(t_test)>(network, optimizer, x_train, x_test,
                                           t_train, t_test, 2);
  trainer.train();
}

TEST(TRSINER_TEST, XOR) {
  constexpr int TRAIN_NUM = 4;
  constexpr int TEACH_NUM = 4;
  constexpr int BATCH_NUM = 1;
  constexpr int N = 2;
  constexpr int M = 2;

  auto network = NetworkBuilder<BATCH_NUM>::Input<N>()
                     .Affine<M>()
                     .Relu()
                     .Affine<M>()
                     .SoftmaxWithLoss()
                     .buildPtr();
  auto optimizer = SGD(0.0001);
  auto x_train = make_ndarray_ptr<float, TRAIN_NUM, N>();
  *x_train << 0, 0, 0, 1, 1, 0, 1, 1;

  auto x_label = make_ndarray_ptr<float, TRAIN_NUM, M>();
  *x_label << 1, 0, 0, 1, 0, 1, 1, 0;

  auto t_train = make_ndarray_ptr<float, TRAIN_NUM, N>();
  *t_train << 0, 0, 0, 1, 1, 0, 1, 1;

  auto t_label = make_ndarray_ptr<float, TRAIN_NUM, M>();
  *t_label << 1, 0, 0, 1, 0, 1, 1, 0;

  auto trainer = Trainer<BATCH_NUM, 4, decltype(network), decltype(optimizer),
                         decltype(x_train), decltype(x_label),
                         decltype(t_train), decltype(t_label)>(
      network, optimizer, x_train, x_label, t_train, t_label, 500);

  trainer.train();
}