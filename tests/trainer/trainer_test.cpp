//
// Created by TakumiYamashita on 2018/07/01.
//

#include <gtest/gtest.h>
#include <algorithm>
#include <iostream>
#include "../src/layer/layer.hpp"
#include "../src/network/builder.hpp"
#include "../src/network/network.hpp"
#include "../src/optimizer/optimizer.hpp"
#include "../src/primitive/primitive.hpp"
#include "../src/trainer/trainer.hpp"

using namespace dpl;

TEST(TRAINER_TEST, TRAIN) {
  constexpr int TRAIN_NUM = 20;
  constexpr int TEACH_NUM = 10;
  constexpr int BATCH_NUM = 2;
  constexpr int C = 1;
  constexpr int H = 9;
  constexpr int W = 9;
  constexpr int M = 10;
//  auto network = NetworkBuilder<BATCH_NUM>::Input<C, H, W>()
//                     .Convolution<8, 3, 3, 1, 1>()
//                     .Relu()
//                     .Pooling<2, 2, 2>()
//                     .Affine<M>()
//                     .Dropout(0.5)
//                     .SoftmaxWithLoss()
//                     .build();
//  auto optimizer = SGD(0.1);
//  auto x_train = make_ndarray_ptr<float, TRAIN_NUM, C, H, W>();
//  auto x_test = make_ndarray_ptr<float, TRAIN_NUM, M>();
//
//  auto t_train = make_ndarray_ptr<float, TEACH_NUM, C, H, W>();
//  auto t_test = make_ndarray_ptr<float, TEACH_NUM, M>();
//
//  auto trainer = Trainer<BATCH_NUM, 2, decltype(network), decltype(optimizer),
//                         decltype(x_train), decltype(x_test), decltype(t_train),
//                         decltype(t_test)>(network, optimizer, x_train, x_test,
//                                           t_train, t_test, 2);
//  trainer.train();
}