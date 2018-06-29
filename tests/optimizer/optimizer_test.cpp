//
// Created by TakumiYamashita on 2018/06/29.
//

#include "../src/optimizer/optimizer.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <iostream>
#include "../src/layer/layer.hpp"
#include "../src/network/builder.hpp"
#include "../src/network/network.hpp"
#include "../src/primitive/primitive.hpp"

using namespace dpl;

TEST(OPTIMIZER_TEST, SGD) {
  auto network = NetworkBuilder<2>::Input<1, 28, 28>()
                     .Convolution<16, 3, 3, 1, 1>()
                     .Relu()
                     .Pooling<2, 2, 2>()
                     .Affine<10>()
                     .Dropout(0.5)
                     .SoftmaxWithLoss()
                     .build();

  auto input = make_ndarray_ptr<float, 2, 1, 28, 28>();
  network.predict(*input);
  SGD sgd(0.1);
  sgd.update(network);
}