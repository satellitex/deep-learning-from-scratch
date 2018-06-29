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
  auto network = NetworkBuilder<1>::Input<1, 9, 9>()
                     .Convolution<8, 3, 3, 1, 1>()
                     .Relu()
                     .Pooling<2, 2, 2>()
                     .Affine<10>()
                     .Dropout(0.5)
                     .SoftmaxWithLoss()
                     .build();

  auto input = make_ndarray_ptr<float, 1, 1, 9, 9>();
  auto teacher = make_ndarray_ptr<float, 1, 10>();
  input->rand();
  teacher->rand();

  network.gradient(*input, *teacher);

  auto cw = *(network.getLayer().w);
  auto cb = *(network.getLayer().b);
  auto cdw = *(network.getLayer().dw);
  auto cdb = *(network.getLayer().db);
  
  auto excw = (cw - *(cdw * (float)0.1));
  auto excb = (cb - *(cdb * (float)0.1));

  auto& affine_layer = network.next().next().next().getLayer();
  auto aw = *(affine_layer.w);
  auto ab = *(affine_layer.b);

  auto adw = *(affine_layer.dw);
  auto adb = *(affine_layer.db);

  auto exaw = (aw - *(adw * (float)0.1));
  auto exab = (ab - *(adb * (float)0.1));

  SGD sgd(0.1);
  sgd.update(network);

  cw = *(network.getLayer().w);
  cb = *(network.getLayer().b);
  aw = *(affine_layer.w);
  ab = *(affine_layer.b);

  ASSERT_EQ(cw, *excw);
  ASSERT_EQ(cb, *excb);

  ASSERT_EQ(aw, *exaw);
  ASSERT_EQ(ab, *exab);
}