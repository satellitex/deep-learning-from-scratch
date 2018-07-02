//
// Created by TakumiYamashita on 2018/06/10.
//

#include <algorithm>
#include <iostream>
#include "layer/layer.hpp"
#include "loader/mnist.hpp"
#include "network/builder.hpp"
#include "network/network.hpp"
#include "optimizer/optimizer.hpp"
#include "primitive/primitive.hpp"
#include "trainer/trainer.hpp"

using namespace dpl;

int main() {
  constexpr int BATCH_NUM = 100;
  constexpr int C = 1;
  constexpr int H = 28;
  constexpr int W = 28;
  constexpr int M = 10;

  std::cout << "sample :: deep learning lib" << std::endl;

  MNISTLoader mnistLoader;
  mnistLoader.load();
  auto t_train = mnistLoader.getTestImage();
  auto x_train = mnistLoader.getTrainImage();
  auto t_label = mnistLoader.getTestLabel();
  auto x_label = mnistLoader.getTrainLabel();

  auto network = NetworkBuilder<BATCH_NUM>::Input<C, H, W>()
                     .Convolution<16, 3, 3, 1, 1>()
                     .Relu()
                     .Convolution<16, 3, 3, 1, 1>()
                     .Relu()
                     .Pooling<2, 2, 2>()
                     .Convolution<32, 3, 3, 1, 1>()
                     .Relu()
                     .Convolution<32, 3, 3, 1, 1>()
                     .Relu()
                     .Pooling<2, 2, 2>()
                     .Convolution<64, 3, 3, 1, 1>()
                     .Relu()
                     .Convolution<64, 3, 3, 1, 1>()
                     .Relu()
                     .Pooling<2, 2, 2>()
                     .Affine<50>()
                     .Relu()
                     .Dropout(0.5)
                     .Affine<10>()
                     .Dropout(0.5)
                     .SoftmaxWithLoss()
                     .buildPtr();
  auto optimizer = SGD(0.001);

  auto trainer =
      Trainer<BATCH_NUM, 1000, decltype(network), decltype(optimizer),
              decltype(x_train), decltype(x_label), decltype(t_train),
              decltype(t_label)>(network, optimizer, x_train, x_label, t_train,
                                 t_label, 20);
  trainer.train();
}