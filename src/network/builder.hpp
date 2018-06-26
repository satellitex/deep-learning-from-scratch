//
// Created by TakumiYamashita on 2018/06/10.
//

#ifndef DEEP_LEARNING_FROM_SCRATCH_BUILDER_HPP
#define DEEP_LEARNING_FROM_SCRATCH_BUILDER_HPP

#include "network.hpp"

namespace dpl {

  template<typename... Layers>
  class NetworkBuilder {
   private:
    Network network;
  };

  // use
  int main() {
    NetworkBuilder< Convolution<1,1,3,3>::type,
        Convolution
    network.Convolution<1,1,3,3>().
        Relu<>().
            Convolution
  }
}  // namespace dpl

#endif  // DEEP_LEARNING_FROM_SCRATCH_BUILDER_HPP
