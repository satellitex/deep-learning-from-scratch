//
// Created by TakumiYamashita on 2018/06/10.
//

#ifndef DEEP_LEARNING_FROM_SCRATCH_OPTIMIZER_HPP
#define DEEP_LEARNING_FROM_SCRATCH_OPTIMIZER_HPP

#include "../network/network.hpp"
#include "../primitive/primitive.hpp"

namespace dpl {
  class SGD {
   public:
    SGD() : lr(0.1) {}
    SGD(float lr) : lr(lr) {}
    template <class First, class... Layers>
    void update(Network<First, Layers...>& network) {
      network.getLayer().update(
          [this](auto& a, auto& b) { *a = *(*a - *(*b * lr)); });
      auto nextNetwork = network.next();
      update(nextNetwork);
    }

    void update(Network<>& network) {
      return;
    }

   private:
    float lr;
  };
}  // namespace dpl

#endif  // DEEP_LEARNING_FROM_SCRATCH_OPTIMIZER_HPP
