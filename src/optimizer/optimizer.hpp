//
// Created by TakumiYamashita on 2018/06/10.
//

#ifndef DEEP_LEARNING_FROM_SCRATCH_OPTIMIZER_HPP
#define DEEP_LEARNING_FROM_SCRATCH_OPTIMIZER_HPP

#include "../primitive/primitive.hpp"

using ndarray = dpl::ndarray;

namespace dpl {
  class SGD {
    SGD();
    SGD(float lr) : lr(lr) {}
    template <class First, class... Layers>
    void update(Network<First, Layers...>& network) {
      network.getLayer().update(
          [lr](ndarrayPtr<float, auto...>& a, ndarrayPtr<float, auto...>& b) {
            *a = *(*a - *(*b * lr));
          });
    }
   private:
    float lr;
  };
}  // namespace dpl

#endif  // DEEP_LEARNING_FROM_SCRATCH_OPTIMIZER_HPP
