//
// Created by TakumiYamashita on 2018/06/10.
//

#ifndef DEEP_LEARNING_FROM_SCRATCH_LAYER_HPP
#define DEEP_LEARNING_FROM_SCRATCH_LAYER_HPP

#include "../primitive/primitive.hpp"

namespace dpl {

  template <typename Type, int... Dims>
  class Relu {
   public:
    void forward(const ndarray<Type, Dims...>& input,
                 ndarray<Type, Dims...>& output) {
      for (int i = 0; i < input.size(); i++) {
        if (input.linerAt(i) >= 0)
          output.linerAt(i) = input.linerAt(i);
        else
          output.linerAt(i) = 0;
      }
    }

    void backward(const ndarray<Type, Dims...>& dout,
                  ndarray<Type, Dims...>& dx) {
      for (int i = 0; i < dout.size(); i++)
        dx.linerAt(i) = dout.linerAt(i) > 0 ? 1 : 0;
    }
  };

};  // namespace dpl

#endif  // DEEP_LEARNING_FROM_SCRATCH_LAYER_HPP
