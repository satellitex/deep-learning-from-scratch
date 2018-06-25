//
// Created by TakumiYamashita on 2018/06/10.
//

#ifndef DEEP_LEARNING_FROM_SCRATCH_LAYER_HPP
#define DEEP_LEARNING_FROM_SCRATCH_LAYER_HPP

#include <cmath>
#include "../primitive/primitive.hpp"

namespace dpl {

  template <typename Type, int... Dims>
  class Relu {
   public:
    ndarray<Type, Dims...> forward(const ndarray<Type, Dims...>& input) {
      ndarray<Type, Dims...> ret;
      for (int i = 0; i < input.size(); i++) {
        if (input.linerAt(i) >= 0)
          ret.linerAt(i) = input.linerAt(i);
        else
          ret.linerAt(i) = 0;
      }
      return std::move(ret);
    }

    ndarray<Type, Dims...> backward(const ndarray<Type, Dims...>& dout) {
      ndarray<Type, Dims...> ret;
      for (int i = 0; i < dout.size(); i++)
        ret.linerAt(i) = dout.linerAt(i) > 0 ? 1 : 0;
      return std::move(ret);
    }
  };

  template <typename Type, int N, int M, int K>
  class Affine {
   public:
    Affine() {
      w.rand();
      w = w * (Type)sqrt(2.0 / N);
      b.fill(0);
    }
    ndarray<Type, N, K> forward(const ndarray<Type, N, M>& input) {
      x = input;
      ndarray<Type, N, K> mult = dot(input, w);
      ndarray<Type, N, K> ret = mult + b;
      return std::move(ret);
    }

    ndarray<Type, N, M> backward(const ndarray<Type, N, K>& dout) {
      ndarray<Type, N, M> ret = dot(dout, w.T());

      dw = dot(x.T(), dout);
      db = dout.template sum<0>();
      return std::move(ret);
    };

    const ndarray<Type, M, K>& getDw() const { return dw; };
    const ndarray<Type, K>& getDb() const { return db; };

   private:
    ndarray<Type, N, M> x;
    ndarray<Type, M, K> w;
    ndarray<Type, N, K> b;

    ndarray<Type, M, K> dw;
    ndarray<Type, K> db;
  };

};  // namespace dpl

#endif  // DEEP_LEARNING_FROM_SCRATCH_LAYER_HPP
