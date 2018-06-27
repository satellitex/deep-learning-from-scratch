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

    using output = ndarray<Type, Dims...>;
  };

  template <typename Type, int N, int K, int... Dims>
  class Affine {
   public:
    struct M {
      enum { value = GetFact<sizeof...(Dims) - 1, Dims...>::value };
    };

    Affine() {
      w.rand();
      w = w * (Type)sqrt(2.0 / N);
      b.fill(0);
    }
    ndarray<Type, N, K> forward(const ndarray<Type, N, Dims...>& input) {
      x = input.template reshape<N, M::value>();
      ndarray<Type, N, K> ret = dot(x, w);
      for (int i = 0; i < N; i++)
        for (int j = 0; j < K; j++) ret.at(i, j) = ret.at(i, j) + b.at(j);
      return std::move(ret);
    }

    ndarray<Type, N, Dims...> backward(const ndarray<Type, N, K>& dout) {
      ndarray<Type, N, M::value> ret = dot(dout, w.T());
      dw = dot(x.T(), dout);
      db = dout.template sum<0>();
      return ret.template reshape<N, Dims...>();
    };

    const ndarray<Type, M::value, K>& getDw() const { return dw; };
    const ndarray<Type, K>& getDb() const { return db; };

    using output = ndarray<Type, N, K>;
   private:
    ndarray<Type, N, M::value> x;
    ndarray<Type, M::value, K> w;
    ndarray<Type, K> b;

    ndarray<Type, M::value, K> dw;
    ndarray<Type, K> db;
  };

  template <typename Type, int... Dims>
  class Dropout {
   public:
    Dropout(float dropout_ratio) : dropout_ratio(dropout_ratio) {}

    ndarray<Type, Dims...> forward(const ndarray<Type, Dims...>& input,
                                   bool train_flag = true) {
      if (train_flag) return input * (float)(1.0 - dropout_ratio);
      ndarray<Type, Dims...> rnd;
      rnd.rand();
      for (int i = 0; i < rnd.size(); i++) {
        if (rnd.linerAt(i) > dropout_ratio)
          mask.linerAt(i) = 1.0;
        else
          mask.linerAt(i) = 0.0;
      }
      return input * mask;
    }
    ndarray<Type, Dims...> backward(const ndarray<Type, Dims...>& dout) {
      return dout * mask;
    }

    using output = ndarray<Type, Dims...>;
   private:
    float dropout_ratio;
    ndarray<float, Dims...> mask;
  };

  template <typename Type, int N, int C, int H, int W, int FILTER_N,
            int FILTER_H, int FILTER_W, int STRIDE, int PAD>
  class Convolution {
   private:
    struct OUT_H {
      enum { value = (H + 2 * PAD - FILTER_H) / STRIDE + 1 };
    };
    struct OUT_W {
      enum { value = (W + 2 * PAD - FILTER_W) / STRIDE + 1 };
    };

   public:
    Convolution() {
      w.rand();
      w = w * (Type)sqrt(2.0 / N);
      b.fill(0);
    }

    ndarray<Type, N, FILTER_N, OUT_H::value, OUT_W::value> forward(
        const ndarray<Type, N, C, H, W>& input) {
      auto col = input.template im2col<FILTER_H, FILTER_W, STRIDE, PAD>();
      col_w = w.template reshape<FILTER_N, C * FILTER_H * FILTER_W>().T();

      auto out = dot(col, col_w);
      for (int i = 0; i < N * OUT_H::value * OUT_W::value; i++)
        for (int l = 0; l < FILTER_N; l++)
          out.at(i, l) = out.at(i, l) + b.at(l);
      ndarray<Type, N, FILTER_N, OUT_H::value, OUT_W::value> ret =
          out.template reshape<N, OUT_H::value, OUT_W::value, FILTER_N>()
              .template transpose<0, 3, 1, 2>();
      return std::move(ret);
    }

    ndarray<Type, N, C, H, W> backward(
        const ndarray<Type, N, FILTER_N, OUT_H::value, OUT_W::value>& dout) {
      auto out =
          dout.template transpose<0, 2, 3, 1>()
              .template reshape<N * OUT_H::value * OUT_W::value, FILTER_N>();
      db = out.template sum<0>();
      auto tdw = dot(col.T(), out);
      dw = tdw.template transpose<1, 0>()
               .template reshape<FILTER_N, C, FILTER_H, FILTER_W>();

      auto dcol = dot(out, col_w.T());
      ndarray<Type, N, C, OUT_H::value, OUT_W::value> ret =
          dcol.template col2im<N, C, H, W, FILTER_H, FILTER_W, STRIDE, PAD>();
      return std::move(ret);
    };

    using output = ndarray<Type, N, FILTER_N, OUT_H::value, OUT_W::value>;
   private:
    ndarray<Type, N, C, H, W> x;

    ndarray<Type, FILTER_N, C, FILTER_H, FILTER_W> w;
    ndarray<Type, FILTER_N> b;

    ndarray<Type, N * OUT_H::value * OUT_W::value, C * FILTER_H * FILTER_W> col;
    ndarray<Type, C * FILTER_H * FILTER_W, FILTER_N> col_w;

    ndarray<Type, FILTER_N> db;
    ndarray<Type, FILTER_N, C, FILTER_H, FILTER_W> dw;
  };

  template <typename Type, int N, int C, int H, int W, int POOL_H, int POOL_W,
            int STRIDE>
  class Pooling {
   private:
    struct OUT_H {
      enum { value = (H - POOL_H) / STRIDE + 1 };
    };
    struct OUT_W {
      enum { value = (W - POOL_W) / STRIDE + 1 };
    };

   public:
    ndarray<Type, N, C, OUT_H::value, OUT_W::value> forward(
        const ndarray<Type, N, C, H, W>& input) {
      x = input;
      auto col_t = x.template im2col<POOL_H, POOL_W, STRIDE, 0>();
      auto col = col_t.template reshape<N * OUT_H::value * OUT_W::value * C,
                                        POOL_H * POOL_W>();

      arg_max = col.template argmax<1>();
      auto out = col.template max<1>();
      ndarray<Type, N, C, OUT_H::value, OUT_W::value> ret =
          out.template reshape<N, OUT_H::value, OUT_W::value, C>()
              .template transpose<0, 3, 1, 2>();
      return std::move(ret);
    }

    ndarray<Type, N, C, H, W> backward(
        const ndarray<Type, N, C, OUT_H::value, OUT_W::value>& dout) {
      auto out = dout.template transpose<0, 2, 3, 1>();

      ndarray<Type, N * OUT_H::value * OUT_W::value * C, POOL_H * POOL_W> dmax;
      dmax.fill(0);
      for (int i = 0; i < arg_max.size(); i++) {
        dmax[i][arg_max.at(i)] = dout.linerAt(i);
      }
      auto dcol = dmax.template reshape<N * OUT_H::value * OUT_W::value,
                                        C * POOL_H * POOL_W>();
      ndarray<Type, N, C, H, W> dx =
          dcol.template col2im<N, C, H, W, POOL_H, POOL_W, STRIDE, 0>();
      return std::move(dx);
    };

    using output = ndarray<Type, N, C, OUT_H::value, OUT_W::value>;
   private:
    ndarray<Type, N, C, H, W> x;
    ndarray<unsigned, N * OUT_H::value * OUT_W::value * C> arg_max;
  };

  template <typename Type, int N, int M>
  class SoftmaxWithLoss {
   public:
    SoftmaxWithLoss() {}

    Type forward(const ndarray<Type, N, M>& input,
                 const ndarray<Type, N, M>& teacher) {
      y = softmax(input);
      t = teacher;

      Type loss = cross_entropy_error(y, teacher);
      return loss;
    };

    ndarray<Type, N, M> backward(const Type dout = (Type)1) {
      // TODO : now only one-hot-expression
      ndarray<Type, N, M> dx = (y - t) / (Type)N;
      return std::move(dx);
    };

    using output = Type;
   private:
    ndarray<Type, N, M> y;
    ndarray<Type, N, M> t;
  };

};  // namespace dpl

#endif  // DEEP_LEARNING_FROM_SCRATCH_LAYER_HPP
