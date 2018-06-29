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
    ndarrayPtr<Type, Dims...> forward(const ndarray<Type, Dims...>& input) {
      auto ret = make_ndarray_ptr<Type, Dims...>();
      for (int i = 0; i < input.size(); i++) {
        if (input.linerAt(i) >= 0)
          ret->linerAt(i) = input.linerAt(i);
        else
          ret->linerAt(i) = 0;
      }
      return std::move(ret);
    }

    ndarrayPtr<Type, Dims...> backward(const ndarray<Type, Dims...>& dout) {
      auto ret = make_ndarray_ptr<Type, Dims...>();
      for (int i = 0; i < dout.size(); i++)
        ret->linerAt(i) = dout.linerAt(i) > 0 ? 1 : 0;
      return std::move(ret);
    }

    using output = ndarray<Type, Dims...>;

    template <class Func>
    void update(Func optimize) {}
  };

  template <typename Type, int... Dims>
  std::ostream& operator<<(std::ostream& os, const Relu<Type, Dims...>& layer) {
    os << "======== Relu Layer ========" << std::endl;
    os << "Args : " << ndarray<int, sizeof...(Dims)>({Dims...}) << std::endl;
    return os;
  }

  template <typename Type, int N, int K, int... Dims>
  class Affine {
   public:
    struct M {
      enum { value = GetFact<sizeof...(Dims) - 1, Dims...>::value };
    };

    Affine() {
      x = make_ndarray_ptr<Type, N, M::value>();
      w = make_ndarray_ptr<Type, M::value, K>();
      b = make_ndarray_ptr<Type, K>();

      dw = make_ndarray_ptr<Type, M::value, K>();
      db = make_ndarray_ptr<Type, K>();

      w->rand();
      w = *w * (Type)sqrt(2.0 / N);
      b->fill(0);
    }
    ndarrayPtr<Type, N, K> forward(const ndarray<Type, N, Dims...>& input) {
      x = input.template reshape<N, M::value>();
      ndarrayPtr<Type, N, K> ret = dot(*x, *w);
      for (int i = 0; i < N; i++)
        for (int j = 0; j < K; j++) ret->at(i, j) = ret->at(i, j) + b->at(j);
      return std::move(ret);
    }

    ndarrayPtr<Type, N, Dims...> backward(const ndarray<Type, N, K>& dout) {
      ndarrayPtr<Type, N, M::value> ret = dot(dout, *(w->T()));
      dw = dot(*(x->T()), dout);
      db = dout.template sum<0>();
      return std::move(ret->template reshape<N, Dims...>());
    }

    using output = ndarray<Type, N, K>;

    template <class Func>
    void update(Func optimize) {
      optimize(w, dw);
      optimize(b, db);
    }

    ndarrayPtr<Type, N, M::value> x;
    ndarrayPtr<Type, M::value, K> w;
    ndarrayPtr<Type, K> b;

    ndarrayPtr<Type, M::value, K> dw;
    ndarrayPtr<Type, K> db;
  };

  template <typename Type, int N, int K, int... Dims>
  std::ostream& operator<<(std::ostream& os,
                           const Affine<Type, N, K, Dims...>& layer) {
    os << "======== Affine Layer ========" << std::endl;
    os << "Args : " << N << ", " << K << ", "
       << ndarray<int, sizeof...(Dims)>({Dims...}) << std::endl;
    os << "w : " << *(layer.w) << std::endl;
    os << "dw: " << *(layer.dw) << std::endl;
    os << "b : " << *(layer.b) << std::endl;
    os << "db: " << *(layer.db) << std::endl;
    return os;
  }

  template <typename Type, int... Dims>
  class Dropout {
   public:
    Dropout() { mask = make_ndarray_ptr<float, Dims...>(); }

    ndarrayPtr<Type, Dims...> forward(const ndarray<Type, Dims...>& input,
                                      bool train_flag = true) {
      if (!train_flag) return input * (float)(1.0 - dropout_ratio);
      auto rnd = make_ndarray_ptr<Type, Dims...>();
      rnd->rand();
      for (int i = 0; i < rnd->size(); i++) {
        if (rnd->linerAt(i) > dropout_ratio)
          mask->linerAt(i) = 1.0;
        else
          mask->linerAt(i) = 0.0;
      }
      return input * *(mask);
    }
    ndarrayPtr<Type, Dims...> backward(const ndarray<Type, Dims...>& dout) {
      return dout * *mask;
    }

    void set_dropout_ratio(float v) { dropout_ratio = v; }

    using output = ndarray<Type, Dims...>;

    template <class Func>
    void update(Func optimize) {}

    float dropout_ratio;
    ndarrayPtr<float, Dims...> mask;
  };

  template <typename Type, int... Dims>
  std::ostream& operator<<(std::ostream& os,
                           const Dropout<Type, Dims...>& layer) {
    os << "======== Dropout Layer ========" << std::endl;
    os << "Args : " << ndarray<int, sizeof...(Dims)>({Dims...}) << std::endl;
    os << "dropout_ratio : " << layer.dropout_ratio << std::endl;
    os << "mask          : " << *(layer.mask) << std::endl;
    return os;
  }

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
      w = make_ndarray_ptr<Type, FILTER_N, C, FILTER_H, FILTER_W>();
      b = make_ndarray_ptr<Type, FILTER_N>();

      col = make_ndarray_ptr<Type, N * OUT_H::value * OUT_W::value,
                             C * FILTER_H * FILTER_W>();
      col_w = make_ndarray_ptr<Type, C * FILTER_H * FILTER_W, FILTER_N>();

      db = make_ndarray_ptr<Type, FILTER_N>();
      dw = make_ndarray_ptr<Type, FILTER_N, C, FILTER_H, FILTER_W>();

      w->rand();
      w = *w * (Type)sqrt(2.0 / N);
      b->fill(0);
    }

    ndarrayPtr<Type, N, FILTER_N, OUT_H::value, OUT_W::value> forward(
        const ndarray<Type, N, C, H, W>& input) {
      auto col = input.template im2col<FILTER_H, FILTER_W, STRIDE, PAD>();
      col_w = w->template reshape<FILTER_N, C * FILTER_H * FILTER_W>()->T();

      auto out = dot(*col, *col_w);
      for (int i = 0; i < N * OUT_H::value * OUT_W::value; i++)
        for (int l = 0; l < FILTER_N; l++)
          out->at(i, l) = out->at(i, l) + b->at(l);
      ndarrayPtr<Type, N, FILTER_N, OUT_H::value, OUT_W::value> ret =
          out->template reshape<N, OUT_H::value, OUT_W::value, FILTER_N>()
              ->template transpose<0, 3, 1, 2>();
      return std::move(ret);
    }

    ndarrayPtr<Type, N, C, H, W> backward(
        const ndarray<Type, N, FILTER_N, OUT_H::value, OUT_W::value>& dout) {
      auto out =
          dout.template transpose<0, 2, 3, 1>()
              ->template reshape<N * OUT_H::value * OUT_W::value, FILTER_N>();
      db = out->template sum<0>();
      auto tdw = dot(*(col->T()), *out);
      dw = tdw->template transpose<1, 0>()
               ->template reshape<FILTER_N, C, FILTER_H, FILTER_W>();

      auto dcol = dot(*out, *(col_w->T()));
      ndarrayPtr<Type, N, C, OUT_H::value, OUT_W::value> ret =
          dcol->template col2im<N, C, H, W, FILTER_H, FILTER_W, STRIDE, PAD>();
      return std::move(ret);
    };

    using output = ndarray<Type, N, FILTER_N, OUT_H::value, OUT_W::value>;

    template <class Func>
    void update(Func optimize) {
      optimize(w, dw);
      optimize(b, db);
    }
    ndarrayPtr<Type, FILTER_N, C, FILTER_H, FILTER_W> w;
    ndarrayPtr<Type, FILTER_N> b;

    ndarrayPtr<Type, N * OUT_H::value * OUT_W::value, C * FILTER_H * FILTER_W>
        col;
    ndarrayPtr<Type, C * FILTER_H * FILTER_W, FILTER_N> col_w;

    ndarrayPtr<Type, FILTER_N> db;
    ndarrayPtr<Type, FILTER_N, C, FILTER_H, FILTER_W> dw;
  };

  template <typename Type, int N, int C, int H, int W, int FILTER_N,
            int FILTER_H, int FILTER_W, int STRIDE, int PAD>
  std::ostream& operator<<(
      std::ostream& os, const Convolution<Type, N, C, H, W, FILTER_N, FILTER_H,
                                          FILTER_W, STRIDE, PAD>& layer) {
    os << "======== Convolution Layer ========" << std::endl;
    os << "Args : "
       << ndarray<int, 9>(
              {N, C, H, W, FILTER_N, FILTER_H, FILTER_W, STRIDE, PAD})
       << std::endl;
    os << "w : " << *(layer.w) << std::endl;
    os << "dw: " << *(layer.dw) << std::endl;
    os << "b : " << *(layer.b) << std::endl;
    os << "db: " << *(layer.db) << std::endl;
    return os;
  }

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
    Pooling() {
      x = make_ndarray_ptr<Type, N, C, H, W>();
      arg_max =
          make_ndarray_ptr<unsigned, N * OUT_H::value * OUT_W::value * C>();
    }

    ndarrayPtr<Type, N, C, OUT_H::value, OUT_W::value> forward(
        const ndarray<Type, N, C, H, W>& input) {
      *x = input;
      auto col_t = x->template im2col<POOL_H, POOL_W, STRIDE, 0>();
      auto col = col_t->template reshape<N * OUT_H::value * OUT_W::value * C,
                                         POOL_H * POOL_W>();

      arg_max = col->template argmax<1>();
      auto out = col->template max<1>();
      ndarrayPtr<Type, N, C, OUT_H::value, OUT_W::value> ret =
          out->template reshape<N, OUT_H::value, OUT_W::value, C>()
              ->template transpose<0, 3, 1, 2>();
      return std::move(ret);
    }

    ndarrayPtr<Type, N, C, H, W> backward(
        const ndarray<Type, N, C, OUT_H::value, OUT_W::value>& dout) {
      auto out = dout.template transpose<0, 2, 3, 1>();

      auto dmax = make_ndarray_ptr<Type, N * OUT_H::value * OUT_W::value * C,
                                   POOL_H * POOL_W>();
      dmax->fill(0);
      for (int i = 0; i < arg_max->size(); i++) {
        dmax->at(i, arg_max->at(i)) = dout.linerAt(i);
      }
      auto dcol = dmax->template reshape<N * OUT_H::value * OUT_W::value,
                                         C * POOL_H * POOL_W>();
      ndarrayPtr<Type, N, C, H, W> dx =
          dcol->template col2im<N, C, H, W, POOL_H, POOL_W, STRIDE, 0>();
      return std::move(dx);
    };

    using output = ndarray<Type, N, C, OUT_H::value, OUT_W::value>;

    template <class Func>
    void update(Func optimize) {}

   private:
    ndarrayPtr<Type, N, C, H, W> x;
    ndarrayPtr<unsigned, N * OUT_H::value * OUT_W::value * C> arg_max;
  };

  template <typename Type, int N, int C, int H, int W, int POOL_H, int POOL_W,
            int STRIDE>
  std::ostream& operator<<(
      std::ostream& os,
      const Pooling<Type, N, C, H, W, POOL_H, POOL_W, STRIDE>& layer) {
    os << "======== Pooling Layer ========" << std::endl;
    os << "Args : " << ndarray<int, 7>({N, C, H, W, POOL_H, POOL_W, STRIDE})
       << std::endl;
    return os;
  }

  template <typename Type, int N, int M>
  class SoftmaxWithLoss {
   public:
    SoftmaxWithLoss() {
      y = make_ndarray_ptr<Type, N, M>();
      t = make_ndarray_ptr<Type, N, M>();
    }

    Type forward(const ndarray<Type, N, M>& input,
                 const ndarray<Type, N, M>& teacher) {
      y = softmax(input);
      *t = teacher;

      Type loss = cross_entropy_error(*y, teacher);
      return loss;
    };

    ndarrayPtr<Type, N, M> backward(const Type dout = (Type)1) {
      // TODO : now only one-hot-expression
      ndarrayPtr<Type, N, M> dx = *(*y - *t) / (Type)N;
      return std::move(dx);
    };

    using output = Type;

    template <class Func>
    void update(Func optimize) {}

   private:
    ndarrayPtr<Type, N, M> y;
    ndarrayPtr<Type, N, M> t;
  };

  template <typename Type, int N, int M>
  std::ostream& operator<<(std::ostream& os, const SoftmaxWithLoss<Type, N, M>& layer) {
    os << "======== SoftmaxWithLoss Layer ========" << std::endl;
    os << "Args : " << N << ", " << M << std::endl;
    return os;
  }

};  // namespace dpl

#endif  // DEEP_LEARNING_FROM_SCRATCH_LAYER_HPP
