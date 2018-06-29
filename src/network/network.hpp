//
// Created by TakumiYamashita on 2018/06/10.
//

#ifndef DEEP_LEARNING_FROM_SCRATCH_NETWORK_HPP
#define DEEP_LEARNING_FROM_SCRATCH_NETWORK_HPP

#include <memory>
#include "../layer/layer.hpp"
#include "../primitive/primitive.hpp"

namespace dpl {

  template <class... Layers>
  class Network;

  template <>
  class Network<> {};

  template <class First, class... Others>
  class Network<First, Others...> {
   public:
    template <int... Dims>
    auto predict(const ndarray<float, Dims...>& in) {
      auto out = layer.forward(in);
      return std::move(network_.predict(*out));
    }

    template <class Teacher, int... Dims>
    float loss(const ndarray<float, Dims...>& in, const Teacher& teacher) {
      auto out = layer.forward(in);
      return network_.loss(*out, teacher);
    }

    template <int BATCH_SIZE, int N, int M, int... Dims>
    float accuracy(const ndarray<float, N, Dims...>& in,
                   const ndarray<float, N, M>& teacher) {
      auto tx = make_ndarray_ptr<float, BATCH_SIZE, Dims...>();

      ndarrayPtr<unsigned, N> t = teacher.template argmax<1>();
      auto tt = make_ndarray_ptr<float, BATCH_SIZE>();

      float acc = 0.0;
      for (int i = 0; i < N / BATCH_SIZE; i++) {
        for (int n = 0; n < BATCH_SIZE; n++) {
          tx->at(n) = in.at(i * BATCH_SIZE + n);
          tt->at(n) = t->at(i * BATCH_SIZE + n);
        }
        ndarrayPtr<float, BATCH_SIZE, M> y = predict(*tx);
        ndarrayPtr<unsigned, BATCH_SIZE> yy = y->template argmax<1>();
        for (int n = 0; n < BATCH_SIZE; n++) {
          if (yy->at(n) == tt->at(n)) acc += 1.0;
        }
      }
      return acc / N;
    };

    auto backward() { return layer.backward(*(network_.backward())); }

    template <int... Dims, int N, int M>
    void gradient(const ndarray<float, N, Dims...>& in,
                  const ndarray<float, N, M>& teacher) {
      loss(in, teacher);
      backward();
    };

    void set_dropout_ratio_(std::vector<float>::iterator now,
                            std::vector<float>::iterator end) {
      network_.set_dropout_ratio_(now, end);
    }

    First& getLayer() { return layer; }
    Network<Others...>& next() { return network_; }

   private:
    First layer;
    Network<Others...> network_;
  };

  template <int... Dims, class... Others>
  class Network<Dropout<float, Dims...>, Others...> {
   public:
    auto predict(const ndarray<float, Dims...>& in) {
      auto out = layer.forward(in, false);
      return network_.predict(*out);
    }

    template <class Teacher>
    float loss(const ndarray<float, Dims...>& in, const Teacher& teacher) {
      auto out = layer.forward(in, true);
      return network_.loss(*out, teacher);
    }

    auto backward() { return layer.backward(*(network_.backward())); }

    void set_dropout_ratio_(std::vector<float>::iterator now,
                            std::vector<float>::iterator end) {
      layer.set_dropout_ratio(*now);
      if (now + 1 == end) return;
      network_.set_dropout_ratio_(now + 1, end);
    }

    Dropout<float, Dims...>& getLayer() { return layer; }
    Network<Others...>& next() { return network_; }

   private:
    Dropout<float, Dims...> layer;
    Network<Others...> network_;
  };

  template <int N, int M>
  class Network<SoftmaxWithLoss<float, N, M>> {
   public:
    ndarrayPtr<float, N, M> predict(const ndarray<float, N, M>& in) {
      auto ret = make_ndarray_ptr<float, N, M>();
      *ret = in;
      return std::move(ret);
    }

    float loss(const ndarray<float, N, M>& in,
               const ndarray<float, N, M>& teacher) {
      return layer.forward(in, teacher);
    }

    ndarrayPtr<float, N, M> backward() { return layer.backward(); };

    void set_dropout_ratio_(std::vector<float>::iterator now,
                            std::vector<float>::iterator end) {}

    SoftmaxWithLoss<float, N, M>& getLayer() { return layer; }
    Network<>& next() { auto ret = Network<>(); return ret; }

   private:
    SoftmaxWithLoss<float, N, M> layer;
  };

};  // namespace dpl

#endif  // DEEP_LEARNING_FROM_SCRATCH_NETWORK_HPP
