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

  template <class First, class... Others>
  class Network<First, Others...> {
    //    virtual void predict(ndarray& input, std::shared_ptr<ndarray> output,
    //    bool train_flag) = 0; virtual void loss(ndarray& input, ndarray&
    //    teacher, std::shared_ptr<ndarray> output) = 0; virtual void
    //    accuracy(ndarray& input, ndarray& teacher, std::shared_ptr<ndarray>
    //    output, int batch_test) = 0; virtual void gradient(ndarray& input,
    //    ndarray& teacher, std::shared_ptr<ndarray> output) = 0;

   public:
    template <class Teacher, int... Dims>
    float predict(const ndarray<float, Dims...>& in, const Teacher& teacher,
                  bool train_flag) {
      auto out = layer.forward(in);
      return network_.predict(out, teacher, train_flag);
    }

    void set_dropout_ratio_(std::vector<float>::iterator now,
                            std::vector<float>::iterator end) {
      network_.set_dropout_ratio_(now, end);
    }

   private:
    First layer;
    Network<Others...> network_;
  };

  template <int... Dims, class... Others>
  class Network<Dropout<float, Dims...>, Others...> {
   public:
    template <class Teacher>
    float predict(const ndarray<float, Dims...>& in, const Teacher& teacher,
                  bool train_flag) {
      auto out = layer.forward(in, train_flag);
      return network_.predict(out, teacher, train_flag);
    }

    void set_dropout_ratio_(std::vector<float>::iterator now,
                            std::vector<float>::iterator end) {
      layer.set_dropout_ratio(*now);
      if (now + 1 == end) return;
      network_.set_dropout_ratio_(now + 1, end);
    }

   private:
    Dropout<float, Dims...> layer;
    Network<Others...> network_;
  };

  template <int N, int M>
  class Network<SoftmaxWithLoss<float, N, M>> {
   public:
    float predict(const ndarray<float, N, M>& in,
                  const ndarray<float, N, M>& teacher, bool train_flag) {
      return layer.forward(in, teacher);
    }
    void set_dropout_ratio_(std::vector<float>::iterator now,
                            std::vector<float>::iterator end) {}

   private:
    SoftmaxWithLoss<float, N, M> layer;
  };
}  // namespace dpl

#endif  // DEEP_LEARNING_FROM_SCRATCH_NETWORK_HPP
