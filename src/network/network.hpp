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
    template <int... Dims>
    float predict(const ndarray<float, Dims...> &in, bool train_flag) {
      auto out = layer.forward(in);
      return network_.predict(out, train_flag);
    }
    //    def predict(self, x, train_flg=False):
    //    for layer in self.layers:
    //    if isinstance(layer, Dropout):
    //    x = layer.forward(x, train_flg)
    //    else:
    //    x = layer.forward(x)
    //    return x

   private:
    First layer;
    Network<Others...> network_;
  };

  template <int... Dims, class... Others>
  class Network<Dropout<float, Dims...>, Others...> {
   public:
    float predict(const ndarray<float, Dims...> &in, bool train_flag) {
      auto out = layer.forward(in, train_flag);
      return network_.predict(out, train_flag);
    }

   private:
    Dropout<float, Dims...> layer;
    Network<Others...> network_;
  };

  template <>
  class Network<> {
    float predict(float out, bool train_flag) { return out; }
  };
}  // namespace dpl

#endif  // DEEP_LEARNING_FROM_SCRATCH_NETWORK_HPP
