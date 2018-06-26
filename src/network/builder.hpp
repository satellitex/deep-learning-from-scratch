//
// Created by TakumiYamashita on 2018/06/10.
//

#ifndef DEEP_LEARNING_FROM_SCRATCH_BUILDER_HPP
#define DEEP_LEARNING_FROM_SCRATCH_BUILDER_HPP

#include "network.hpp"

namespace dpl {

  // ======= InputLayer はじめにする ========
  template <int... Ints>
  class InputLayer {
   public:
    using output = ndarray<float, Ints...>;
  };

  template <class Last, class... Layers>
  class NetworkBuilder_ {
   public:
    // =========================== Convolution =============================
    template <class OUT, int FILTER_N, int FILTER_H, int FILTER_W, int STRIDE,
              int PAD>
    struct ConvolutionBuild;

    template <int N, int C, int H, int W, int FILTER_N, int FILTER_H,
              int FILTER_W, int STRIDE, int PAD>
    struct ConvolutionBuild<ndarray<float, N, C, H, W>, FILTER_N, FILTER_H,
                            FILTER_W, STRIDE, PAD> {
      using type = Convolution<float, N, C, H, W, FILTER_N, FILTER_H, FILTER_W,
                               STRIDE, PAD>;
    };

    template <int FILTER_N, int FILTER_H, int FILTER_W, int STRIDE, int PAD>
    auto Convolution() {
      NetworkBuilder_<
          typename ConvolutionBuild<typename Last::output, FILTER_N, FILTER_H,
                                    FILTER_W, STRIDE, PAD>::type,
          Layers...>
          builder_;
      return std::move(builder_);
    };
    // ====================================================================
  };

  class NetworkBuilder {
   public:
    template <int... Ints>
    static auto Input() {
      NetworkBuilder_<InputLayer<Ints...>> builder_;
      return std::move(builder_);
    }
  };

}  // namespace dpl

#endif  // DEEP_LEARNING_FROM_SCRATCH_BUILDER_HPP
