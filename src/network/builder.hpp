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
    // ============================== Relu ===================================
    template <class OUT>
    struct ReluBuild;

    template <int... Dims>
    struct ReluBuild<ndarray<float, Dims...>> {
      using type = Relu<float, Dims...>;
    };

    auto Relu() {
      NetworkBuilder_<typename ReluBuild<typename Last::output>::type, Last,
                      Layers...>
          builder_;
      return std::move(builder_);
    }
    // =======================================================================

    // ============================ Affine ====================================
    template <class OUT, int K>
    struct AffineBuild;

    template <int N, int K, int... Dims>
    struct AffineBuild<ndarray<float, N, Dims...>, K> {
      using type = Affine<float, N, K, Dims...>;
    };

    template <int K>
    auto Affine() {
      NetworkBuilder_<typename AffineBuild<typename Last::output, K>::type,
                      Last, Layers...>
          builder_;
      return std::move(builder_);
    }
    // ========================================================================

    // ============================== Dropout =================================
    template <class OUT>
    struct DropoutBuild;

    template <int... Dims>
    struct DropoutBuild<ndarray<float, Dims...>> {
      using type = Dropout<float, Dims...>;
    };

    auto Dropout() {
      NetworkBuilder_<typename DropoutBuild<typename Last::output>::type, Last,
                      Layers...>
          builder_;
      return std::move(builder_);
    }
    // =====================================================================

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
          Last, Layers...>
          builder_;
      return std::move(builder_);
    };
    // =====================================================================

    // ============================= Pooling ===============================
    template <class OUT, int POOL_H, int POOL_W, int STRIDE>
    struct PoolingBuild;

    template <int N, int C, int H, int W, int POOL_H, int POOL_W, int STRIDE>
    struct PoolingBuild<ndarray<float, N, C, H, W>, POOL_H, POOL_W, STRIDE> {
      using type = Pooling<float, N, C, H, W, POOL_H, POOL_W, STRIDE>;
    };

    template <int POOL_H, int POOL_W, int STRIDE>
    auto Pooling() {
      NetworkBuilder_<typename PoolingBuild<typename Last::output, POOL_H,
                                            POOL_W, STRIDE>::type,
                      Last, Layers...>
          builder_;
      return std::move(builder_);
    };
    // =====================================================================

    // ======================== SoftmaxWithLoss ============================
    template <class OUT>
    struct SoftmaxWithLossBuild;

    template <int N, int M>
    struct SoftmaxWithLossBuild<ndarray<float, N, M>> {
      using type = SoftmaxWithLoss<float, N, M>;
    };

    auto SoftmaxWithLoss() {
      NetworkBuilder_<
          typename SoftmaxWithLossBuild<typename Last::output>::type, Last,
          Layers...>
          builder_;
      return std::move(builder_);
    }
    // =====================================================================

  };  // namespace dpl

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
