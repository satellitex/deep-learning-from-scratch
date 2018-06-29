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

    /**
     * Relu Layer
     *
     * @return NetworkBuilder adding Relu Layer.
     */
    auto Relu() {
      NetworkBuilder_<typename ReluBuild<typename Last::output>::type, Last,
                      Layers...>
          builder_;
      builder_.set_dropout_ratio_list_(dropout_ratio_list);
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

    /**
     * Affine Layer
     *
     * @tparam K Affine Layer output to [N][K]. So, K is output dimension.
     * @return NetworkBuilder adding Affine Layer.
     */
    template <int K>
    auto Affine() {
      NetworkBuilder_<typename AffineBuild<typename Last::output, K>::type,
                      Last, Layers...>
          builder_;
      builder_.set_dropout_ratio_list_(dropout_ratio_list);
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

    /**
     * Dropout Layer
     *
     * @param dropout_ratio drop ratio of this Dropout layer.
     * @return NetworkBuilder adding Dropout Layer
     */
    auto Dropout(float dropout_ratio) {
      NetworkBuilder_<typename DropoutBuild<typename Last::output>::type, Last,
                      Layers...>
          builder_;
      builder_.set_dropout_ratio_list_(dropout_ratio_list);
      builder_.set_dropout_ratio_(dropout_ratio);
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

    /**
     * Convolution Layer
     *
     * @tparam FILTER_N Number of Filter.
     * @tparam FILTER_H Height of Filter.
     * @tparam FILTER_W Width of Filter.
     * @tparam STRIDE Stride of Filter.
     * @tparam PAD Pad of Filter.
     * @return NetwrokBuilder adding Convolution Layer
     */
    template <int FILTER_N, int FILTER_H, int FILTER_W, int STRIDE, int PAD>
    auto Convolution() {
      NetworkBuilder_<
          typename ConvolutionBuild<typename Last::output, FILTER_N, FILTER_H,
                                    FILTER_W, STRIDE, PAD>::type,
          Last, Layers...>
          builder_;
      builder_.set_dropout_ratio_list_(dropout_ratio_list);
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

    /**
     * Pooling Layer
     *
     * @tparam POOL_H Height of Pool.
     * @tparam POOL_W Width of Pool.
     * @tparam STRIDE Stride of Pool.
     * @return NetwrokBuilder adding Pooling Layer.
     */
    template <int POOL_H, int POOL_W, int STRIDE>
    auto Pooling() {
      NetworkBuilder_<typename PoolingBuild<typename Last::output, POOL_H,
                                            POOL_W, STRIDE>::type,
                      Last, Layers...>
          builder_;
      builder_.set_dropout_ratio_list_(dropout_ratio_list);
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

    /**
     * SoftmaxWithLoss Layer
     *
     * @return NetworkBuillder adding SoftMaxWithLoss Layer.
     */
    auto SoftmaxWithLoss() {
      NetworkBuilder_<
          typename SoftmaxWithLossBuild<typename Last::output>::type, Last,
          Layers...>
          builder_;
      builder_.set_dropout_ratio_list_(dropout_ratio_list);
      return std::move(builder_);
    }
    // =====================================================================

    template <class Net, class L>
    struct NetExpand;

    template <class L, class... Ls>
    struct NetExpand<Network<Ls...>, L> {
      using type = Network<Ls..., L>;
    };

    template <class... Ls>
    struct NetworkBuild;

    template <class F, class... Ls>
    struct NetworkBuild<F, Ls...> {
      using type =
          typename NetExpand<typename NetworkBuild<Ls...>::type, F>::type;
    };

    template <int... Ints>
    struct NetworkBuild<InputLayer<Ints...>> {
      using type = Network<>;
    };

    /**
     * Build Network.
     *
     * @return Network including Layers.
     */
    auto build() {
      typename NetworkBuild<Last, Layers...>::type network;
      if (!dropout_ratio_list.empty()) {
        reverse(dropout_ratio_list.begin(), dropout_ratio_list.end());
        network.set_dropout_ratio_(dropout_ratio_list.begin(),
                                   dropout_ratio_list.end());
      }
      return std::move(network);
    }

    // ========================= dropout ratio ===========================
    void set_dropout_ratio_(float v) { dropout_ratio_list.emplace_back(v); }
    void set_dropout_ratio_list_(std::vector<float> dropout_ratio_list) {
      this->dropout_ratio_list = dropout_ratio_list;
    }

   private:
    std::vector<float> dropout_ratio_list;

  };  // namespace dpl

  /**
   * NetworkBuilder
   *
   * @tparam N : batch size (if you don't use batch, N = 1)
   */
  template <int N>
  class NetworkBuilder {
   public:
    /**
     * Setting Network Input Paramater.
     *
     * @tparam Ints : Input dimention size...
     * @return NetworkBuilder_ including InputLayer.
     */
    template <int... Ints>
    static auto Input() {
      NetworkBuilder_<InputLayer<N, Ints...>> builder_;
      return std::move(builder_);
    }
  };

}  // namespace dpl

#endif  // DEEP_LEARNING_FROM_SCRATCH_BUILDER_HPP
