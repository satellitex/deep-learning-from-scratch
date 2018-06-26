//
// Created by TakumiYamashita on 2018/06/10.
//

#ifndef DEEP_LEARNING_FROM_SCRATCH_BUILDER_HPP
#define DEEP_LEARNING_FROM_SCRATCH_BUILDER_HPP

#include "network.hpp"

namespace dpl {



  template<class... Layers>
  class NetworkBuilder {

    template<template<typename Type, int... ints>, int... params>
        struct BuilderExpand;

    template<int... Dims>
    struct BuilderExpand<Relu<float,Dims...>> {
      using type = NetworkBuilder<Relu<float, Dims...>, Layers...>;
    };

    template<int N, int...Dims>
    struct BuilderExpand<Affine, N, Dims...> {
      using type = NetworkBuilder<Affine<float, N, K, Dims...>, Layers...>;
    };

    template<int FILTER_N, int FILTER_H, int FILTER_W, int STRIDE, int PAD>
    auto Convolution();

    auto Relu {

    };
  };


  template<int... Dims, class... Layers>
  class NetworkBuilder< Relu<float, Dims...>, Layers...> {

  };

  struct Builder {
    using type = NetworkBuilder<N,C,H,W>::
        Convolution<FILTER_N, FILTER_H,FILTER_W, STRIDE, PAD>::type::Relu<H,W>
  };

}  // namespace dpl

#endif  // DEEP_LEARNING_FROM_SCRATCH_BUILDER_HPP
