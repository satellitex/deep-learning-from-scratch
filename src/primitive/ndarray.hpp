//
// Created by TakumiYamashita on 2018/06/10.
//

#ifndef DEEP_LEARNING_FROM_SCRATCH_NDARRAY_HPP
#define DEEP_LEARNING_FROM_SCRATCH_NDARRAY_HPP

#include <algorithm>
#include <array>
#include <bitset>
#include <complex>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace dpl {

  class initialize_ndarray_error : public std::logic_error {
   public:
    explicit initialize_ndarray_error()
        : std::logic_error("initilaize out of range") {}
  };

  //===============================================================
  // Get<I, Ints...>
  // I : index
  // Ints... : Args...
  // Get<I, Ints...>::valu
  template <int I, int... Ints>
  struct Get;

  template <int I, int F, int... Ints>
  struct Get<I, F, Ints...> {
    enum { value = Get<I - 1, Ints...>::value };
  };

  template <int F, int... Ints>
  struct Get<0, F, Ints...> {
    enum { value = F };
  };
  //================================================================

  //================================================================
  // GetFact
  // I : index
  // Ints... : Args...
  // GetFact<I, Ints...>::value = Ints[0] * Ints[1] * ... Ints[I]
  template <int I, int... Ints>
  struct GetFact;

  template <int I, int F, int... Ints>
  struct GetFact<I, F, Ints...> {
    enum { value = F * GetFact<I - 1, Ints...>::value };
  };

  template <int F, int... Ints>
  struct GetFact<0, F, Ints...> {
    enum { value = F };
  };
  //================================================================

  template <typename Type, int... Args>
  class ndarray;

  template <typename Type, int... Args>
  using ndarrayPtr = std::unique_ptr<ndarray<Type, Args...>>;

  template <class Type, int... Args>
  ndarrayPtr<Type, Args...> make_ndarray_ptr() {
    return std::move(std::make_unique<ndarray<Type, Args...>>());
  };

  template <typename Type>
  class ndarray<Type> {};

  template <typename Type, int First>
  class ndarray<Type, First> : public std::array<Type, First> {
   public:
    ndarray() : initialize_ps_(0) {
      std::random_device rd;
      mt = new std::mt19937(rd());
      score = new std::uniform_real_distribution<float>(0.0, 1.0);
    }
    ndarray(const std::array<Type, First>& cp) : std::array<Type, First>(cp) {
      ndarray();
    }

    ndarray<Type, First>& at() { return *this; }
    const ndarray<Type, First>& at() const { return *this; }

    Type& at(int i) { return std::array<Type, First>::at(i); }
    const Type& at(int i) const { return std::array<Type, First>::at(i); }

    Type& linerAt(int index) { return at(index); }
    const Type& linerAt(int index) const { return at(index); }

    ndarray<Type, First>& rand() {
      for (int i = 0; i < First; i++) at(i) = (*score)(*mt);
      return *this;
    }

    Type max() const {
      Type maxi = std::numeric_limits<Type>::min();
      for (int i = 0; i < size(); i++) {
        maxi = std::max(maxi, at(i));
      }
      return maxi;
    }

    constexpr size_t size() const { return First; }
    constexpr auto shape() const { return std::make_tuple(First); }
    template <int... NArgs>
    ndarrayPtr<Type, NArgs...> reshape() const {
      static_assert(
          GetFact<sizeof...(NArgs) - 1, NArgs...>::value == First,
          "usage : reshape<NArgs...> number of elements of reshaped array "
          "equal to called ndarray.");
      auto ret = make_ndarray_ptr<Type, NArgs...>();
      for (int i = 0; i < size(); i++) ret->linerAt(i) = linerAt(i);
      return std::move(ret);
    }

    ndarray<Type, First>& operator<<(const Type& v) {
      at(0) = v;
      initialize_ps_ = 1;
      return *this;
    }

    ndarray<Type, First>& operator,(const Type& v) {
      if (initialize_ps_ >= size()) throw initialize_ndarray_error();
      at(initialize_ps_) = v;
      initialize_ps_++;
      return *this;
    }

   private:
    size_t initialize_ps_;
    std::mt19937* mt;
    std::uniform_real_distribution<float>* score;
  };

  template <typename Type, int First>
  std::ostream& operator<<(std::ostream& os, const ndarray<Type, First>& a) {
    os << "[ ";
    for (int i = 0; i < First; i++) {
      if (i) os << ", ";
      os << a.at(i);
    }
    os << " ]";
    return os;
  }

  template <typename Type, int First, int Second, int... Args>
  class ndarray<Type, First, Second, Args...>
      : public std::array<ndarray<Type, Second, Args...>, First> {
   private:
    //================================================================
    // DimExpand<D, Dims...>
    // D : sizeof Dimentions
    // Dims... : ndarray<Type, Dims...>
    template <typename Arr, int D>
    struct DimExpand;

    // version ndarray DimExpand<D, Dims...>::value = ndarray<Type, D, Dims...>
    template <typename U, int D, int... Dims>
    struct DimExpand<ndarray<U, Dims...>, D> {
      using type = ndarray<U, D, Dims...>;
    };
    //================================================================

    //================================================================
    // GetTranposedNdArray<int... Ints>
    // Ints : Args[ Ints[i] ]...
    // type = ndarray< T, Args[Ints[i]...] >
    template <int... Ints>
    struct GetTransposedArray;

    template <int F, int S, int... Ints>
    struct GetTransposedArray<F, S, Ints...> {
      using type =
          typename DimExpand<typename GetTransposedArray<S, Ints...>::type,
                             Get<F, First, Second, Args...>::value>::type;
    };
    template <int F>
    struct GetTransposedArray<F> {
      using type = ndarray<Type, Get<F, First, Second, Args...>::value>;
    };
    //================================================================

    //================================================================
    // GetReversedTranposedArray
    // I : Args Size
    // GetReversedTransposeArray<I>::type = ndarray<T,Args[I-1],
    // Args[I-2],...,Args[0]>
    template <int I, typename Dummy = void>
    struct GetReversedTransposedArray {
      using type = typename DimExpand<
          typename GetReversedTransposedArray<I - 1, Dummy>::type,
          Get<I - 1, First, Second, Args...>::value>::type;
    };

    template <typename Dummy>
    struct GetReversedTransposedArray<0, Dummy> {
      using type = ndarray<Type>;
    };
    //================================================================

    //================================================================
    // GetSlicedArray
    // I : i-th dimension is sliced
    // SZ : i-th dimention's number of elements = SZ
    // Ints : dimension's
    // GetSlicedArray<I,SZ,Ints...>::type
    // ndarray<Type,Ints[0],...SZ,...,Ints[N-1]>;
    template <int I, int SZ, int... Ints>
    struct GetSlicedArray;

    template <int I, int SZ, int F, int... Ints>
    struct GetSlicedArray<I, SZ, F, Ints...> {
      using type =
          typename DimExpand<typename GetSlicedArray<I - 1, SZ, Ints...>::type,
                             F>::type;
    };

    template <int SZ, int F, int... Ints>
    struct GetSlicedArray<0, SZ, F, Ints...> {
      using type =
          typename DimExpand<typename GetSlicedArray<-1, SZ, Ints...>::type,
                             SZ>::type;
    };

    template <int I, int SZ>
    struct GetSlicedArray<I, SZ> {
      using type = ndarray<Type>;
    };
    //================================================================

    //================================================================
    // GetDecreaseDimArray<int I, int... Ints>
    // I : delete I-th dimension
    // Ints : Args[ Ints[i] ]...
    // type = ndarray< T, Args[0],...Args[I-1],Args[I+1],...,Args[N-1] >
    template <typename U, int I, int... Ints>
    struct GetDecreaseDimArray;

    template <typename U, int I, int F, int... Ints>
    struct GetDecreaseDimArray<U, I, F, Ints...> {
      using type = typename DimExpand<
          typename GetDecreaseDimArray<U, I - 1, Ints...>::type, F>::type;
    };

    template <typename U, int F, int... Ints>
    struct GetDecreaseDimArray<U, 0, F, Ints...> {
      using type = typename GetDecreaseDimArray<U, -1, Ints...>::type;
    };

    template <typename U, int I>
    struct GetDecreaseDimArray<U, I> {
      using type = ndarray<U>;
    };
    //================================================================

    //================================================================
    // GetReshapedByIndexArray<int I, int P, int... Ints>
    // I : reshaped Index is I-th dimension
    // P : reshaped add number of elements by Index is I-th dimension.
    // Ints : dimensions
    // type = ndarray< T, Ints[0], ..., Ints[I] + P, ..., Ints[N-1] >
    template <int I, int P, int... Ints>
    struct GetReshapedByIndexArray;

    template <int I, int P, int F, int... Ints>
    struct GetReshapedByIndexArray<I, P, F, Ints...> {
      using type = typename DimExpand<
          typename GetReshapedByIndexArray<I - 1, P, Ints...>::type, F>::type;
    };

    template <int P, int F, int... Ints>
    struct GetReshapedByIndexArray<0, P, F, Ints...> {
      using type = typename DimExpand<
          typename GetReshapedByIndexArray<-1, P, Ints...>::type, F + P>::type;
    };

    template <int I, int P>
    struct GetReshapedByIndexArray<I, P> {
      using type = ndarray<Type>;
    };
    //================================================================

   public:
    ndarray() : initialize_ps_(0) {}

    ndarray<Type, First, Second, Args...>& at() { return *this; }
    const ndarray<Type, First, Second, Args...>& at() const { return *this; }

    template <typename... Int>
    auto& at(int i, Int... args) {
      return std::array<ndarray<Type, Second, Args...>, First>::at(i).at(
          args...);
    }
    template <typename... Int>
    const auto& at(int i, Int... args) const {
      return std::array<ndarray<Type, Second, Args...>, First>::at(i).at(
          args...);
    }

    Type& linerAt(int index) {
      return at(index / at(0).size()).linerAt(index % at(0).size());
    }
    const Type& linerAt(int index) const {
      return at(index / at(0).size()).linerAt(index % at(0).size());
    }

    ndarray<Type, First, Second, Args...>& fill(const Type& v) {
      for (int i = 0; i < First; i++) at(i).fill(v);
      return *this;
    }
    ndarray<Type, First, Second, Args...>& rand() {
      for (int i = 0; i < First; i++) at(i).rand();
      return *this;
    };

    constexpr size_t size() const { return First * at(0).size(); }
    constexpr auto shape() const {
      return std::make_tuple(First, Second, Args...);
    }
    template <int... NArgs>
    ndarrayPtr<Type, NArgs...> reshape() const {
      static_assert(
          GetFact<sizeof...(NArgs) - 1, NArgs...>::value ==
              GetFact<sizeof...(Args) + 1, First, Second, Args...>::value,
          "usage : reshape<NArgs...> number of elements of reshaped array "
          "equal to called ndarray.");
      auto ret = make_ndarray_ptr<Type, NArgs...>();
      for (int i = 0; i < size(); i++) ret->linerAt(i) = linerAt(i);
      return std::move(ret);
    }

   private:
    template <int... NArgs>
    void make_transpose_(Type& v, int id) const {
      static_assert(sizeof...(NArgs) == 0);
      v = linerAt(id);
    }

    template <int F, int... NArgs>
    void make_transpose_(
        typename GetTransposedArray<F, NArgs...>::type& transpose_array,
        int id) const {
      for (int i = 0; i < Get<F, First, Second, Args...>::value;
           i++, id += size() / GetFact<F, First, Second, Args...>::value)
        make_transpose_<NArgs...>(transpose_array.at(i), id);
    }

   public:
    // transpose<I1,I2,...,IN>
    // e.g 1) ndarray<3,4,5>.transpose<2,1,0>() -> ndarrayPtr<5,4,3>
    // e.g 2) ndarray<5,6,7>.transpose<1,0,2>() -> ndarrayPtr<6,5,7>
    template <int... NArgs>
    std::unique_ptr<typename GetTransposedArray<NArgs...>::type> transpose()
        const {
      static_assert(sizeof...(NArgs) == sizeof...(Args) + 2,
                    "Transpose don't match number of arguments.");
      auto ret =
          std::make_unique<typename GetTransposedArray<NArgs...>::type>();
      make_transpose_<NArgs...>(*ret, 0);
      return std::move(ret);
    }

   private:
    template <int I>
    void make_reverse_transpose_(Type& v, int id) const {
      v = linerAt(id);
    }

    template <int I>
    void make_reverse_transpose_(
        typename GetReversedTransposedArray<I>::type& transposed_array,
        int id) const {
      for (int i = 0; i < Get<I - 1, First, Second, Args...>::value;
           i++, id += size() / GetFact<I - 1, First, Second, Args...>::value)
        make_reverse_transpose_<I - 1>(transposed_array.at(i), id);
    }

   public:
    // reverse transpose
    std::unique_ptr<
        typename GetReversedTransposedArray<sizeof...(Args) + 2>::type>
    T() const {
      auto ret = std::make_unique<
          typename GetReversedTransposedArray<sizeof...(Args) + 2>::type>();
      make_reverse_transpose_<sizeof...(Args) + 2>(*ret, 0);
      return std::move(ret);
    }

    // argmax, axis = I
    template <int I>
    std::unique_ptr<
        typename GetDecreaseDimArray<unsigned, I, First, Second, Args...>::type>
    argmax() const {
      auto ret = std::make_unique<typename GetDecreaseDimArray<
          unsigned, I, First, Second, Args...>::type>();
      const int jk = size() / GetFact<I, First, Second, Args...>::value;
      const int f = Get<I, First, Second, Args...>::value;
      auto fl = std::make_unique<std::bitset<
          GetFact<sizeof...(Args) + 1, First, Second, Args...>::value>>();
      (*fl) = 0;
      for (int i = 0, id = 0; i < ret->size(); i++) {
        ret->linerAt(i) = 0;
        while ((*fl)[id]) id++;
        for (int j = 0, jd = id; j < f; j++, jd += jk) {
          (*fl)[jd] = true;
          if (linerAt(id + ret->linerAt(i) * jk) < linerAt(jd))
            ret->linerAt(i) = j;
        }
      }
      return std::move(ret);
    }

    // max, axis = I
    template <int I>
    std::unique_ptr<
        typename GetDecreaseDimArray<Type, I, First, Second, Args...>::type>
    max() const {
      auto ret =
          std::make_unique<typename GetDecreaseDimArray<Type, I, First, Second,
                                                        Args...>::type>();
      const int jk = size() / GetFact<I, First, Second, Args...>::value;
      const int f = Get<I, First, Second, Args...>::value;
      auto fl = std::make_unique<std::bitset<
          GetFact<sizeof...(Args) + 1, First, Second, Args...>::value>>();
      (*fl) = 0;
      for (int i = 0, id = 0; i < ret->size(); i++) {
        while ((*fl)[id]) id++;
        ret->linerAt(i) = linerAt(id);
        for (int j = 0, jd = id; j < f; j++, jd += jk) {
          (*fl)[jd] = true;
          ret->linerAt(i) = std::max(ret->linerAt(i), linerAt(jd));
        }
      }
      return std::move(ret);
    }

    Type max() const {
      Type maxi = std::numeric_limits<Type>::min();
      for (int i = 0; i < size(); i++) {
        maxi = std::max(maxi, linerAt(i));
      }
      return maxi;
    }

    // sum, axis = I
    template <int I>
    auto sum() const {
      auto ret =
          std::make_unique<typename GetDecreaseDimArray<Type, I, First, Second,
                                                        Args...>::type>();
      const int jk = size() / GetFact<I, First, Second, Args...>::value;
      const int f = Get<I, First, Second, Args...>::value;
      auto fl = std::make_unique<std::bitset<
          GetFact<sizeof...(Args) + 1, First, Second, Args...>::value>>();
      (*fl) = 0;
      for (int i = 0, id = 0; i < ret->size(); i++) {
        while ((*fl)[id]) id++;
        ret->linerAt(i) = 0;
        for (int j = 0, jd = id; j < f; j++, jd += jk) {
          (*fl)[jd] = true;
          ret->linerAt(i) += linerAt(jd);
        }
      }
      return std::move(ret);
    }

    // sliced i-th[S, E) step is ST
    template <int I, int S, int E, int ST>
    auto slice() const {
      static_assert(ST > 0, "ST must be ST > 0");
      auto ret =
          std::make_unique<typename GetSlicedArray<I, (E - S) / ST, First,
                                                   Second, Args...>::type>();
      const int jk = size() / GetFact<I, First, Second, Args...>::value;
      const int f = Get<I, First, Second, Args...>::value;
      for (int i = 0, id = jk * S; i < size() / jk / f; i++, id += f * jk) {
        for (int j = 0; j < (E - S); j += ST) {
          for (int k = 0; k < jk; k++)
            ret->linerAt(i * jk * ((E - S) / ST) + j / ST * jk + k) =
                linerAt(id + j * jk + k);
        }
      }
      return std::move(ret);
    }

    /*
     * Parameters
     * ----------
     * *this : ndarray<データ数, チャンネル, 高さ, 幅>
     * の4次元配列からなる入力データであるときのみ使用可能
     *
     * template<FILTER_H,FILTER_W,STRIDE,PAD>
     * FILTER_H : フィルターの高さ
     * FILTER_W : フィルターの幅
     * STRIDE : ストライド
     * PAD : パディング
     *
     * Returns
     * -------
     * col : ndarray<Type, N * OUT_H * OUT_W, C * FILTER_H * FILTER_W>
     * の2次元配列
     *
     * OUT_H = (H + 2*PAD - FILTER_H)/STRIDE + 1
     * OUT_W = (W + 2*PAD - FILTER_W)/STRIDE + 1
     */
    template <int FILTER_H, int FILTER_W, int STRIDE, int PAD>
    auto im2col() const {
      static_assert(sizeof...(Args) == 2,
                    "usage : can only used by ndarray<Type, number of data, "
                    "number of cahnel, "
                    "height, weight>.im2col type");
      constexpr int N = First;
      constexpr int C = Second;
      constexpr int H = Get<0, Args...>::value;
      constexpr int W = Get<1, Args...>::value;
      constexpr int OUT_H = (H + 2 * PAD - FILTER_H) / STRIDE + 1;
      constexpr int OUT_W = (W + 2 * PAD - FILTER_W) / STRIDE + 1;

      auto img = make_ndarray_ptr<Type, N, C, H + PAD * 2, W + PAD * 2>();
      for (int n = 0; n < N; n++)
        for (int c = 0; c < C; c++)
          for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
              img->at(n, c, y + PAD, x + PAD) = at(n, c, y, x);

      auto col =
          make_ndarray_ptr<Type, N, OUT_H, OUT_W, C, FILTER_H, FILTER_W>();
      col->fill(0);
      for (int n = 0; n < N; n++)
        for (int c = 0; c < C; c++)
          for (int y = 0; y < FILTER_H; y++)
            for (int x = 0; x < FILTER_W; x++)
              for (int oy = 0, iy = y; iy < y + STRIDE * OUT_H;
                   oy++, iy += STRIDE)
                for (int ox = 0, ix = x; ix < x + STRIDE * OUT_W;
                     ox++, ix += STRIDE)
                  col->at(n, oy, ox, c, y, x) = img->at(n, c, iy, ix);

      auto ret =
          make_ndarray_ptr<Type, N * OUT_H * OUT_W, C * FILTER_H * FILTER_W>();
      for (int i = 0; i < ret->size(); i++) ret->linerAt(i) = col->linerAt(i);
      return std::move(ret);
    }

    template <int N, int C, int H, int W, int FILTER_H, int FILTER_W,
              int STRIDE, int PAD>
    auto col2im() const {
      static_assert(
          sizeof...(Args) == 0,
          "usage : can only used by ndarray<Type, number of data * "
          "out_h * out_w, c * filter_h * filter_w * stride * pad>.col2im ");
      constexpr int OUT_H = (H + 2 * PAD - FILTER_H) / STRIDE + 1;
      constexpr int OUT_W = (W + 2 * PAD - FILTER_W) / STRIDE + 1;

      auto col =
          make_ndarray_ptr<Type, N, OUT_H, OUT_W, C, FILTER_H, FILTER_W>();
      for (int i = 0; i < col->size(); i++) col->linerAt(i) = linerAt(i);

      auto img = make_ndarray_ptr<Type, N, C, H + PAD * 2, W + PAD * 2>();
      img->fill(0);
      for (int n = 0; n < N; n++)
        for (int c = 0; c < C; c++)
          for (int y = 0; y < FILTER_H; y++)
            for (int x = 0; x < FILTER_W; x++)
              for (int oy = 0, iy = y; iy < y + STRIDE * OUT_H;
                   oy++, iy += STRIDE)
                for (int ox = 0, ix = x; ix < x + STRIDE * OUT_W;
                     ox++, ix += STRIDE)
                  img->at(n, c, iy, ix) = col->at(n, oy, ox, c, y, x);
      auto ret = img->template slice<2, PAD, PAD + H, 1>()
                     ->template slice<3, PAD, PAD + H, 1>();
      return std::move(ret);
    };

    template <int I, int PAD_L, int PAD_R>
    auto pad() const {
      auto ret = std::make_unique<typename GetReshapedByIndexArray<
          I, PAD_L + PAD_R, First, Second, Args...>::type>();
      ret->fill(0);

      const int jk = size() / GetFact<I, First, Second, Args...>::value;
      const int f = Get<I, First, Second, Args...>::value;

      for (int i = 0, id = jk * PAD_L; i < size() / jk / f;
           i++, id += jk * (f + PAD_L + PAD_R))
        for (int j = 0, jd = id; j < jk * f; j++, jd++)
          ret->linerAt(jd) = linerAt(i * jk * f + j);

      return std::move(ret);
    }

    ndarray<Type, First, Second, Args...>& operator<<(const Type& v) {
      at(0) << v;
      initialize_ps_ = 1;
      return *this;
    }
    ndarray<Type, First, Second, Args...>& operator,(const Type& v) {
      if (initialize_ps_ >= size()) throw initialize_ndarray_error();
      at(initialize_ps_ / at(0).size()), v;
      initialize_ps_++;
      return *this;
    }

    template <int Index>
    struct GetDim {
      enum { value = Get<Index, First, Second, Args...>::value };
    };

   private:
    size_t initialize_ps_;
  };  // namespace dpl

  template <typename Type, int... Ints>
  ndarrayPtr<Type, Ints...> operator+(const ndarray<Type, Ints...>& a,
                                      const ndarray<Type, Ints...>& b) {
    auto ret = make_ndarray_ptr<Type, Ints...>();
    for (int i = 0; i < ret->size(); i++)
      ret->linerAt(i) = a.linerAt(i) + b.linerAt(i);
    return std::move(ret);
  }

  template <typename Type, int... Ints>
  ndarrayPtr<Type, Ints...> operator+(const ndarray<Type, Ints...>& a,
                                      const Type& v) {
    auto ret = make_ndarray_ptr<Type, Ints...>();
    for (int i = 0; i < ret->size(); i++) ret->linerAt(i) = a.linerAt(i) + v;
    return std::move(ret);
  }

  template <typename Type, int... Ints>
  ndarrayPtr<Type, Ints...> operator*(const ndarray<Type, Ints...>& a,
                                      const ndarray<Type, Ints...>& b) {
    auto ret = make_ndarray_ptr<Type, Ints...>();
    for (int i = 0; i < ret->size(); i++)
      ret->linerAt(i) = a.linerAt(i) * b.linerAt(i);
    return std::move(ret);
  }

  template <typename Type, int... Ints>
  ndarrayPtr<Type, Ints...> operator*(const ndarray<Type, Ints...>& a,
                                      const Type& v) {
    auto ret = make_ndarray_ptr<Type, Ints...>();
    for (int i = 0; i < ret->size(); i++) ret->linerAt(i) = a.linerAt(i) * v;
    return std::move(ret);
  }

  template <typename Type, int... Ints>
  ndarrayPtr<Type, Ints...> operator-(const ndarray<Type, Ints...>& a,
                                      const ndarray<Type, Ints...>& b) {
    auto ret = make_ndarray_ptr<Type, Ints...>();
    for (int i = 0; i < ret->size(); i++)
      ret->linerAt(i) = a.linerAt(i) - b.linerAt(i);
    return std::move(ret);
  }

  template <typename Type, int... Ints>
  ndarrayPtr<Type, Ints...> operator-(const ndarray<Type, Ints...>& a,
                                      const Type& v) {
    auto ret = make_ndarray_ptr<Type, Ints...>();
    for (int i = 0; i < ret->size(); i++) ret->linerAt(i) = a.linerAt(i) - v;
    return std::move(ret);
  }

  template <typename Type, int... Ints>
  ndarrayPtr<Type, Ints...> operator/(const ndarray<Type, Ints...>& a,
                                      const ndarray<Type, Ints...>& b) {
    auto ret = make_ndarray_ptr<Type, Ints...>();
    for (int i = 0; i < ret->size(); i++)
      ret->linerAt(i) = a.linerAt(i) / b.linerAt(i);
    return std::move(ret);
  }

  template <typename Type, int... Ints>
  ndarrayPtr<Type, Ints...> operator/(const ndarray<Type, Ints...>& a,
                                      const Type& v) {
    auto ret = make_ndarray_ptr<Type, Ints...>();
    for (int i = 0; i < ret->size(); i++) ret->linerAt(i) = a.linerAt(i) / v;
    return std::move(ret);
  }

  template <typename Type, int... Ints>
  bool nearly(const ndarray<Type, Ints...>& a, const ndarray<Type, Ints...>& b,
              const Type& eps) {
    for (int i = 0; i < a.size(); i++)
      if (!(a.linerAt(i) - eps < b.linerAt(i) &&
            b.linerAt(i) < a.linerAt(i) + eps))
        return false;
    return true;
  };

  template <typename Type, int First, int Second, int... Args>
  std::ostream& operator<<(std::ostream& os,
                           const ndarray<Type, First, Second, Args...>& a) {
    os << "[ ";
    for (int i = 0; i < First; i++) {
      if (i) os << " ,";
      os << a.at(i);
    }
    os << " ]";
    return os;
  }

  template <typename Type, int First, int Second, int Third>
  ndarrayPtr<Type, First, Third> dot(const ndarray<Type, First, Second>& a,
                                     const ndarray<Type, Second, Third>& b) {
    auto ret = make_ndarray_ptr<Type, First, Third>();
    ret->fill(0);
    for (int i = 0; i < First; i++)
      for (int j = 0; j < Second; j++)
        for (int k = 0; k < Third; k++)
          ret->at(i, k) += a.at(i, j) * b.at(j, k);
    return std::move(ret);
  }

  template <typename Type, int... Ints>
  ndarrayPtr<Type, Ints...> maximum(const ndarray<Type, Ints...>& a,
                                    const ndarray<Type, Ints...>& b) {
    auto ret = make_ndarray_ptr<Type, Ints...>();
    for (int i = 0; i < a.size(); i++)
      ret->linerAt(i) = std::max(a.linerAt(i), b.linerAt(i));
    return std::move(ret);
  }

  template <typename Type, int... Dims>
  ndarrayPtr<Type, Dims...> exp(const ndarray<Type, Dims...>& input) {
    auto ret = make_ndarray_ptr<Type, Dims...>();
    for (int i = 0; i < ret->size(); i++)
      ret->linerAt(i) = std::exp(input.linerAt(i));
    return std::move(ret);
  };

  template <typename Type, int... Ints>
  ndarrayPtr<Type, Ints...> softmax(const ndarray<Type, Ints...>& x) {
    return std::move(exp(*(x - x.max())));
  }

  template <typename Type, int First, int Second>
  ndarrayPtr<Type, First, Second> softmax(
      const ndarray<Type, First, Second>& x) {
    ndarrayPtr<Type, Second, First> xt = x.T();
    ndarrayPtr<Type, Second> xm = x.template max<0>();
    for (int i = 0; i < Second; i++) {
      xt->at(i) = *(xt->at(i) - xm->at(i));
    }
    ndarrayPtr<Type, Second, First> exp_x = exp(*xt);
    ndarrayPtr<Type, First> sum_exp_x = exp_x->template sum<0>();
    auto ret = make_ndarray_ptr<Type, Second, First>();
    for (int i = 0; i < Second; i++) {
      ret->at(i) = *(exp_x->at(i) / *sum_exp_x);
    }
    return std::move(ret->T());
  }

  template <typename Type, int N, int M>
  Type cross_entropy_error(const ndarray<Type, N, M>& input,
                           const ndarray<Type, N, M>& teacher) {
    auto t = teacher.template argmax<1>();
    Type sum = 0.0;
    for (int i = 0; i < N; i++) {
      sum += std::log(input.at(i, t->at(i)) + 1e-7);
    }
    return sum / N;
  };

}  // namespace dpl

#endif  // DEEP_LEARNING_FROM_SCRATCH_NDARRAY_HPP
