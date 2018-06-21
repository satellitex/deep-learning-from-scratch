//
// Created by TakumiYamashita on 2018/06/10.
//

#ifndef DEEP_LEARNING_FROM_SCRATCH_NDARRAY_HPP
#define DEEP_LEARNING_FROM_SCRATCH_NDARRAY_HPP

#include <algorithm>
#include <array>
#include <bitset>
#include <iostream>
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

  template <typename Type, int... Args>
  class ndarray;

  template <typename Type>
  class ndarray<Type> {};

  template <typename Type, int First>
  class ndarray<Type, First> : public std::array<Type, First> {
   public:
    ndarray() : initialize_ps_(0) {}
    ndarray(const std::array<Type, First>& cp) : std::array<Type, First>(cp) {
      ndarray();
    }

    ndarray<Type, First>& at() { return *this; }
    const ndarray<Type, First>& at() const { return *this; }

    Type& at(int i) { return std::array<Type, First>::at(i); }
    const Type& at(int i) const { return std::array<Type, First>::at(i); }

    Type& linerAt(int index) { return at(index); }
    const Type& linerAt(int index) const { return at(index); }

    constexpr size_t size() const { return First; }
    constexpr auto shape() const { return std::make_tuple(First); }
    template <int NFirst, int... NArgs>
    ndarray<Type, NFirst, NArgs...> reshape() const {
      ndarray<Type, NFirst, NArgs...> ret;
      for (int i = 0; i < size(); i++) ret.linerAt(i) = linerAt(i);
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
    //===============================================================
    // Get<I, Ints...>
    // I : index
    // Ints... : Args...
    // Get<I, Ints...>::value = Args[I]
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

    constexpr size_t size() const { return First * at(0).size(); }
    constexpr auto shape() const {
      return std::make_tuple(First, Second, Args...);
    }
    template <int NFirst, int... NArgs>
    ndarray<Type, NFirst, NArgs...> reshape() const {
      ndarray<Type, NFirst, NArgs...> ret;
      for (int i = 0; i < size(); i++) ret.linerAt(i) = linerAt(i);
      return std::move(ret);
    };

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
    // e.g 1) ndarray<3,4,5>.transpose<2,1,0>() -> ndarray<5,4,3>
    // e.g 2) ndarray<5,6,7>.transpose<1,0,2>() -> ndarray<6,5,7>
    template <int... NArgs>
    auto transpose() const {
      static_assert(sizeof...(NArgs) == sizeof...(Args) + 2,
                    "Transpose don't match number of arguments.");

      typename GetTransposedArray<NArgs...>::type ret;
      make_transpose_<NArgs...>(ret, 0);
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
    auto T() const {
      typename GetReversedTransposedArray<sizeof...(Args) + 2>::type ret;
      make_reverse_transpose_<sizeof...(Args) + 2>(ret, 0);
      return ret;
    }

    // argmax, axis = I
    template <int I>
    auto argmax() const {
      typename GetDecreaseDimArray<unsigned, I, First, Second, Args...>::type
          ret;
      const int jk = size() / GetFact<I, First, Second, Args...>::value;
      const int f = Get<I, First, Second, Args...>::value;
      std::bitset<GetFact<sizeof...(Args) + 1, First, Second, Args...>::value>
          fl = 0;
      for (int i = 0, id = 0; i < ret.size(); i++) {
        ret.linerAt(i) = 0;
        while (fl[id]) id++;
        for (int j = 0, jd = id; j < f; j++, jd += jk) {
          fl[jd] = true;
          if (linerAt(id + ret.linerAt(i) * jk) < linerAt(jd))
            ret.linerAt(i) = j;
        }
      }
      return std::move(ret);
    }

    // max, axis = I
    template <int I>
    auto max() const {
      typename GetDecreaseDimArray<Type, I, First, Second, Args...>::type ret;
      const int jk = size() / GetFact<I, First, Second, Args...>::value;
      const int f = Get<I, First, Second, Args...>::value;
      std::bitset<GetFact<sizeof...(Args) + 1, First, Second, Args...>::value>
          fl = 0;
      for (int i = 0, id = 0; i < ret.size(); i++) {
        while (fl[id]) id++;
        ret.linerAt(i) = linerAt(id);
        for (int j = 0, jd = id; j < f; j++, jd += jk) {
          fl[jd] = true;
          ret.linerAt(i) = std::max(ret.linerAt(i), linerAt(jd));
        }
      }
      return std::move(ret);
    }

    // sum, axis = I
    template <int I>
    auto sum() const {
      typename GetDecreaseDimArray<Type, I, First, Second, Args...>::type ret;
      const int jk = size() / GetFact<I, First, Second, Args...>::value;
      const int f = Get<I, First, Second, Args...>::value;
      std::bitset<GetFact<sizeof...(Args) + 1, First, Second, Args...>::value>
          fl = 0;
      for (int i = 0, id = 0; i < ret.size(); i++) {
        while (fl[id]) id++;
        ret.linerAt(i) = 0;
        for (int j = 0, jd = id; j < f; j++, jd += jk) {
          fl[jd] = true;
          ret.linerAt(i) += linerAt(jd);
        }
      }
      return std::move(ret);
    }

    // sliced i-th[S, E) step is ST
    template <int I, int S, int E, int ST>
    auto slice() const {
      static_assert(ST > 0, "ST must be ST > 0");
      typename GetSlicedArray<I, (E - S) / ST, First, Second, Args...>::type
          ret;
      const int jk = size() / GetFact<I, First, Second, Args...>::value;
      const int f = Get<I, First, Second, Args...>::value;
      for (int i = 0, id = jk * S; i < size() / jk / f; i++, id += f * jk) {
        for (int j = 0; j < (E - S); j += ST) {
          for (int k = 0; k < jk; k++)
            ret.linerAt(i * jk * ((E - S) / ST) + j / ST * jk + k) =
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
     * col : ndarray<Type, OUT_H, OUT_W> の2次元配列
     *
     * OUT_H = (H + 2*PAD - FILTER_H)/STRIDE + 1
     * OUT_W = (W + 2*PAD - FILTER_W)/STRIDE + 1
     */
    template <int FILTER_H, int FILTER_W, int PAD, int STRIDE>
    auto im2col() const;

   public:
    template <int I, int PAD_L, int PAD_R>
    auto pad() const {
      typename GetReshapedByIndexArray<I, PAD_L + PAD_R, First, Second,
                                       Args...>::type ret;
      ret.fill(0);

      const int jk = size() / GetFact<I, First, Second, Args...>::value;
      const int f = Get<I, First, Second, Args...>::value;

      for (int i = 0, id = jk * PAD_L; i < size() / jk / f;
           i++, id += jk * (f + PAD_L + PAD_R))
        for (int j = 0, jd = id; j < jk * f; j++, jd++)
          ret.linerAt(jd) = linerAt(i * jk * f + j);

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

   private:
    size_t initialize_ps_;
  };  // namespace dpl

  template <typename Type, int... Ints>
  ndarray<Type, Ints...> operator+(const ndarray<Type, Ints...>& a,
                                   const ndarray<Type, Ints...>& b) {
    ndarray<Type, Ints...> ret;
    for (int i = 0; i < ret.size(); i++)
      ret.linerAt(i) = a.linerAt(i) + b.linerAt(i);
    return std::move(ret);
  }

  template <typename Type, int... Ints>
  ndarray<Type, Ints...> operator+(const ndarray<Type, Ints...>& a,
                                   const Type& v) {
    ndarray<Type, Ints...> ret;
    for (int i = 0; i < ret.size(); i++) ret.linerAt(i) = a.linerAt(i) + v;
    return std::move(ret);
  }

  template <typename Type, int... Ints>
  ndarray<Type, Ints...> operator*(const ndarray<Type, Ints...>& a,
                                   const ndarray<Type, Ints...>& b) {
    ndarray<Type, Ints...> ret;
    for (int i = 0; i < ret.size(); i++)
      ret.linerAt(i) = a.linerAt(i) * b.linerAt(i);
    return std::move(ret);
  }

  template <typename Type, int... Ints>
  ndarray<Type, Ints...> operator*(const ndarray<Type, Ints...>& a,
                                   const Type& v) {
    ndarray<Type, Ints...> ret;
    for (int i = 0; i < ret.size(); i++) ret.linerAt(i) = a.linerAt(i) * v;
    return std::move(ret);
  }

  template <typename Type, int... Ints>
  ndarray<Type, Ints...> operator-(const ndarray<Type, Ints...>& a,
                                   const ndarray<Type, Ints...>& b) {
    ndarray<Type, Ints...> ret;
    for (int i = 0; i < ret.size(); i++)
      ret.linerAt(i) = a.linerAt(i) - b.linerAt(i);
    return std::move(ret);
  }

  template <typename Type, int... Ints>
  ndarray<Type, Ints...> operator-(const ndarray<Type, Ints...>& a,
                                   const Type& v) {
    ndarray<Type, Ints...> ret;
    for (int i = 0; i < ret.size(); i++) ret.linerAt(i) = a.linerAt(i) - v;
    return std::move(ret);
  }

  template <typename Type, int... Ints>
  ndarray<Type, Ints...> operator/(const ndarray<Type, Ints...>& a,
                                   const ndarray<Type, Ints...>& b) {
    ndarray<Type, Ints...> ret;
    for (int i = 0; i < ret.size(); i++)
      ret.linerAt(i) = a.linerAt(i) / b.linerAt(i);
    return std::move(ret);
  }

  template <typename Type, int... Ints>
  ndarray<Type, Ints...> operator/(const ndarray<Type, Ints...>& a,
                                   const Type& v) {
    ndarray<Type, Ints...> ret;
    for (int i = 0; i < ret.size(); i++) ret.linerAt(i) = a.linerAt(i) / v;
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
  ndarray<Type, First, Third> dot(const ndarray<Type, First, Second>& a,
                                  const ndarray<Type, Second, Third>& b) {
    ndarray<Type, First, Third> ret;
    ret.fill(0);
    for (int i = 0; i < First; i++)
      for (int j = 0; j < Second; j++)
        for (int k = 0; k < Third; k++) ret.at(i, k) += a.at(i, j) * b.at(j, k);
    return std::move(ret);
  }

  template <typename Type, int... Ints>
  ndarray<Type, Ints...> maximum(const ndarray<Type, Ints...>& a,
                                 const ndarray<Type, Ints...>& b) {
    ndarray<Type, Ints...> ret;
    for (int i = 0; i < a.size(); i++)
      ret.linerAt(i) = std::max(a.linerAt(i), b.linerAt(i));
    return std::move(ret);
  }

  template <typename Type, int First, int Second, int... Args>
  template <int FILTER_H, int FILTER_W, int PAD, int STRIDE>
  auto ndarray<Type, First, Second, Args...>::im2col() const {
    static_assert(sizeof...(Args) != 2,
                  "usage : can only used by ndarray<データ数,チャンネル数, "
                  "高さ, 幅> type");
    constexpr int N = First;
    constexpr int C = Second;
    constexpr int H = Get<0, Args...>::value;
    constexpr int W = Get<1, Args...>::value;
    constexpr int OUT_H = (H + 2 * PAD - FILTER_H) / STRIDE + 1;
    constexpr int OUT_W = (W + 2 * PAD - FILTER_W) / STRIDE + 1;

    ndarray<Type, N, C, H + PAD * 2, W + PAD * 2, H + PAD * 2> img;
    for (int n = 0; n < N; n++)
      for (int c = 0; c < C; c++)
        for (int y = 0; y < H; y++)
          for (int x = 0; x < W; x++)
            img.at(n, c, y + PAD, x + PAD) = at(n, c, y, x);

    ndarray<Type, N, C, FILTER_H, FILTER_W, OUT_H, OUT_W> col;
    col.fill(0);
    for (int n = 0; n < N; n++)
      for (int c = 0; c < C; c++)
        for (int y = 0; y < FILTER_H; y++)
          for (int x = 0; x < FILTER_W; x++)
            for (int oy = 0, iy = y; iy < y + STRIDE * OUT_H;
                 oy++, iy += STRIDE)
              for (int ox = 0, ix = x; ix < x + STRIDE * OUT_W;
                   ox++, ix += STRIDE)
                col.at(n, c, y, x, oy, ox) = img.at(n, c, y, x, ix, iy);

//    ndarray<Type, N, OUT_H, OUT_W, C, FILTER_H, FILTER_W> ret =
//        col.transpose<0, 4, 5, 1, 2, 3>();

    ndarray<Type, N * OUT_H * OUT_W, Second * FILTER_H * FILTER_W> ret2 =
        col.reshape<N * OUT_H * OUT_W, Second * FILTER_H * FILTER_W>();

    return std::move(ret);
  }

}  // namespace dpl

#endif  // DEEP_LEARNING_FROM_SCRATCH_NDARRAY_HPP
