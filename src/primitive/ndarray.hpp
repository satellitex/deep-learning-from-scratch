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
#include <utility>
#include <vector>

namespace dpl {

  class initialize_ndarray_error : public std::logic_error {
   public:
    explicit initialize_ndarray_error()
        : std::logic_error("initilaize out of range") {}
  };

  template <typename T, int... Args>
  class ndarray;

  template <typename T>
  class ndarray<T> {};

  template <typename T, int First>
  class ndarray<T, First> : public std::array<T, First> {
   public:
    ndarray() : initialize_ps_(0) {}
    ndarray(const std::array<T, First>& cp) : std::array<T, First>(cp) {
      ndarray();
    }

    ndarray<T, First>& at() { return *this; }
    const ndarray<T, First>& at() const { return *this; }

    T& at(int i) { return std::array<T, First>::at(i); }
    const T& at(int i) const { return std::array<T, First>::at(i); }

    ndarray<T, First>& operator<<(const T& v) {
      at(0) = v;
      initialize_ps_ = 1;
      return *this;
    };
    ndarray<T, First>& operator,(const T& v) {
      if (initialize_ps_ >= size()) throw initialize_ndarray_error();
      at(initialize_ps_) = v;
      initialize_ps_++;
      return *this;
    };

    constexpr size_t size() const { return First; }
    constexpr auto shape() const { return std::make_tuple(First); }
    template <int NFirst, int... NArgs>
    ndarray<T, NFirst, NArgs...> reshape() const {
      ndarray<T, NFirst, NArgs...> ret;
      for (int i = 0; i < size(); i++) ret.linerAt(i) = linerAt(i);
      return std::move(ret);
    };

    T& linerAt(int index) { return at(index); }
    const T& linerAt(int index) const { return at(index); }

   private:
    size_t initialize_ps_;
  };

  template <typename T, int First>
  std::ostream& operator<<(std::ostream& os, const ndarray<T, First>& a) {
    os << "[ ";
    for (int i = 0; i < First; i++) {
      if (i) os << ", ";
      os << a.at(i);
    }
    os << " ]";
    return os;
  }

  template <typename T, int First, int Second, int... Args>
  class ndarray<T, First, Second, Args...>
      : public std::array<ndarray<T, Second, Args...>, First> {
   public:
    ndarray() : initialize_ps_(0) {}

    ndarray<T, First, Second, Args...>& at() { return *this; }

    const ndarray<T, First, Second, Args...>& at() const { return *this; }

    template <typename... Int>
    auto& at(int i, Int... args) {
      return std::array<ndarray<T, Second, Args...>, First>::at(i).at(args...);
    }

    template <typename... Int>
    const auto& at(int i, Int... args) const {
      return std::array<ndarray<T, Second, Args...>, First>::at(i).at(args...);
    }

    ndarray<T, First, Second, Args...>& fill(const T& v) {
      for (int i = 0; i < First; i++) at(i).fill(v);
      return *this;
    }

    ndarray<T, First, Second, Args...>& operator<<(const T& v) {
      at(0) << v;
      initialize_ps_ = 1;
      return *this;
    }
    ndarray<T, First, Second, Args...>& operator,(const T& v) {
      if (initialize_ps_ >= size()) throw initialize_ndarray_error();
      at(initialize_ps_ / at(0).size()), v;
      initialize_ps_++;
      return *this;
    }

    constexpr size_t size() const { return First * at(0).size(); }
    constexpr auto shape() const {
      return std::make_tuple(First, Second, Args...);
    }
    template <int NFirst, int... NArgs>
    ndarray<T, NFirst, NArgs...> reshape() const {
      ndarray<T, NFirst, NArgs...> ret;
      for (int i = 0; i < size(); i++) ret.linerAt(i) = linerAt(i);
      return std::move(ret);
    };

    // Get<I, Ints...>
    // I : index
    // Ints... : Args...
    // Get<I, Ints...>::value = Args[I]
    template <int I, int... Ints>
    class Get;

    template <int I, int F, int... Ints>
    class Get<I, F, Ints...> {
     public:
      enum { value = Get<I - 1, Ints...>::value };
    };

    template <int F, int... Ints>
    class Get<0, F, Ints...> {
     public:
      enum { value = F };
    };
    //================================================================

    // GetFact
    // I : index
    // Ints... : Args...
    // GetFact<I, Ints...>::value = Ints[0] * Ints[1] * ... Ints[I]
    template <int I, int... Ints>
    class GetFact;

    template <int I, int F, int... Ints>
    class GetFact<I, F, Ints...> {
     public:
      enum { value = F * GetFact<I - 1, Ints...>::value };
    };

    template <int F, int... Ints>
    class GetFact<0, F, Ints...> {
     public:
      enum { value = F };
    };
    //=============================================================

    // DimExpand<D, Dims...>
    // D : sizeof Dimentions
    // Dims... : ndarray<T, Dims...>
    template <typename Arr, int D>
    class DimExpand;

    // version ndarray DimExpand<D, Dims...>::value = ndarray<T, D, Dims...>
    template <typename U, int D, int... Dims>
    class DimExpand<ndarray<U, Dims...>, D> {
     public:
      using type = ndarray<U, D, Dims...>;
    };

    //==============================================================

    // GetTranposedNdArray<int... Ints>
    // Ints : Args[ Ints[i] ]...
    // type = ndarray< T, Args[Ints[i]...] >
    template <int... Ints>
    class GetTransposedArray;

    template <int F, int S, int... Ints>
    class GetTransposedArray<F, S, Ints...> {
     public:
      using type =
          typename DimExpand<typename GetTransposedArray<S, Ints...>::type,
                             Get<F, First, Second, Args...>::value>::type;
    };
    template <int F>
    class GetTransposedArray<F> {
     public:
      using type = ndarray<T, Get<F, First, Second, Args...>::value>;
    };
    //=============================================================

    template <int... NArgs>
    void make_transpose_(T& v, int id) const {
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

    template <int... NArgs>
    auto transpose() const {
      static_assert(sizeof...(NArgs) == sizeof...(Args) + 2,
                    "Transpose don't match number of arguments.");

      typename GetTransposedArray<NArgs...>::type ret;
      make_transpose_<NArgs...>(ret, 0);
      return std::move(ret);
    }

    // GetDecreaseDimArray<int I, int... Ints>
    // I : delete I-th dimension
    // Ints : Args[ Ints[i] ]...
    // type = ndarray< T, Args[0],...Args[I-1],Args[I+1],...,Args[N-1] >
    template <typename U, int I, int... Ints>
    class GetDecreaseDimArray;

    template <typename U, int I, int F, int... Ints>
    class GetDecreaseDimArray<U, I, F, Ints...> {
     public:
      using type = typename DimExpand<
          typename GetDecreaseDimArray<U, I - 1, Ints...>::type, F>::type;
    };

    template <typename U, int F, int... Ints>
    class GetDecreaseDimArray<U, 0, F, Ints...> {
     public:
      using type = typename GetDecreaseDimArray<U, -1, Ints...>::type;
    };

    template <typename U, int I>
    class GetDecreaseDimArray<U, I> {
     public:
      using type = ndarray<U>;
    };
    //=============================================================

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

    template <int I>
    auto max() const {
      typename GetDecreaseDimArray<T, I, First, Second, Args...>::type ret;
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

    template <int I>
    auto sum() const {
      typename GetDecreaseDimArray<T, I, First, Second, Args...>::type ret;
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

    T& linerAt(int index) {
      return at(index / at(0).size()).linerAt(index % at(0).size());
    }
    const T& linerAt(int index) const {
      return at(index / at(0).size()).linerAt(index % at(0).size());
    }

   private:
    size_t initialize_ps_;
  };  // namespace dpl

  template <typename T, int... Ints>
  ndarray<T,Ints...> operator+(const ndarray<T, Ints...>& a, const ndarray<T, Ints...>& b) {
    ndarray<T, Ints...> ret;
    for (int i = 0; i < ret.size(); i++)
      ret.linerAt(i) = a.linerAt(i) + b.linerAt(i);
    return std::move(ret);
  }

  template <typename T, int... Ints>
  ndarray<T,Ints...> operator+(const ndarray<T, Ints...>& a, const T& v) {
    ndarray<T, Ints...> ret;
    for (int i = 0; i < ret.size(); i++) ret.linerAt(i) = a.linerAt(i) + v;
    return std::move(ret);
  }

  template <typename T, int First, int Second, int... Args>
  std::ostream& operator<<(std::ostream& os,
                           const ndarray<T, First, Second, Args...>& a) {
    os << "[ ";
    for (int i = 0; i < First; i++) {
      if (i) os << " ,";
      os << a.at(i);
    }
    os << " ]";
    return os;
  }

  template <typename T, int First, int Second, int Third>
  ndarray<T, First, Third> dot(const ndarray<T, First, Second>& a,
                               const ndarray<T, Second, Third>& b) {
    ndarray<T, First, Third> ret;
    ret.fill(0);
    for (int i = 0; i < First; i++)
      for (int j = 0; j < Second; j++)
        for (int k = 0; k < Third; k++) ret.at(i, k) += a.at(i, j) * b.at(j, k);
    return std::move(ret);
  }

  template <typename T, int... Ints>
  ndarray<T, Ints...> maximum(const ndarray<T, Ints...>& a,
                              const ndarray<T, Ints...>& b) {
    ndarray<T, Ints...> ret;
    for (int i = 0; i < a.size(); i++)
      ret.linerAt(i) = std::max(a.linerAt(i), b.linerAt(i));
    return std::move(ret);
  }

}  // namespace dpl

#endif  // DEEP_LEARNING_FROM_SCRATCH_NDARRAY_HPP
