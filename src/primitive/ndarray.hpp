//
// Created by TakumiYamashita on 2018/06/10.
//

#ifndef DEEP_LEARNING_FROM_SCRATCH_NDARRAY_HPP
#define DEEP_LEARNING_FROM_SCRATCH_NDARRAY_HPP

#include <algorithm>
#include <array>
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

    T& linerAt(int index) {
      return at(index / at(0).size()).linerAt(index % at(0).size());
    }
    const T& linerAt(int index) const {
      return at(index / at(0).size()).linerAt(index % at(0).size());
    }

   private:
    size_t initialize_ps_;
  };

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

}  // namespace dpl

#endif  // DEEP_LEARNING_FROM_SCRATCH_NDARRAY_HPP
