#include <algorithm>
#include <array>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace std;

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

 private:
  size_t initialize_ps_;
};

template <typename T, int First>
ostream& operator<<(ostream& os, const ndarray<T, First>& a) {
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
  };

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

 private:
  size_t initialize_ps_;
};

template <typename T, int First, int Second, int... Args>
ostream& operator<<(ostream& os, const ndarray<T, First, Second, Args...>& a) {
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
};

int main() {
  ndarray<float, 3, 3, 3> array;
  std::cout << array << std::endl;

  ndarray<float, 3> b({1, 2, 3});
  std::cout << b << std::endl;

  ndarray<float, 3, 3> x;
  x << 1, 2, 3, 4, 5, 6, 7, 8, 9;
  std::cout << x << std::endl;

  // 3-dims initialize
  ndarray<float, 2, 3, 3> y;
  y << 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19;
  std::cout << y << std::endl;

  // 4-dims initialize
  ndarray<float, 3, 3, 2, 2> z;
  z << 01, 02, 03, 04, 11, 12, 13, 14, 22, 22, 23, 24, 101, 102, 103, 104, 111,
      112, 113, 114, 121, 122, 123, 124, 201, 202, 203, 204, 211, 212, 213, 214,
      221, 222, 223, 224;
  std::cout << z << std::endl;

  std::cout << "z(0) : " << z.at(0) << std::endl;
  std::cout << "z(0,0) : " << z.at(0, 0) << std::endl;
  std::cout << "z(0,0,0) : " << z.at(0, 0, 0) << std::endl;
  std::cout << "z(0,0,0,0) : " << z.at(0, 0, 0, 0) << std::endl;

  // iniailize out range error
  ndarray<float, 2, 2> ww;
  try {
    ww << 1, 2, 3, 4, 5;  // rotate number
  } catch (std::exception& e) {
    std::cout << e.what() << std::endl;
  }

  // dot
  ndarray<float, 2, 3> x1;
  x1 << 1, 2, 3, 4, 5, 6;
  ndarray<float, 3, 4> x2;
  x2 << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;
  auto xres = dot(x1, x2);
  std::cout << x1 << std::endl;
  std::cout << x2 << std::endl;

  std::cout << xres << std::endl;

  ndarray<float, 2, 2> y1;
  y1 << 1, 2, 3, 4;
  ndarray<float, 2, 2> y2;
  y2 << 5, 6, 7, 8;
  std::swap(y1, y2);
  std::cout << y1 << std::endl;
  std::cout << y2 << std::endl;

  ndarray<float,2,3> z1; z1 << 1,2,3,4,5,6;
  ndarray<float,2,3> z2; z2 << 1,2,3,4,5,6;
  std::cout << (z1 == z2) << std::endl;
  std::cout << (z1 != z2) << std::endl;
  ndarray<float,2,3> z3; z3 << 2,1,1,1,1,1;
  std::cout << (z1 < z3) << std::endl;
  std::cout << (z1 >= z3) << std::endl;
  std::cout << (z1 >= z2) << std::endl;
}