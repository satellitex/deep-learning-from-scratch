#include <iostream>
#include <array>

template< typename T, int... args>
class myArray;

template< typename T, int F>
class myArray<T,F> : std::array<T,F> {
 public:
  void print() {
    for(int i=0;i<F;i++) {
      if (i) std::cout << ",";
      std::cout << this->at(i);
    }
  }
};

template< typename T, int F, int S, int... args>
class myArray<T,F,S,args...> : std::array<myArray<T,S,args...>,F> {
 public:
  void print() {
    for(int i=0;i<F;i++) {
      if (i) std::cout << ",";
      this->at(i).print();
    }
  }
};


int main() {
  myArray<int,5> array;
  array.print();

  std::cout << std::endl;

  myArray<int,2,2,2> a2;
  a2.print();
}