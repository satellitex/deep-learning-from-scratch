//
// Created by TakumiYamashita on 2018/06/10.
//

#include <gtest/gtest.h>
#include <iostream>
#include "../src/primitive/primitive.hpp"

using namespace dpl;

TEST(NDARRAY_TEST, NO_ERROR) { std::cout << "passed" << std::endl; }

TEST(ND_ARRAY_TEST, INITTIALIZE_ALL_ZERO_3x3x3) {
  ndarray<float, 3, 3, 3> array;
  array.fill(0);
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      for (int k = 0; k < 3; k++) ASSERT_FLOAT_EQ(array.at(i, j, k), 0);
}

TEST(ND_ARRAY_TEST, INITIALIZE_VECTOR) {
  ndarray<float, 3> b({1, 2, 3});
  ASSERT_FLOAT_EQ(b.at(0), 1);
  ASSERT_FLOAT_EQ(b.at(1), 2);
  ASSERT_FLOAT_EQ(b.at(2), 3);
}

TEST(ND_ARRAY_TEST, STATIC_INITIALIZE_TEST_2DIM) {
  ndarray<float, 3, 3> x;
  x << 1, 2, 3, 4, 5, 6, 7, 8, 9;
  int cnt = 1;
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++, cnt++) ASSERT_FLOAT_EQ(x.at(i, j), cnt);
}

TEST(ND_ARRAY_TEST, STATIC_INITIALIZE_TEST_3DIM) {
  // 3-dims initialize
  ndarray<float, 2, 3, 3> x;
  x << 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19;
  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 3; j++)
      for (int k = 0; k < 3; k++)
        ASSERT_FLOAT_EQ(x.at(i, j, k), i * 10 + j * 3 + k + 1);
}

TEST(ND_ARRAY_TEST, STATIC_INITIALIZE_TEST_4DIM) {
  // 4-dims initialize
  ndarray<float, 3, 3, 2, 2> x;
  x << 01, 02, 03, 04, 11, 12, 13, 14, 21, 22, 23, 24, 101, 102, 103, 104, 111,
      112, 113, 114, 121, 122, 123, 124, 201, 202, 203, 204, 211, 212, 213, 214,
      221, 222, 223, 224;
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      for (int k = 0; k < 2; k++)
        for (int l = 0; l < 2; l++)
          ASSERT_FLOAT_EQ(x.at(i, j, k, l), i * 100 + j * 10 + k * 2 + l + 1);
}

TEST(ND_ARRAY_TEST, ARRAY_ACCESS_AT) {
  ndarray<float, 3, 3, 2, 2> z;
  z << 01, 02, 03, 04, 11, 12, 13, 14, 21, 22, 23, 24, 101, 102, 103, 104, 111,
      112, 113, 114, 121, 122, 123, 124, 201, 202, 203, 204, 211, 212, 213, 214,
      221, 222, 223, 224;

  // get 3-dim
  auto z1 = z.at(0);
  for (int j = 0; j < 3; j++)
    for (int k = 0; k < 2; k++)
      for (int l = 0; l < 2; l++)
        ASSERT_FLOAT_EQ(z1.at(j, k, l), j * 10 + k * 2 + l + 1);

  // get 2-dim
  auto z2 = z.at(0, 0);
  for (int k = 0; k < 2; k++)
    for (int l = 0; l < 2; l++) ASSERT_FLOAT_EQ(z2.at(k, l), k * 2 + l + 1);

  // get 1-dim
  auto z3 = z.at(0, 0, 0);
  for (int l = 0; l < 2; l++) ASSERT_FLOAT_EQ(z3.at(l), l + 1);

  // get element
  ASSERT_FLOAT_EQ(z3.at(0), 1);
}

TEST(ND_ARRAY_TEST, STATIC_INITIALIZE_RANGE_ERROR) {
  ndarray<float, 2, 2> ww;
  bool catch_flag = false;
  try {
    ww << 1, 2, 3, 4, 5;  // rotate number
  } catch (std::exception& e) {
    catch_flag = true;
  }
  ASSERT_TRUE(catch_flag);
}

TEST(ND_ARRAY_TEST, DOT_CALC) {
  ndarray<float, 2, 3> x1;
  x1 << 1, 2, 3, 4, 5, 6;
  ndarray<float, 3, 4> x2;
  x2 << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;
  auto xres = dot(x1, x2);

  ndarray<float, 2, 4> expected_res;
  expected_res << 38, 44, 50, 56, 83, 98, 113, 128;

  ASSERT_TRUE(xres == expected_res);
}

TEST(ND_ARRAY_TEST, SWAP_AND_COPY) {
  ndarray<float, 2, 2> y1, y3;
  y1 << 1, 2, 3, 4;
  y3 = y1;
  ndarray<float, 2, 2> y2, y4;
  y2 << 5, 6, 7, 8;
  y4 = y2;
  std::swap(y1, y2);

  ASSERT_TRUE(y1 == y4);
  ASSERT_TRUE(y2 == y3);

  y3.at(0, 0) = 100;

  ASSERT_FALSE(y2 == y3);
}

TEST(ND_ARRAY_TEST, EQ_TEST) {
  ndarray<float, 2, 3> z1;
  z1 << 1, 2, 3, 4, 5, 6;
  ndarray<float, 2, 3> z2;
  z2 << 1, 2, 3, 4, 5, 6;
  ASSERT_TRUE(z1 == z2);
  ASSERT_FALSE(z1 != z2);

  ndarray<float, 2, 3> z3;
  z3 << 2, 1, 1, 1, 1, 1;
  ASSERT_TRUE(z1 < z3);

  ASSERT_FALSE(z1 >= z3);
  ASSERT_TRUE(z1 >= z2);
}