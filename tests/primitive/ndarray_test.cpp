//
// Created by TakumiYamashita on 2018/06/10.
//

#include <gtest/gtest.h>
#include <iostream>
#include "../src/primitive/primitive.hpp"

using namespace dpl;

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

  ASSERT_TRUE(*xres == expected_res);
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

TEST(ND_ARRAY_TEST, SHAPE_TEST) {
  ndarray<float, 2, 2> z1;
  ASSERT_EQ(std::make_tuple(2, 2), z1.shape());

  ndarray<float, 10, 9, 8, 7, 2> z2;
  ASSERT_EQ(std::make_tuple(10, 9, 8, 7, 2), z2.shape());

  ndarray<float, 3> z3;
  ASSERT_EQ(std::make_tuple(3), z3.shape());
}

TEST(ND_ARRAY_TEST, RESHAPE_TEST_1_TO_3x4) {
  ndarray<float, 12> x;
  x << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;
  auto nx = x.reshape<3, 4>();
  for (int i = 0, cnt = 0; i < 3; i++)
    for (int j = 0; j < 4; j++, cnt++) ASSERT_FLOAT_EQ(x.at(cnt), nx->at(i, j));
}

TEST(ND_ARRAY_TEST, RESHAPE_TEST_3x2x2_TO_12) {
  ndarray<float, 3, 2, 2> x;
  x << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;
  auto nx = x.reshape<12>();
  for (int i = 0, cnt = 0; i < 3; i++)
    for (int j = 0; j < 2; j++)
      for (int k = 0; k < 2; k++, cnt++)
        ASSERT_FLOAT_EQ(x.at(i, j, k), nx->at(cnt));
}

TEST(ND_ARRAY_TEST, RESHAPE_TEST_3x2x2_TO_4x3) {
  ndarray<float, 3, 2, 2> x;
  x << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;
  auto nx = x.reshape<4, 3>();
  for (int i = 0, cnt = 0; i < 3; i++)
    for (int j = 0; j < 2; j++)
      for (int k = 0; k < 2; k++, cnt++)
        ASSERT_FLOAT_EQ(x.at(i, j, k), nx->at(cnt / 3, cnt % 3));
}

TEST(ND_ARRAY_TEST, TRANSPOSE_3x2x2) {
  ndarray<float, 3, 2, 2> x;
  x << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;
  auto nx = x.transpose<2, 1, 0>();
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 2; j++)
      for (int k = 0; k < 2; k++)
        ASSERT_FLOAT_EQ(x.at(i, j, k), nx->at(k, j, i));
}

TEST(ND_ARRAY_TEST, TRANSPOSE_5x4x3x2x3) {
  ndarray<float, 5, 4, 3, 2, 6> x;

  for (int i = 0; i < 5; i++)
    for (int j = 0; j < 4; j++)
      for (int k = 0; k < 3; k++)
        for (int n = 0; n < 2; n++)
          for (int m = 0; m < 6; m++)
            x.at(i, j, k, n, m) = i * 10000 + j * 1000 + k * 100 + n * 10 + m;

  auto tx = x.transpose<4, 3, 2, 1, 0>();
  for (int i = 0; i < 5; i++)
    for (int j = 0; j < 4; j++)
      for (int k = 0; k < 3; k++)
        for (int n = 0; n < 2; n++)
          for (int m = 0; m < 6; m++)
            ASSERT_FLOAT_EQ(x.at(i, j, k, n, m), tx->at(m, n, k, j, i));

  auto tx2 = x.transpose<3, 4, 0, 2, 1>();
  for (int i = 0; i < 5; i++)
    for (int j = 0; j < 4; j++)
      for (int k = 0; k < 3; k++)
        for (int n = 0; n < 2; n++)
          for (int m = 0; m < 6; m++)
            ASSERT_FLOAT_EQ(x.at(i, j, k, n, m), tx2->at(n, m, i, k, j));
}

TEST(ND_ARRAY_TEST, REVERSE_TRANSPOSE_3x2x2) {
  ndarray<float, 3, 2, 2> x;
  x << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;
  auto nx = x.T();
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 2; j++)
      for (int k = 0; k < 2; k++)
        ASSERT_FLOAT_EQ(x.at(i, j, k), nx->at(k, j, i));
}

TEST(ND_ARRAY_TEST, REVERSE_TRANSPOSE_5x4x3x2x3) {
  ndarray<float, 5, 4, 3, 2, 6> x;

  for (int i = 0; i < 5; i++)
    for (int j = 0; j < 4; j++)
      for (int k = 0; k < 3; k++)
        for (int n = 0; n < 2; n++)
          for (int m = 0; m < 6; m++)
            x.at(i, j, k, n, m) = i * 10000 + j * 1000 + k * 100 + n * 10 + m;

  auto tx = x.T();
  for (int i = 0; i < 5; i++)
    for (int j = 0; j < 4; j++)
      for (int k = 0; k < 3; k++)
        for (int n = 0; n < 2; n++)
          for (int m = 0; m < 6; m++)
            ASSERT_FLOAT_EQ(x.at(i, j, k, n, m), tx->at(m, n, k, j, i));
}

TEST(ND_ARRAY_TEST, ARGMAX_3x3x3) {
  ndarray<float, 3, 4, 5> x;
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 4; j++)
      for (int k = 0; k < 5; k++)
        x.at(i, j, k) = (i * 331 + j * 41 + k * 11) % 127;
  ndarray<unsigned, 4, 5> r1;
  r1.fill(0);
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 4; j++)
      for (int k = 0; k < 5; k++)
        if (x.at(i, j, k) > x.at(r1.at(j, k), j, k)) r1.at(j, k) = unsigned(i);

  ASSERT_EQ(r1, *x.argmax<0>());

  ndarray<unsigned, 3, 5> r2;
  r2.fill(0);
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 4; j++)
      for (int k = 0; k < 5; k++)
        if (x.at(i, j, k) > x.at(i, r2.at(i, k), k)) r2.at(i, k) = unsigned(j);

  ASSERT_EQ(r2, *x.argmax<1>());

  ndarray<unsigned, 3, 4> r3;
  r3.fill(0);
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 4; j++)
      for (int k = 0; k < 5; k++)
        if (x.at(i, j, k) > x.at(i, j, r3.at(i, j))) r3.at(i, j) = unsigned(k);

  ASSERT_EQ(r3, *x.argmax<2>());
}

TEST(ND_ARRAY_TEST, MAX_3x4x5x6) {
  ndarray<float, 3, 4, 5, 6> x;
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 4; j++)
      for (int k = 0; k < 5; k++)
        for (int n = 0; n < 6; n++)
          x.at(i, j, k, n) = (i * 331 + j * 41 + k * 11 + n * 7) % 127;

  ndarray<float, 4, 5, 6> r1;
  r1.fill(-1e9);
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 4; j++)
      for (int k = 0; k < 5; k++)
        for (int n = 0; n < 6; n++)
          r1.at(j, k, n) = std::max(r1.at(j, k, n), x.at(i, j, k, n));
  ASSERT_EQ(r1, *x.max<0>());

  ndarray<float, 3, 5, 6> r2;
  r2.fill(-1e9);
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 4; j++)
      for (int k = 0; k < 5; k++)
        for (int n = 0; n < 6; n++)
          r2.at(i, k, n) = std::max(r2.at(i, k, n), x.at(i, j, k, n));
  ASSERT_EQ(r2, *x.max<1>());

  ndarray<float, 3, 4, 6> r3;
  r3.fill(-1e9);
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 4; j++)
      for (int k = 0; k < 5; k++)
        for (int n = 0; n < 6; n++)
          r3.at(i, j, n) = std::max(r3.at(i, j, n), x.at(i, j, k, n));
  ASSERT_EQ(r3, *x.max<2>());

  ndarray<float, 3, 4, 5> r4;
  r4.fill(-1e9);
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 4; j++)
      for (int k = 0; k < 5; k++)
        for (int n = 0; n < 6; n++)
          r4.at(i, j, k) = std::max(r4.at(i, j, k), x.at(i, j, k, n));
  ASSERT_EQ(r4, *x.max<3>());
}

TEST(ND_ARRAY_TEST, SUM_3x4x5x6) {
  ndarray<float, 3, 4, 5, 6> x;
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 4; j++)
      for (int k = 0; k < 5; k++)
        for (int n = 0; n < 6; n++)
          x.at(i, j, k, n) = (i * 331 + j * 41 + k * 11 + n * 7) % 127;

  ndarray<float, 4, 5, 6> r1;
  r1.fill(0);
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 4; j++)
      for (int k = 0; k < 5; k++)
        for (int n = 0; n < 6; n++) r1.at(j, k, n) += x.at(i, j, k, n);
  ASSERT_EQ(r1, *x.sum<0>());

  ndarray<float, 3, 5, 6> r2;
  r2.fill(0);
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 4; j++)
      for (int k = 0; k < 5; k++)
        for (int n = 0; n < 6; n++) r2.at(i, k, n) += x.at(i, j, k, n);
  ASSERT_EQ(r2, *x.sum<1>());

  ndarray<float, 3, 4, 6> r3;
  r3.fill(0);
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 4; j++)
      for (int k = 0; k < 5; k++)
        for (int n = 0; n < 6; n++) r3.at(i, j, n) += x.at(i, j, k, n);
  ASSERT_EQ(r3, *x.sum<2>());

  ndarray<float, 3, 4, 5> r4;
  r4.fill(0);
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 4; j++)
      for (int k = 0; k < 5; k++)
        for (int n = 0; n < 6; n++) r4.at(i, j, k) += x.at(i, j, k, n);
  ASSERT_EQ(r4, *x.sum<3>());
}

TEST(ND_ARRAY_TEST, MAXIMUM_5x4x3) {
  ndarray<float, 5, 4, 3> x1;
  ndarray<float, 5, 4, 3> y1;
  ndarray<float, 5, 4, 3> exp;

  for (int i = 0; i < 5; i++)
    for (int j = 0; j < 4; j++)
      for (int k = 0; k < 3; k++) {
        x1.at(i, j, k) = (i * 7 + j * 117 + k * 41) % 37;
        y1.at(i, j, k) = (i * 11 + j * 191 + k * 71) % 37;
        exp.at(i, j, k) = std::max(x1.at(i, j, k), y1.at(i, j, k));
      }
  ASSERT_EQ(exp, *maximum(x1, y1));
}

TEST(ND_ARRAY_TEST, PLUS_OPERSTOR_3x4x5) {
  ndarray<float, 3, 4, 5> x1;
  ndarray<float, 3, 4, 5> y1;
  ndarray<float, 3, 4, 5> exp;

  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 4; j++)
      for (int k = 0; k < 5; k++) {
        x1.at(i, j, k) = (i * 7 + j * 117 + k * 41) % 37;
        y1.at(i, j, k) = (i * 11 + j * 191 + k * 71) % 37;
        exp.at(i, j, k) = x1.at(i, j, k) + y1.at(i, j, k);
      }
  ASSERT_EQ(exp, *(x1 + y1));
}

TEST(ND_ARRAY_TEST, PLUS_OPERSTOR_3x4x5_V) {
  ndarray<float, 3, 4, 5> x1;
  float v = 100.0;
  ndarray<float, 3, 4, 5> exp;

  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 4; j++)
      for (int k = 0; k < 5; k++) {
        x1.at(i, j, k) = (i * 7 + j * 117 + k * 41) % 37;
        exp.at(i, j, k) = x1.at(i, j, k) + v;
      }
  ASSERT_EQ(exp, *(x1 + v));
}

TEST(ND_ARRAY_TEST, PLUS_OPERSTOR_11) {
  ndarray<float, 11> x1;
  ndarray<float, 11> y1;
  ndarray<float, 11> exp;

  for (int i = 0; i < 11; i++) {
    x1.at(i) = i;
    y1.at(i) = i * 2;
    exp.at(i) = x1.at(i) + y1.at(i);
  }
  ASSERT_EQ(exp, *(x1 + y1));
}

TEST(ND_ARRAY_TEST, PLUS_OPERSTOR_11_V) {
  ndarray<float, 11> x1;
  float v = 100.0;
  ndarray<float, 11> exp;

  for (int i = 0; i < 11; i++) {
    x1.at(i) = i;
    exp.at(i) = x1.at(i) + v;
  }
  ASSERT_EQ(exp, *(x1 + v));
}

TEST(ND_ARRAY_TEST, MULT_OPERSTOR_3x4x5) {
  ndarray<float, 3, 4, 5> x1;
  ndarray<float, 3, 4, 5> y1;
  ndarray<float, 3, 4, 5> exp;

  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 4; j++)
      for (int k = 0; k < 5; k++) {
        x1.at(i, j, k) = (i * 7 + j * 117 + k * 41) % 37;
        y1.at(i, j, k) = (i * 11 + j * 191 + k * 71) % 37;
        exp.at(i, j, k) = x1.at(i, j, k) * y1.at(i, j, k);
      }
  ASSERT_EQ(exp, *(x1 * y1));
}

TEST(ND_ARRAY_TEST, MULT_OPERSTOR_3x4x5_V) {
  ndarray<float, 3, 4, 5> x1;
  float v = 100.0;
  ndarray<float, 3, 4, 5> exp;

  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 4; j++)
      for (int k = 0; k < 5; k++) {
        x1.at(i, j, k) = (i * 7 + j * 117 + k * 41) % 37;
        exp.at(i, j, k) = x1.at(i, j, k) * v;
      }
  ASSERT_EQ(exp, *(x1 * v));
}

TEST(ND_ARRAY_TEST, MULT_OPERSTOR_11) {
  ndarray<float, 11> x1;
  ndarray<float, 11> y1;
  ndarray<float, 11> exp;

  for (int i = 0; i < 11; i++) {
    x1.at(i) = i;
    y1.at(i) = i * 2;
    exp.at(i) = x1.at(i) * y1.at(i);
  }
  ASSERT_EQ(exp, *(x1 * y1));
}

TEST(ND_ARRAY_TEST, MULT_OPERSTOR_11_V) {
  ndarray<float, 11> x1;
  float v = 100.0;
  ndarray<float, 11> exp;

  for (int i = 0; i < 11; i++) {
    x1.at(i) = i;
    exp.at(i) = x1.at(i) * v;
  }
  ASSERT_EQ(exp, *(x1 * v));
}

TEST(ND_ARRAY_TEST, MINUS_OPERSTOR_3x4x5) {
  ndarray<float, 3, 4, 5> x1;
  ndarray<float, 3, 4, 5> y1;
  ndarray<float, 3, 4, 5> exp;

  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 4; j++)
      for (int k = 0; k < 5; k++) {
        x1.at(i, j, k) = (i * 7 + j * 117 + k * 41) % 37;
        y1.at(i, j, k) = (i * 11 + j * 191 + k * 71) % 37;
        exp.at(i, j, k) = x1.at(i, j, k) - y1.at(i, j, k);
      }
  ASSERT_EQ(exp, *(x1 - y1));
}

TEST(ND_ARRAY_TEST, MINUS_OPERSTOR_3x4x5_V) {
  ndarray<float, 3, 4, 5> x1;
  float v = 100.0;
  ndarray<float, 3, 4, 5> exp;

  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 4; j++)
      for (int k = 0; k < 5; k++) {
        x1.at(i, j, k) = (i * 7 + j * 117 + k * 41) % 37;
        exp.at(i, j, k) = x1.at(i, j, k) - v;
      }
  ASSERT_EQ(exp, *(x1 - v));
}

TEST(ND_ARRAY_TEST, MINUS_OPERSTOR_11) {
  ndarray<float, 11> x1;
  ndarray<float, 11> y1;
  ndarray<float, 11> exp;

  for (int i = 0; i < 11; i++) {
    x1.at(i) = i;
    y1.at(i) = i * 2;
    exp.at(i) = x1.at(i) - y1.at(i);
  }
  ASSERT_EQ(exp, *(x1 - y1));
}

TEST(ND_ARRAY_TEST, MINUS_OPERSTOR_11_V) {
  ndarray<float, 11> x1;
  float v = 100.0;
  ndarray<float, 11> exp;

  for (int i = 0; i < 11; i++) {
    x1.at(i) = i;
    exp.at(i) = x1.at(i) - v;
  }
  ASSERT_EQ(exp, *(x1 - v));
}

TEST(ND_ARRAY_TEST, DIV_OPERSTOR_3x4x5) {
  ndarray<float, 3, 4, 5> x1;
  ndarray<float, 3, 4, 5> y1;
  ndarray<float, 3, 4, 5> exp;

  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 4; j++)
      for (int k = 0; k < 5; k++) {
        x1.at(i, j, k) = (i * 7 + j * 117 + k * 41) % 37 + 1;
        y1.at(i, j, k) = (i * 11 + j * 191 + k * 71) % 37 + 1;
        exp.at(i, j, k) = x1.at(i, j, k) / y1.at(i, j, k);
      }
  ASSERT_EQ(exp, *(x1 / y1));
}

TEST(ND_ARRAY_TEST, DIV_OPERSTOR_3x4x5_V) {
  ndarray<float, 3, 4, 5> x1;
  float v = 100.0;
  ndarray<float, 3, 4, 5> exp;

  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 4; j++)
      for (int k = 0; k < 5; k++) {
        x1.at(i, j, k) = (i * 7 + j * 117 + k * 41) % 37 + 1;
        exp.at(i, j, k) = x1.at(i, j, k) / v;
      }
  ASSERT_EQ(exp, *(x1 / v));
}

TEST(ND_ARRAY_TEST, DIV_OPERSTOR_11) {
  ndarray<float, 11> x1;
  ndarray<float, 11> y1;
  ndarray<float, 11> exp;

  for (int i = 0; i < 11; i++) {
    x1.at(i) = i + 1;
    y1.at(i) = (i + 1) * 2;
    exp.at(i) = x1.at(i) / y1.at(i);
  }
  ASSERT_EQ(exp, *(x1 / y1));
}

TEST(ND_ARRAY_TEST, DIV_OPERSTOR_11_V) {
  ndarray<float, 11> x1;
  float v = 100.0;
  ndarray<float, 11> exp;

  for (int i = 0; i < 11; i++) {
    x1.at(i) = (i + 1);
    exp.at(i) = x1.at(i) / v;
  }
  ASSERT_EQ(exp, *(x1 / v));
}

TEST(ND_ARRAY_TEST, SLICE_3x4x5) {
  constexpr int a = 0, b = 2;
  constexpr int c = 1, d = 3;
  constexpr int e = 4, f = 5;

  ndarray<float, 3, 4, 5> x1;
  ndarray<float, b - a, 4, 5> exp1;
  ndarray<float, 3, d - c, 5> exp2;
  ndarray<float, 3, 4, f - e> exp3;
  ndarray<float, 3, 2, 5> exp4;

  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 4; j++)
      for (int k = 0; k < 5; k++) {
        x1.at(i, j, k) = (i * 7 + j * 117 + k * 41) % 37 + 1;
        if (a <= i && i < b) exp1.at(i - a, j, k) = x1.at(i, j, k);
        if (c <= j && j < d) exp2.at(i, j - c, k) = x1.at(i, j, k);
        if (e <= k && k < f) exp3.at(i, j, k - e) = x1.at(i, j, k);
        if (j % 2 == 0) exp4.at(i, j / 2, k) = x1.at(i, j, k);
      }

  auto res1 = x1.slice<0, a, b, 1>();
  ASSERT_EQ(exp1, *res1);

  auto res2 = x1.slice<1, c, d, 1>();
  ASSERT_EQ(exp2, *res2);

  auto res3 = x1.slice<2, e, f, 1>();
  ASSERT_EQ(exp3, *res3);

  auto res4 = x1.slice<1, 0, 4, 2>();
  ASSERT_EQ(exp4, *res4);
}

TEST(ND_ARRAY_TEST, SLICE_3x4x5x6x7) {
  constexpr int a = 1, b = 7, st = 3;

  ndarray<float, 3, 4, 5, 8, 7> x1;
  ndarray<float, 3, 4, 5, (b - a) / st, 7> exp1;

  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 4; j++)
      for (int k = 0; k < 5; k++)
        for (int n = 0; n < 8; n++)
          for (int m = 0; m < 7; m++) {
            x1.at(i, j, k, n, m) =
                (i * 7 + j * 117 + k * 41 + n * 991 + m * 117) % 91 + 1;
            if (a <= n && n < b && (n - a) % st == 0)
              exp1.at(i, j, k, (n - a) / st, m) = x1.at(i, j, k, n, m);
          }

  auto res1 = x1.slice<3, a, b, st>();
  ASSERT_EQ(exp1, *res1);
}

TEST(ND_ARRAY_TEST, PAD_3x4x5) {
  constexpr int a = 2, b = 3;
  ndarray<float, 3, 4, 5> x;
  ndarray<float, 3, 9, 5> exp;
  ndarray<float, 3, 9, 10> exp2;
  exp.fill(0);
  exp2.fill(0);
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 4; j++)
      for (int k = 0; k < 5; k++) {
        x.at(i, j, k) = (i * 7 + j * 117 + k * 41) % 17;
        exp.at(i, j + a, k) = x.at(i, j, k);
        exp2.at(i, j + a, k + a) = x.at(i, j, k);
      }
  auto res = x.pad<1, a, b>();
  ASSERT_EQ(exp, *res);

  auto res2 = res->pad<2, a, b>();
  ASSERT_EQ(exp2, *res2);
}

TEST(ND_ARRAY_TEST, IM2COL_10x8x50x60) {
  ndarray<float, 10, 8, 6, 6> im;
  constexpr int OUT_H = (6 + 2 * 1 - 3) / 1 + 1;
  constexpr int OUT_W = (6 + 2 * 1 - 3) / 1 + 1;
  ndarrayPtr<float, 10 * OUT_H * OUT_W, 8 * 3 * 3> col =
      im.im2col<3, 3, 1, 1>();
  ndarrayPtr<float, 10, 8, 6, 6> r_im = col->col2im<10, 8, 6, 6, 3, 3, 1, 1>();
}

TEST(ND_ARRAY_TEST, ND_RAND) {
  ndarray<float, 10> a;
  a.rand();

  ndarray<float, 10, 10, 10> b;
  b.rand();
}

TEST(ND_ARRAY_TEST, MAX_1DIM) {
  ndarray<float, 100> a;
  a.rand();
  auto b = a.max();
  for (int i = 0; i < 100; i++) ASSERT_TRUE(b >= a.at(i));
}

TEST(ND_ARRAY_TEST, MAX_3DIM) {
  ndarray<float, 100, 10, 5> a;
  a.rand();
  auto b = a.max();
  for (int i = 0; i < 5000; i++) ASSERT_TRUE(b >= a.linerAt(i));
}

TEST(ND_ARRAY_TEST, SOFT_MAX_1DIM) {
  ndarray<float, 100> a;
  a.rand();
  auto b = softmax(a);

  auto maxi = a.max();
  float sumi = 0;
  for (int i = 0; i < 100; i++) {
    sumi += std::exp(a.at(i) - maxi);
  }
  for (int i = 0; i < 100; i++) {
    ASSERT_FLOAT_EQ(std::exp(a.at(i) - maxi) / sumi, b->linerAt(i));
  }
}

TEST(ND_ARRAY_TEST, SOFT_MAX_1DIM_1) {
  ndarray<float, 2> a;
  a << (float)-0.65, (float)0.65;
  auto b = softmax(a);
  ASSERT_FLOAT_EQ((float)0.21416502, b->at(0));
  ASSERT_FLOAT_EQ((float)0.78583498, b->at(1));
}

TEST(ND_ARRAY_TEST, SOFT_MAX_2DIM) {
  ndarray<float, 32, 32> x;
  x.rand();
  auto y = softmax(x);
  auto maxi = x.max();
  for (int i = 0; i < 32 * 32; i++) {
    ASSERT_NE(std::exp(x.linerAt(i) - maxi), y->linerAt(i));
  }
}

TEST(ND_ARRAY_TEST, SOFT_MAX_2DIM_1) {
  ndarray<float, 1, 2> x;
  x << -0.65, 0.65;
  auto y = softmax(x);
  ASSERT_FLOAT_EQ((float)0.21416502, y->at(0, 0));
  ASSERT_FLOAT_EQ((float)0.78583498, y->at(0, 1));
}

TEST(ND_ARRAY_TEST, CROSS_ENTROPY_ERROR) {
  ndarray<float, 5, 100> x;
  ndarray<float, 5, 100> t;
  x.rand();
  t.rand();
  float e = cross_entropy_error(x, t);
  ASSERT_LT((float)0.0, e);
}

TEST(ND_ARRAY_TEST, NDARRAY_PTR) {
  auto ptr = make_ndarray_ptr<float, 100, 100>();
  for (int i = 0; i < 100; i++)
    for (int j = 0; j < 100; j++) ptr->at(i, j) = i * 100 + j;
  for (int i = 0; i < 100; i++)
    for (int j = 0; j < 100; j++) ASSERT_FLOAT_EQ(i * 100 + j, ptr->at(i, j));
  ptr->fill(1);
  for (int i = 0; i < 100; i++)
    for (int j = 0; j < 100; j++) ASSERT_FLOAT_EQ(1, ptr->at(i, j));
}

TEST(ND_ARRAY_TEST, LINRE_AT) {
  auto ptr = make_ndarray_ptr<float, 3, 3, 3>();
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      for (int k = 0; k < 3; k++) {
        ptr->at(i, j, k) = i * 9 + j * 3 + k;
        ASSERT_FLOAT_EQ(ptr->at(i, j, k), ptr->linerAt(i * 9 + j * 3 + k));
      }
}

TEST(ND_ARRAY_TEST, EACH_TEST) {
  auto ptr = make_ndarray_ptr<float, 3, 3, 3>();
  ptr->each([](float& v) { v = 100; });
  for (int i = 0; i < ptr->size(); i++)
    ASSERT_FLOAT_EQ((float)100, ptr->linerAt(i));
}

TEST(ND_ARRAY_TEST, EACH_INDEX) {
  auto ptr = make_ndarray_ptr<float, 3, 3, 3>();
  ptr->each([](float& v, int i) { v = i; });
  for (int i = 0; i < ptr->size(); i++)
    ASSERT_FLOAT_EQ((float)i, ptr->linerAt(i));
}

TEST(ND_ARRAY_TEST, RNDOM_CHOICE) {
  auto ptr = make_ndarray_ptr<int, 10, 2>();
  ptr->each([](int& v, int i) { v = i + 1; });

  auto ch = ptr->random_choice<5>();
  int p = 0;
  for (int i = 0; i < 5; i++) {
    int now = ch->at(i, 0);
    ASSERT_LT(p, now);  // p < now
  }
}

TEST(ND_ARRAY_TEST, MASK_AND_CHOICE) {
  auto mask = make_ndarray_ptr<bool, 10>();
  mask->random_mask<5>();

  auto ptr = make_ndarray_ptr<float, 10, 2>();
  ptr->each([](float& v, int i) { v = i + 1; });

  auto ch = ptr->choice<5>(*mask);
  int p = 0;
  for (int i = 0; i < 5; i++) {
    int now = ch->at(i, 0);
    ASSERT_LT(p, now);  // p < now
  }
}

TEST(ND_ARRAY_TEST, GET_DIM) {
  constexpr int k = ndarray<float, 3, 3, 3>::GetDim<0>::value;
  ASSERT_EQ(3, k);
}