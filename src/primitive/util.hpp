//
// Created by TakumiYamashita on 2018/06/10.
//

#ifndef DEEP_LEARNING_FROM_SCRATCH_UTIL_HPP
#define DEEP_LEARNING_FROM_SCRATCH_UTIL_HPP

#include "primitive.hpp"

/*
 * def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """

    Parameters
    ----------
    input_data : (データ数, チャンネル, 高さ, 幅)の4次元配列からなる入力データ
    filter_h : フィルターの高さ
    filter_w : フィルターの幅
    stride : ストライド
    pad : パディング

    Returns
    -------
    col : 2次元配列
    """
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col
*/
namespace dpl {
  /*
   * Parameters
   * ----------
   * input_data : ndarray<データ数, チャンネル, 高さ, 幅>
   * の4次元配列からなる入力データ
   *
   * template<I,J,K,L,FILTER_H,FILTER_W,STRIDE,PAD>
   * N : データ数
   * C : チャンネル数
   * H : 高さ
   * W : 幅
   * FILTER_H : フィルターの高さ
   * FILTER_W : フィルターの幅
   * STRIDE : ストライド
   * PAD : パディング
   *
   * Returns
   * -------
   * col : ndarray<OUT_H, OUT_W> の2次元配列
   *
   * OUT_H = (H + 2*PAD - FILTER_H)/STRIDE + 1
   * OUT_W = (W + 2*PAD - FILTER_W)/STRIDE + 1
   */
  template <typename T, int N, int C, int H, int W, int FILTER_H, int FILTER_W,
            int STRIDE = 1, int PAD = 0>
  auto im2col(const ndarray<T, I, J, K, L>& input_data) {
    constexpr int OUT_H = (H + 2 * PAD - FILTER_H) / STRIDE + 1;
    constexpr int OUT_W = (W + 2 * PAD - FILTER_W) / STRIDE + 1;

    /*
     *
    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col
     */

  };
}  // namespace dpl
/*
def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
  """

  Parameters
  ----------
  col :
  input_shape : 入力データの形状（例：(10, 1, 28, 28)）
  filter_h :
  filter_w
  stride
  pad

  Returns
  -------

  """
  N, C, H, W = input_shape
  out_h = (H + 2*pad - filter_h)//stride + 1
  out_w = (W + 2*pad - filter_w)//stride + 1
  col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4,
5, 1, 2)

  img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
  for y in range(filter_h):
      y_max = y + stride*out_h
      for x in range(filter_w):
          x_max = x + stride*out_w
          img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

  return img[:, :, pad:H + pad, pad:W + pad]
*/

#endif  // DEEP_LEARNING_FROM_SCRATCH_UTIL_HPP
