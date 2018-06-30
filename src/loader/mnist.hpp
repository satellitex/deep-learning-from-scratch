//
// Created by TakumiYamashita on 2018/06/10.
//

#ifndef DEEP_LEARNING_FROM_SCRATCH_MNIST_HPP
#define DEEP_LEARNING_FROM_SCRATCH_MNIST_HPP

#include <sys/stat.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include "../config.hpp"
#include "../primitive/ndarray.hpp"
#include "curl/curl.h"

namespace dpl {
  class Loader {};

  class MNISTLoader {
   private:
    static constexpr int TRAIN_NUM = 60000;
    static constexpr int TEST_NUM = 10000;
    static constexpr int IMAGE_C = 1;
    static constexpr int IMAGE_H = 28;
    static constexpr int IMAGE_W = 28;
    static constexpr int IMAGE_SIZE = 784;

    void download_(std::string file) {
      struct stat st;
      if (!stat(file.c_str(), &st)) {
        std::cout << "already exist " << file << std::endl;
        return;
      }

      file += ".gz";

      CURL *curl;
      CURLcode res;
      curl = curl_easy_init();

      std::cout << "download_ " << file << ": " << curl << std::endl;

      if (curl) {
        FILE *fp;
        fp = fopen(file.c_str(), "wb");
        curl_easy_setopt(curl, CURLOPT_URL, (url_base + file).c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, NULL);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);
        res = curl_easy_perform(curl);
        curl_easy_cleanup(curl);
        fclose(fp);
      }

      std::string command = "gzip -d " + file;
      std::cout << "execute : " << command << std::endl;
      if (system(command.c_str()) == -1)
        std::cout << "Failed Command : " << command << std::endl;
      else
        std::cout << "Success Ungzip" << std::endl;
    }

    template <int SIZE>
    ndarrayPtr<float, SIZE> load_label_(std::string file) {
      std::ifstream fin(file, std::ios::in | std::ios::binary);
      if (!fin) {
        std::cout << "can not open file : " << file << std::endl;
        return nullptr;
      }

      auto array = make_ndarray_ptr<float, SIZE>();

      fin.seekg(8, std::ios_base::beg);  // offset = 8
      for (int i = 0; i < SIZE; i++) {   // TODO 高速読み取り
        uint8_t c;
        fin.read((char *)&c, sizeof(c));
        array->at(i) = (float)c;
      }
      fin.close();
      return std::move(array);
    }

    template <int N, int C, int H, int W>
    ndarrayPtr<float, N, C, H, W> load_image_(std::string file) {
      FILE *fp;
      fp = fopen(file.c_str(), "rb");
      if (!fp) {
        std::cout << "can not open file : " << file << std::endl;
        return nullptr;
      }

      std::cout << "start load" << std::endl;
      uint8_t *array = new uint8_t[N * C * H * W];
      fseek(fp, 16, SEEK_SET);  // offset = 16
      fread((void *)array, sizeof(uint8_t), N * C * H * W, fp);
      fclose(fp);

      std::cout << "load!" << std::endl;
      auto ret = make_ndarray_ptr<float, N, C, H, W>();
      auto &ret_ref = *ret;
      for (int i = 0; i < N * C * H * W; i++)
        ret_ref.linerAt(i) = (float)array[i];
      delete[] array;
      std::cout << "ok convert" << std::endl;
      return std::move(ret);
    };

    template <int N>
    ndarrayPtr<float, N, 10> one_hot_label_(const ndarrayPtr<float, N> &array) {
      auto ret = make_ndarray_ptr<float, N, 10>();
      ret->fill(0);
      for (int i = 0; i < N; i++) ret->at(i, array->at(i)) = (float)1;
      return std::move(ret);
    };

   public:
    MNISTLoader() {
      url_base = MNIST_CONFIG_URL_BASE;
      key_files = {MNIST_CONFIG_TRAIN_IMAGES, MNIST_CONFIG_TRAIN_LABELS,
                   MNIST_CONFIG_TEST_IMAGES, MNIST_CONFIG_TEST_LABELS};
    }

    void download() {
      for (std::string file : key_files) {
        download_(file);
      }
    }

    void load() {
      std::cout << "::download mnist data::" << std::endl;
      download();

      std::cout << "::load label::" << std::endl;
      auto train_label = load_label_<TRAIN_NUM>(key_files[1]);
      auto test_label = load_label_<TEST_NUM>(key_files[3]);

      std::cout << "::load image::" << std::endl;
      train_img =
          load_image_<TRAIN_NUM, IMAGE_C, IMAGE_H, IMAGE_W>(key_files[0]);
      test_img = load_image_<TEST_NUM, IMAGE_C, IMAGE_H, IMAGE_W>(key_files[2]);

      {  // normalize
        std::cout << "::normalize::" << std::endl;
        train_img->each([](float &v) { v /= 255.0; });
        test_img->each([](float &v) { v /= 255.0; });
      }

      {  // one-hot-label
        std::cout << "::convert one-hot-label::" << std::endl;
        train_one_hot_label = one_hot_label_(train_label);
        test_one_hot_label = one_hot_label_(test_label);
      }
    }

    const ndarrayPtr<float, TRAIN_NUM, IMAGE_C, IMAGE_H, IMAGE_W>
        &getTrainImage() {
      return std::move(train_img);
    };

    const ndarrayPtr<float, TEST_NUM, IMAGE_C, IMAGE_H, IMAGE_W>
        &getTestImage() {
      return std::move(test_img);
    };

    const ndarrayPtr<float, TRAIN_NUM, 10> &getTrainLabel() {
      return std::move(train_one_hot_label);
    };
    const ndarrayPtr<float, TEST_NUM, 10> &getTestLabel() {
      return std::move(test_one_hot_label);
    };

   private:
    std::string url_base;
    std::array<std::string, 4> key_files;

    ndarrayPtr<float, TRAIN_NUM, 10> train_one_hot_label;
    ndarrayPtr<float, TEST_NUM, 10> test_one_hot_label;

    ndarrayPtr<float, TRAIN_NUM, IMAGE_C, IMAGE_H, IMAGE_W> train_img;
    ndarrayPtr<float, TEST_NUM, IMAGE_C, IMAGE_H, IMAGE_W> test_img;
  };
}  // namespace dpl

#endif  // DEEP_LEARNING_FROM_SCRATCH_MNIST_HPP
