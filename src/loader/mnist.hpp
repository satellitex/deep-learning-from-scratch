//
// Created by TakumiYamashita on 2018/06/10.
//

#ifndef DEEP_LEARNING_FROM_SCRATCH_MNIST_HPP
#define DEEP_LEARNING_FROM_SCRATCH_MNIST_HPP

#include <sys/stat.h>
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
    void download_(std::string file) {
      struct stat st;
      if (!stat(file.c_str(), &st)) {
        std::cout << "already exist " << file << std::endl;
        return;
      }

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
    }

    void load_label_(std::string file) {
      std::ifstream fin(file, std::ios::in | std::ios::binary);
      if (!fin) {
        std::cout << "can not open file : " << file << std::endl;
        return;
      }
      while (!fin.eof()) {
        unsigned char c;
        fin.read((char *)&c, sizeof(unsigned char));
        std::cout << (int)c << " ";
      }
    }

   public:
    MNISTLoader() {
      url_base = MNIST_CONFIG_URL_BASE;
      key_files = {MNIST_CONFIG_TRAIN_IMAGES, MNIST_CONFIG_TRAIN_LABELS,
                   MNIST_CONFIG_TEACHER_IMAGES, MNIST_CONFIG_TEACHER_LABELS};
      dir = MNIST_CONFIG_SAVE_DIR;
    }

    void download() {
      for (std::string file : key_files) {
        download_(file);
      }
    }

    void load() {
      download();
      load_label_(key_files[1]);
    }

   private:
    std::string url_base;
    std::array<std::string, 4> key_files;
    std::string dir;
  };
}  // namespace dpl

#endif  // DEEP_LEARNING_FROM_SCRATCH_MNIST_HPP
