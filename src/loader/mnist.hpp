//
// Created by TakumiYamashita on 2018/06/10.
//

#ifndef DEEP_LEARNING_FROM_SCRATCH_MNIST_HPP
#define DEEP_LEARNING_FROM_SCRATCH_MNIST_HPP

#include <string>
#include "../config.hpp"
#include "../primitive/ndarray.hpp"

namespace dpl {
  class Loader {};

  class MNISTLoader {
   private:
    void download_(std::string file) {}

   public:
    MNISTLoader() {
      url_base = MNIST_CONFIG_URL_BASE;
      key_files = {MNIST_CONFIG_TRAIN_IMAGES, MNIST_CONFIG_TRAIN_LABELS,
                   MNIST_CONFIG_TEACHER_IMAGES, MNIST_CONFIG_TEACHER_LABELS};
    }
    void download() {
      for (std::string file : key_files) {
        download_(file);
      }
    }

   private:
    std::string url_base;
    std::array<std::string, 4> key_files;
  };
}  // namespace dpl

#endif  // DEEP_LEARNING_FROM_SCRATCH_MNIST_HPP
