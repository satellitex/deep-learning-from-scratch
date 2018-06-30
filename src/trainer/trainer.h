//
// Created by TakumiYamashita on 2018/06/10.
//

#ifndef DEEP_LEARNING_FROM_SCRATCH_TRAINER_H
#define DEEP_LEARNING_FROM_SCRATCH_TRAINER_H

#include <memory>
#include "../network/network.hpp"
#include "../optimizer/optimizer.hpp"

namespace dpl {
  template <class Network, class Optimizer, class TrainInput, class TrainLabel,
            class TestInput, class TestLabel>
  class Trainer {
   public:
    Trainer(Network network, Optimizer optimizer, TrainInput x_train,
            TrainLabel t_train, TestInput x_test, TestLabel t_test, int epochs,
            int mini_batch_size, int evaluate_sample_num_per_epoch,
            bool verbose)
        : epochs_(epochs),
          mini_batch_size_(mini_batch_size),
          evaluate_sample_num_per_epoch_(evaluate_sample_num_per_epoch),
          verbose_(verbose) {
      *x_train_ = *x_train;
      *t_train_ = *t_train;
      *x_test_ = *x_test;
      *t_test_ = *t_test;
    }

    void train_step() {
      std::cout << "================= train step ==================="
                << std::endl;
    }
    void train() {
      std::cout << "================= train ===================" << std::endl;
    }

   private:
    Network network_;
    Optimizer optimizer_;
    TrainInput x_train_;
    TrainLabel t_train_;
    TestInput x_test_;
    TestLabel t_test_;
    int epochs_, mini_batch_size_, evaluate_sample_num_per_epoch_;
    bool verbose_;
  };
}  // namespace dpl

#endif  // DEEP_LEARNING_FROM_SCRATCH_TRAINER_H
