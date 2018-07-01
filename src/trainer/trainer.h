//
// Created by TakumiYamashita on 2018/06/10.
//

#ifndef DEEP_LEARNING_FROM_SCRATCH_TRAINER_H
#define DEEP_LEARNING_FROM_SCRATCH_TRAINER_H

#include <memory>
#include "../network/network.hpp"
#include "../optimizer/optimizer.hpp"

namespace dpl {
  template <int BATCH_SIZE, int EVALUEATE_SAMPLE_NUM_PER_EPOCH, class Network,
            class Optimizer, class TrainInput, class TrainLabel,
            class TestInput, class TestLabel>
  class Trainer {
   public:
    Trainer(const Network& network, const Optimizer& optimizer,
            const TrainInput& x_train, const TrainLabel& t_train,
            const TestInput& x_test, const TestLabel& t_test, int epochs,
            bool verbose)
        : epochs_(epochs), verbose_(verbose) {
      *x_train_ = *x_train;
      *t_train_ = *t_train;
      *x_test_ = *x_test;
      *t_test_ = *t_test;

      iter_per_epoch_ = std::max(TrainInput::GetDim<0>::value / BATCH_SIZE, 1);
      max_iter_ = epochs_ * iter_per_epoch_;
      current_iter_ = 0;
      current_epoch_ = 0;
    }

    void train_step() {
      constexpr int TRAIN_NUM = TrainInput::GetDim<0>::value;
      auto mask = make_ndarray_ptr<bool, TRAIN_NUM>();
      mask->template random_choice<BATCH_SIZE>();

      auto x_batch = x_train_->template choice<BATCH_SIZE>(*mask);
      auto t_batch = t_train_->template choice<BATCH_SIZE>(*mask);

      network_.gradient(x_batch, t_batch);
      optimizer_.update(network_);

      float loss = network_.loss(x_batch, t_batch);
      std::vector<float> train_loss_list;
      train_loss_list.emplace_back(loss);

      std::cout << "train loss : " << loss << std::endl;

      if (current_iter_ % iter_per_epoch_ == 0) {
        current_epoch_++;

        auto x_train_sample_ =
            x_train_->template slice<0, 0, EVALUEATE_SAMPLE_NUM_PER_EPOCH, 1>();
        auto t_train_sample_ =
            t_train_->template slice<0, 0, EVALUEATE_SAMPLE_NUM_PER_EPOCH, 1>();

        auto x_test_sample_ =
            x_test_->template slice<0, 0, EVALUEATE_SAMPLE_NUM_PER_EPOCH, 1>();
        auto t_test_sample_ =
            t_test_->template slice<0, 0, EVALUEATE_SAMPLE_NUM_PER_EPOCH, 1>();

        float train_acc = network_.accuracy(x_train_sample_, t_train_sample_);
        float test_acc = network_.accuracy(x_test_sample_, t_test_sample_);
        train_acc_list_.emplace_back(train_acc);
        test_acc_list_.emplace_back(test_acc);

        std::cout << "==== epoch: " << current_epoch_
                  << ", train acc: " << train_acc << ", test acc : " << test_acc
                  << "====" << std::endl;
      }
      current_iter_++;
    }
    void train() {
      std::cout << "================= train ===================" << std::endl;
      for (int i = 0; i < max_iter_; i++) train_step();

      auto test_acc = network_.accuracy(x_test_, t_test_);
      std::cout << "=============== Final Test Accuracy ==============="
                << std::endl;
      std::cout << "test acc: " << test_acc_ << std::endl;
    }

   private:
    Network network_;
    Optimizer optimizer_;
    TrainInput x_train_;
    TrainLabel t_train_;
    TestInput x_test_;
    TestLabel t_test_;
    int epochs_, evaluate_sample_num_per_epoch_;

    int iter_per_epoch_, max_iter_, current_iter_, current_epoch_;
    std::vector<flaot> train_loss_list_, train_acc_list_, test_acc_list_;
  };
}  // namespace dpl

#endif  // DEEP_LEARNING_FROM_SCRATCH_TRAINER_H
