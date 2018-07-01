//
// Created by TakumiYamashita on 2018/06/10.
//

#ifndef DEEP_LEARNING_FROM_SCRATCH_TRAINER_HPP
#define DEEP_LEARNING_FROM_SCRATCH_TRAINER_HPP

#include <memory>
#include "../src/network/network.hpp"
#include "../src/optimizer/optimizer.hpp"
#include "../src/primitive/ndarray.hpp"

namespace dpl {
  template <int BATCH_SIZE, int EVALUEATE_SAMPLE_NUM_PER_EPOCH, class NETWORK,
            class OPTIMZIER, class TRAIN_INPUT, class TRAIN_LABEL,
            class TEST_INPUT, class TEST_LABEL>
  class Trainer;

  template <int BATCH_SIZE, int EVALUEATE_SAMPLE_NUM_PER_EPOCH, class... Layers,
            class Optimizer, int... TrainInputArgs, int... TrainLabelArgs,
            int... TestInputArgs, int... TestLabelArgs>
  class Trainer<
      BATCH_SIZE, EVALUEATE_SAMPLE_NUM_PER_EPOCH, NetworkPtr<Layers...>,
      Optimizer, ndarrayPtr<float, TrainInputArgs...>,
      ndarrayPtr<float, TrainLabelArgs...>, ndarrayPtr<float, TestInputArgs...>,
      ndarrayPtr<float, TestLabelArgs...>> {
   public:
    Trainer(NetworkPtr<Layers...> network, const Optimizer& optimizer,
            ndarrayPtr<float, TrainInputArgs...> x_train,
            ndarrayPtr<float, TrainLabelArgs...> t_train,
            ndarrayPtr<float, TestInputArgs...> x_test,
            ndarrayPtr<float, TestLabelArgs...> t_test, int epochs)
        : epochs_(epochs) {
      network_ = std::move(network);
      x_train_ = std::move(x_train);
      t_train_ = std::move(t_train);

      x_test_ = std::move(x_test);
      t_test_ = std::move(t_test);

      iter_per_epoch_ =
          std::max(Get<0, TrainInputArgs...>::value / BATCH_SIZE, 1);
      max_iter_ = epochs_ * iter_per_epoch_;
      current_iter_ = 0;
      current_epoch_ = 0;
    }

    void train_step() {
      constexpr int TRAIN_NUM = Get<0, TrainInputArgs...>::value;
      auto mask = make_ndarray_ptr<bool, TRAIN_NUM>();
      mask->template random_mask<BATCH_SIZE>();

      auto x_batch = x_train_->template choice<BATCH_SIZE>(*mask);
      auto t_batch = t_train_->template choice<BATCH_SIZE>(*mask);

      network_->gradient(x_batch, t_batch);
      optimizer_.update(*network_);

      float loss = network_->loss(x_batch, t_batch);
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

        float train_acc = network_->template accuracy<BATCH_SIZE>(
            x_train_sample_, t_train_sample_);
        float test_acc = network_->template accuracy<BATCH_SIZE>(
            x_test_sample_, t_test_sample_);
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

      auto test_acc = network_->template accuracy<BATCH_SIZE>(x_test_, t_test_);
      std::cout << "=============== Final Test Accuracy ==============="
                << std::endl;
      std::cout << "test acc: " << test_acc << std::endl;
    }

   private:
    NetworkPtr<Layers...> network_;
    Optimizer optimizer_;
    ndarrayPtr<float, TrainInputArgs...> x_train_;
    ndarrayPtr<float, TrainLabelArgs...> t_train_;
    ndarrayPtr<float, TestInputArgs...> x_test_;
    ndarrayPtr<float, TestLabelArgs...> t_test_;
    int epochs_, evaluate_sample_num_per_epoch_;

    int iter_per_epoch_, max_iter_, current_iter_, current_epoch_;
    std::vector<float> train_loss_list_, train_acc_list_, test_acc_list_;
  };
}  // namespace dpl

#endif  // DEEP_LEARNING_FROM_SCRATCH_TRAINER_HPP
