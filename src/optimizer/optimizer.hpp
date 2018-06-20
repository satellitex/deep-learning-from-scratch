//
// Created by TakumiYamashita on 2018/06/10.
//

#ifndef DEEP_LEARNING_FROM_SCRATCH_OPTIMIZER_HPP
#define DEEP_LEARNING_FROM_SCRATCH_OPTIMIZER_HPP

#include "../primitive/primitive.hpp"

using ndarray = dpl::ndarray;

namespace dpl {
    class Optimizer {
        virtual void update(std::shared_ptr<ndarray> params, const ndarray& grads) = 0;
    };
}

#endif //DEEP_LEARNING_FROM_SCRATCH_OPTIMIZER_HPP
