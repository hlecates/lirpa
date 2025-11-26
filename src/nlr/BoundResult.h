#ifndef __BOUND_RESULT_H__
#define __BOUND_RESULT_H__

#include <torch/torch.h>
#include <memory>
#include <variant>
#include "Patches.h"

namespace NLR {

class BoundA {
public:
    // Can hold either a Tensor or a Patches object
    std::variant<torch::Tensor, std::shared_ptr<Patches>> data;

    BoundA() : data(torch::Tensor()) {}
    BoundA(torch::Tensor t) : data(t) {}
    BoundA(std::shared_ptr<Patches> p) : data(p) {}

    bool isTensor() const {
        return std::holds_alternative<torch::Tensor>(data);
    }

    bool isPatches() const {
        return std::holds_alternative<std::shared_ptr<Patches>>(data);
    }

    torch::Tensor asTensor() const {
        if (isTensor()) return std::get<torch::Tensor>(data);
        return torch::Tensor();
    }
    
    std::shared_ptr<Patches> asPatches() const {
        if (isPatches()) return std::get<std::shared_ptr<Patches>>(data);
        return nullptr;
    }

    bool defined() const {
        if (isTensor()) return std::get<torch::Tensor>(data).defined();
        if (isPatches()) return std::get<std::shared_ptr<Patches>>(data) != nullptr;
        return false;
    }
};

} // namespace NLR

#endif // __BOUND_RESULT_H__

