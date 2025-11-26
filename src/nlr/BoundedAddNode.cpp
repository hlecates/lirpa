#include "BoundedAddNode.h"
#include <iostream>

namespace NLR {

BoundedAddNode::BoundedAddNode() {
    _input_size = 0;
    _output_size = 0;
    _nodeIndex = 0;
    _nodeName = "add";
}

torch::Tensor BoundedAddNode::forward(const torch::Tensor& input) {
    // Single input forward: add constant if available
    if (_constantValue.defined()) {
        return input + _constantValue;
    }
    return input;
}

torch::Tensor BoundedAddNode::forward(const std::vector<torch::Tensor>& inputs) {
    if (inputs.size() == 1) {
        return forward(inputs[0]);
    } else if (inputs.size() == 2) {
        // Two inputs: x + y
        return inputs[0] + inputs[1];
    } else {
        throw std::runtime_error("BoundedAddNode::forward expects 1 or 2 inputs, got " + std::to_string(inputs.size()));
    }
}

torch::Tensor BoundedAddNode::broadcast_backward(const torch::Tensor& last_A, const BoundedTensor<torch::Tensor>& input) const {
    (void)input; // Suppress unused parameter warning - may be used for broadcasting in the future
    // Simple passthrough for now - broadcasting will be handled by PyTorch
    return last_A;
}

void BoundedAddNode::boundBackward(
    const BoundA& last_lA,
    const BoundA& last_uA,
    const Vector<BoundedTensor<torch::Tensor>>& inputBounds,
    Vector<Pair<BoundA, BoundA>>& outputA_matrices,
    torch::Tensor& lbias,
    torch::Tensor& ubias) {

    // For Add node: the backward bound propagation simply passes through the A matrices
    // Since d(x+y)/dx = 1 and d(x+y)/dy = 1

    // Clear output matrices
    outputA_matrices.clear();

    if (inputBounds.size() == 1) {
        // Single input case (x + constant)
        // The gradient w.r.t. x is 1, so we just pass through the A matrices
        outputA_matrices.append(Pair<BoundA, BoundA>(last_lA, last_uA));

        // Add contribution from constant to bias if we have one
        if (_constantValue.defined()) {
            if (last_lA.isPatches() || last_uA.isPatches()) {
                throw std::runtime_error("BoundedAddNode: Patches mode with constant bias not implemented (requires conversion)");
            }

            torch::Tensor lA_tensor = last_lA.asTensor();
            torch::Tensor uA_tensor = last_uA.asTensor();

            // For x + c, we need to compute A @ c where A is the backward propagation matrix
            // This properly accounts for the linear transformation in the backward pass
            if (lA_tensor.defined()) {
                torch::Tensor constant = _constantValue.flatten();

                // Compute A @ constant to get the bias contribution
                // last_lA has shape (batch, spec, features) or (spec, features)
                // constant has shape (features,)
                torch::Tensor constant_contrib;
                if (lA_tensor.dim() == 3) {
                    // Shape: (batch, spec, features) @ (features,) -> (batch, spec)
                    constant_contrib = torch::matmul(lA_tensor, constant.unsqueeze(-1)).squeeze(-1);
                    // Flatten to (spec,) if batch dimension is 1
                    if (constant_contrib.size(0) == 1) {
                        constant_contrib = constant_contrib.squeeze(0);
                    }
                } else if (lA_tensor.dim() == 2) {
                    // Shape: (spec, features) @ (features,) -> (spec,)
                    constant_contrib = torch::matmul(lA_tensor, constant);
                } else {
                    throw std::runtime_error("BoundedAddNode::boundBackward: unexpected last_lA dimensions");
                }

                if (lbias.defined()) {
                    lbias = lbias + constant_contrib;
                } else {
                    lbias = constant_contrib;
                }
            }

            if (uA_tensor.defined()) {
                torch::Tensor constant = _constantValue.flatten();

                // Compute A @ constant for upper bound
                torch::Tensor constant_contrib;
                if (uA_tensor.dim() == 3) {
                    // Shape: (batch, spec, features) @ (features,) -> (batch, spec)
                    constant_contrib = torch::matmul(uA_tensor, constant.unsqueeze(-1)).squeeze(-1);
                    // Flatten to (spec,) if batch dimension is 1
                    if (constant_contrib.size(0) == 1) {
                        constant_contrib = constant_contrib.squeeze(0);
                    }
                } else if (uA_tensor.dim() == 2) {
                    // Shape: (spec, features) @ (features,) -> (spec,)
                    constant_contrib = torch::matmul(uA_tensor, constant);
                } else {
                    throw std::runtime_error("BoundedAddNode::boundBackward: unexpected last_uA dimensions");
                }

                if (ubias.defined()) {
                    ubias = ubias + constant_contrib;
                } else {
                    ubias = constant_contrib;
                }
            }
        }
    } else if (inputBounds.size() == 2) {
        // Two input case (x + y)
        // Both inputs get the same A matrices since derivatives are 1
        outputA_matrices.append(Pair<BoundA, BoundA>(last_lA, last_uA));
        outputA_matrices.append(Pair<BoundA, BoundA>(last_lA, last_uA));
    } else {
        throw std::runtime_error("BoundedAddNode::boundBackward expects 1 or 2 input bounds");
    }
}

BoundedTensor<torch::Tensor> BoundedAddNode::computeIntervalBoundPropagation(
    const Vector<BoundedTensor<torch::Tensor>>& inputBounds) {

    if (inputBounds.size() == 1) {
        // Single input case (x + constant)
        const auto& x = inputBounds[0];
        if (_constantValue.defined()) {
            // Add constant to both lower and upper bounds
            torch::Tensor lower = x.lower() + _constantValue;
            torch::Tensor upper = x.upper() + _constantValue;
            return BoundedTensor<torch::Tensor>(lower, upper);
        } else {
            // No constant, just pass through
            return x;
        }
    } else if (inputBounds.size() == 2) {
        // Two input case (x + y)
        const auto& x = inputBounds[0];
        const auto& y = inputBounds[1];

        // For addition: [a,b] + [c,d] = [a+c, b+d]
        torch::Tensor lower = x.lower() + y.lower();
        torch::Tensor upper = x.upper() + y.upper();

        return BoundedTensor<torch::Tensor>(lower, upper);
    } else {
        throw std::runtime_error("BoundedAddNode::computeIntervalBoundPropagation expects 1 or 2 input bounds");
    }
}

void BoundedAddNode::setInputSize(unsigned size) {
    _input_size = size;
}

void BoundedAddNode::setOutputSize(unsigned size) {
    _output_size = size;
}

} // namespace NLR
