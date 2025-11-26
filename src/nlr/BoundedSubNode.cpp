#include "BoundedSubNode.h"

namespace NLR {

BoundedSubNode::BoundedSubNode() {
    _nodeName = "sub";  // Set default name
    _nodeIndex = 0;
    _input_size = 0;  // Will be set dynamically
    _output_size = 0; // Will be set dynamically
    _constantIsSecond = true; // Default: x - constant
}

// Standard PyTorch forward pass (single input - with constant if available)
torch::Tensor BoundedSubNode::forward(const torch::Tensor& input) {
    if (_constantValue.defined()) {
        if (_constantIsSecond) {
            // x - constant
            return input - _constantValue;
        } else {
            // constant - x
            return _constantValue - input;
        }
    }
    std::cerr << "[BoundedSubNode] Warning: forward() called with single input and no constant. Returning input unchanged." << std::endl;
    return input;
}

// Multi-input forward pass for Sub: x - y
torch::Tensor BoundedSubNode::forward(const std::vector<torch::Tensor>& inputs) {
    if (inputs.size() != 2) {
        throw std::runtime_error("BoundedSubNode::forward() expects exactly 2 inputs for subtraction (x - y)");
    }

    const torch::Tensor& x = inputs[0];
    const torch::Tensor& y = inputs[1];

    // Perform element-wise subtraction (PyTorch handles broadcasting)
    torch::Tensor result = x - y;

    // Update sizes dynamically
    if (x.dim() > 0) {
        _input_size = x.numel();
        _output_size = result.numel();
    }

    return result;
}

// CROWN Backward Mode bound propagation
// For Sub: output = x - y
// Lower bound: lA @ lower(x) - uA @ upper(y)
// Upper bound: uA @ upper(x) - lA @ lower(y)
void BoundedSubNode::boundBackward(
    const BoundA& last_lA,
    const BoundA& last_uA,
    const Vector<BoundedTensor<torch::Tensor>>& inputBounds,
    Vector<Pair<BoundA, BoundA>>& outputA_matrices,
    torch::Tensor& lbias,
    torch::Tensor& ubias) {

    // Clear and prepare output A matrices
    outputA_matrices.clear();

    if (inputBounds.size() == 1) {
        // Single input case with constant
        if (_constantValue.defined()) {
            if (_constantIsSecond) {
                // x - constant: gradient w.r.t. x is 1
                outputA_matrices.append(Pair<BoundA, BoundA>(last_lA, last_uA));

                // Contribution to bias from constant: compute A @ (-constant)
                if (last_lA.isPatches() || last_uA.isPatches()) {
                    throw std::runtime_error("BoundedSubNode: Patches mode with constant bias not implemented (requires conversion)");
                }

                torch::Tensor last_lA_tensor = last_lA.asTensor();
                torch::Tensor last_uA_tensor = last_uA.asTensor();

                if (last_lA_tensor.defined()) {
                    torch::Tensor constant = _constantValue.flatten();

                    // Compute -A @ constant to get the bias contribution
                    torch::Tensor constant_contrib;
                    if (last_lA_tensor.dim() == 3) {
                        // Shape: (batch, spec, features) @ (features,) -> (batch, spec)
                        constant_contrib = torch::matmul(last_lA_tensor, constant.unsqueeze(-1)).squeeze(-1);
                        if (constant_contrib.size(0) == 1) {
                            constant_contrib = constant_contrib.squeeze(0);
                        }
                    } else if (last_lA_tensor.dim() == 2) {
                        // Shape: (spec, features) @ (features,) -> (spec,)
                        constant_contrib = torch::matmul(last_lA_tensor, constant);
                    } else {
                        throw std::runtime_error("BoundedSubNode::boundBackward: unexpected last_lA dimensions");
                    }

                    // Negate because we're subtracting the constant
                    lbias = -constant_contrib;
                }

                if (last_uA_tensor.defined()) {
                    torch::Tensor constant = _constantValue.flatten();

                    // Compute -A @ constant for upper bound
                    torch::Tensor constant_contrib;
                    if (last_uA_tensor.dim() == 3) {
                        // Shape: (batch, spec, features) @ (features,) -> (batch, spec)
                        constant_contrib = torch::matmul(last_uA_tensor, constant.unsqueeze(-1)).squeeze(-1);
                        if (constant_contrib.size(0) == 1) {
                            constant_contrib = constant_contrib.squeeze(0);
                        }
                    } else if (last_uA_tensor.dim() == 2) {
                        // Shape: (spec, features) @ (features,) -> (spec,)
                        constant_contrib = torch::matmul(last_uA_tensor, constant);
                    } else {
                        throw std::runtime_error("BoundedSubNode::boundBackward: unexpected last_uA dimensions");
                    }

                    // Negate because we're subtracting the constant
                    ubias = -constant_contrib;
                }
            } else {
                // constant - x: gradient w.r.t. x is -1, so negate the A matrices
                BoundA neg_lA, neg_uA;
                
                if (last_lA.isTensor()) {
                    neg_lA = last_lA.asTensor().defined() ? BoundA(-last_lA.asTensor()) : BoundA();
                } else {
                    auto p = last_lA.asPatches();
                    neg_lA = BoundA(p->create_similar(-p->patches));
                }
                
                if (last_uA.isTensor()) {
                    neg_uA = last_uA.asTensor().defined() ? BoundA(-last_uA.asTensor()) : BoundA();
                } else {
                    auto p = last_uA.asPatches();
                    neg_uA = BoundA(p->create_similar(-p->patches));
                }
                
                outputA_matrices.append(Pair<BoundA, BoundA>(neg_lA, neg_uA));

                // Contribution to bias from constant: compute A @ constant
                if (last_lA.isPatches() || last_uA.isPatches()) {
                    throw std::runtime_error("BoundedSubNode: Patches mode with constant bias not implemented (requires conversion)");
                }
                
                torch::Tensor last_lA_tensor = last_lA.asTensor();
                torch::Tensor last_uA_tensor = last_uA.asTensor();

                if (last_lA_tensor.defined()) {
                    torch::Tensor constant = _constantValue.flatten();

                    // Compute A @ constant to get the bias contribution
                    torch::Tensor constant_contrib;
                    if (last_lA_tensor.dim() == 3) {
                        constant_contrib = torch::matmul(last_lA_tensor, constant.unsqueeze(-1)).squeeze(-1);
                        if (constant_contrib.size(0) == 1) constant_contrib = constant_contrib.squeeze(0);
                    } else if (last_lA_tensor.dim() == 2) {
                        constant_contrib = torch::matmul(last_lA_tensor, constant);
                    } else {
                        throw std::runtime_error("BoundedSubNode::boundBackward: unexpected last_lA dimensions");
                    }

                    // Positive because constant comes first
                    lbias = constant_contrib;
                }

                if (last_uA_tensor.defined()) {
                    torch::Tensor constant = _constantValue.flatten();

                    // Compute A @ constant for upper bound
                    torch::Tensor constant_contrib;
                    if (last_uA_tensor.dim() == 3) {
                        constant_contrib = torch::matmul(last_uA_tensor, constant.unsqueeze(-1)).squeeze(-1);
                        if (constant_contrib.size(0) == 1) constant_contrib = constant_contrib.squeeze(0);
                    } else if (last_uA_tensor.dim() == 2) {
                        constant_contrib = torch::matmul(last_uA_tensor, constant);
                    } else {
                        throw std::runtime_error("BoundedSubNode::boundBackward: unexpected last_uA dimensions");
                    }

                    // Positive because constant comes first
                    ubias = constant_contrib;
                }
            }
        } else {
            throw std::runtime_error("BoundedSubNode::boundBackward with 1 input requires a constant value");
        }
    } else if (inputBounds.size() >= 2) {
        // Two input case
        // First input (x): positive sign, normal A matrices
        outputA_matrices.append(Pair<BoundA, BoundA>(last_lA, last_uA));

        // Second input (y): negative sign, SWAPPED A matrices!
        // For lower bound of (x - y): lA @ lower(x) - uA @ upper(y) → use -uA for y
        // For upper bound of (x - y): uA @ upper(x) - lA @ lower(y) → use -lA for y
        
        BoundA neg_lA, neg_uA;
        
        if (last_lA.isTensor()) {
            neg_lA = last_lA.asTensor().defined() ? BoundA(-last_lA.asTensor()) : BoundA();
        } else {
            auto p = last_lA.asPatches();
            neg_lA = BoundA(p->create_similar(-p->patches));
        }
        
        if (last_uA.isTensor()) {
            neg_uA = last_uA.asTensor().defined() ? BoundA(-last_uA.asTensor()) : BoundA();
        } else {
            auto p = last_uA.asPatches();
            neg_uA = BoundA(p->create_similar(-p->patches));
        }

        // CRITICAL: Swap neg_uA and neg_lA when appending!
        outputA_matrices.append(Pair<BoundA, BoundA>(neg_uA, neg_lA));
    } else {
        throw std::runtime_error("BoundedSubNode::boundBackward expects at least 1 input bound");
    }

    // Initialize bias if needed - but DON'T override if already set by constant handling above
    // Only handling tensor bias initialization for now
    
    if (last_lA.isPatches() || last_uA.isPatches()) {
        // Bias calculation for patches not fully implemented (relies on constant logic or passed-in)
        // But here we are initializing bias.
        // auto_LiRPA says: "sum_bias = 0" if not constant.
        // So if bias not defined, we can set to 0.
        
        if (!lbias.defined()) {
            // We need to know output size to init zero bias?
            // Patches mode generally handles bias differently (propagated as sum_bias).
            // If lbias is passed by reference, we can leave it undefined if 0?
            // auto_LiRPA returns 0 for sum_bias if no bias.
            // But `boundBackward` signature has `torch::Tensor& lbias`.
            // If we return undefined tensor, caller might fail.
            // Caller accumulates bias. undefined + undefined = undefined.
            // undefined + tensor = tensor.
            // So returning undefined is fine if it means 0.
        }
        
    } else {
        // Existing tensor logic
        torch::Tensor last_lA_tensor = last_lA.asTensor();
        torch::Tensor last_uA_tensor = last_uA.asTensor();
        
        // ... same code as before ...
        if (last_lA_tensor.defined()) {
            int output_size = last_lA_tensor.size(1); 
            if (!lbias.defined()) {
                lbias = torch::zeros({output_size});
            } else if (lbias.numel() != output_size) {
                if (lbias.numel() == 1) lbias = lbias.expand({output_size});
            }
        } else {
            if (!lbias.defined()) lbias = torch::zeros({1});
        }

        if (last_uA_tensor.defined()) {
            int output_size = last_uA_tensor.size(1);
            if (!ubias.defined()) {
                ubias = torch::zeros({output_size});
            } else if (ubias.numel() != output_size) {
                if (ubias.numel() == 1) ubias = ubias.expand({output_size});
            }
        } else {
            if (!ubias.defined()) ubias = torch::zeros({1});
        }
    }
}

// IBP (Interval Bound Propagation): Fast interval-based bound computation for Sub
// For Sub: output = x - y
// Lower bound: lower(x) - upper(y)
// Upper bound: upper(x) - lower(y)
BoundedTensor<torch::Tensor> BoundedSubNode::computeIntervalBoundPropagation(
    const Vector<BoundedTensor<torch::Tensor>>& inputBounds) {

    if (inputBounds.size() == 1) {
        // Single input case with constant
        const auto& xBounds = inputBounds[0];
        if (_constantValue.defined()) {
            if (_constantIsSecond) {
                // x - constant: [lower(x) - constant, upper(x) - constant]
                torch::Tensor resultLower = xBounds.lower() - _constantValue;
                torch::Tensor resultUpper = xBounds.upper() - _constantValue;
                return BoundedTensor<torch::Tensor>(resultLower, resultUpper);
            } else {
                // constant - x: [constant - upper(x), constant - lower(x)]
                torch::Tensor resultLower = _constantValue - xBounds.upper();
                torch::Tensor resultUpper = _constantValue - xBounds.lower();
                return BoundedTensor<torch::Tensor>(resultLower, resultUpper);
            }
        } else {
            throw std::runtime_error("BoundedSubNode::computeIntervalBoundPropagation with 1 input requires a constant value");
        }
    } else if (inputBounds.size() >= 2) {
        const auto& xBounds = inputBounds[0];
        const auto& yBounds = inputBounds[1];

        torch::Tensor xLower = xBounds.lower();
        torch::Tensor xUpper = xBounds.upper();
        torch::Tensor yLower = yBounds.lower();
        torch::Tensor yUpper = yBounds.upper();

        // For subtraction: [lower(x) - upper(y), upper(x) - lower(y)]
        torch::Tensor resultLower = xLower - yUpper;
        torch::Tensor resultUpper = xUpper - yLower;

        return BoundedTensor<torch::Tensor>(resultLower, resultUpper);
    } else {
        throw std::runtime_error("BoundedSubNode::computeIntervalBoundPropagation requires at least 1 input");
    }
}

void BoundedSubNode::setInputSize(unsigned size) {
    _input_size = size;
}

void BoundedSubNode::setOutputSize(unsigned size) {
    _output_size = size;
}

} // namespace NLR
