// BoundedConcatNode.cpp - Concatenation with bound propagation
#include "BoundedConcatNode.h"
#include <torch/torch.h>
#include <cmath>

namespace NLR {

BoundedConcatNode::BoundedConcatNode(int axis, unsigned numInputs, const String& name)
    : _axis(axis), _numInputs(numInputs), _nodeName(name), _nodeIndex(0), 
      _input_size(0), _output_size(0) {
}

torch::Tensor BoundedConcatNode::forward(const torch::Tensor& input) {
    // Single input case - just return it
    return input;
}

torch::Tensor BoundedConcatNode::forward(const std::vector<torch::Tensor>& inputs) {
    if (inputs.empty()) {
        throw std::runtime_error("BoundedConcatNode::forward - no inputs provided");
    }

    if (inputs.size() == 1) {
        return inputs[0];
    }

    // Adjust axis for actual tensor dimensions (batch dim may be missing)
    int64_t axis = _axis;
    int64_t ndim = inputs[0].dim();

    // Make axis non-negative
    if (axis < 0) {
        axis += ndim;
    }

    // If axis is still out of range, adjust by -1 (batch dimension was removed)
    if (axis >= ndim && axis > 0) {
        axis = axis - 1;
    }

    // Ensure axis is valid
    if (axis < 0 || axis >= ndim) {
        throw std::runtime_error("BoundedConcatNode::forward - axis out of range");
    }

    // Concatenate all inputs along the adjusted axis
    torch::Tensor result = torch::cat(inputs, axis);
    return result;
}

void BoundedConcatNode::boundBackward(
    const BoundA& last_lA,
    const BoundA& last_uA,
    const Vector<BoundedTensor<torch::Tensor>>& inputBounds,
    Vector<Pair<BoundA, BoundA>>& outputA_matrices,
    torch::Tensor& lbias,
    torch::Tensor& ubias) {

    // Concatenation backward: split the A matrix along concat axis
    // and distribute chunks to each input

    if (inputBounds.size() != _numInputs) {
        throw std::runtime_error("BoundedConcatNode: input count mismatch");
    }

    // If _input_sizes is incomplete, extract from inputBounds
    // This is more robust than relying on shape metadata from parsing
    if (_input_sizes.size() != _numInputs) {
        _input_sizes.clear();
        for (unsigned i = 0; i < _numInputs; ++i) {
            if (inputBounds[i].lower().defined()) {
                torch::Tensor lb = inputBounds[i].lower();
                // Compute effective axis (may need adjustment if batch dim is missing)
                int64_t effective_axis = _axis;
                if (_axis < 0) {
                    effective_axis = _axis + lb.dim();
                } else if (_axis >= lb.dim() && _axis > 0) {
                    effective_axis = _axis - 1;  // batch dim was removed
                }
                if (effective_axis >= 0 && effective_axis < lb.dim()) {
                    _input_sizes.push_back(static_cast<unsigned>(lb.size(effective_axis)));
                } else {
                    _input_sizes.push_back(1);  // fallback
                }
            } else {
                _input_sizes.push_back(1);  // fallback
            }
        }
    }

    // Initialize output A matrices for each input
    outputA_matrices.clear();
    for (unsigned i = 0; i < _numInputs; ++i) {
        outputA_matrices.append(Pair<BoundA, BoundA>(BoundA(), BoundA()));
    }
    
    // Split A matrices along the concatenation axis
    // Following auto_LiRPA logic: split(A, input_size, dim=axis+1)
    auto splitA = [&](const BoundA& A, const char* name) -> std::vector<BoundA> {
        std::vector<BoundA> result;

        if (!A.defined() || !A.isTensor()) {
            // Return empty BoundA for each input
            for (unsigned i = 0; i < _numInputs; ++i) {
                result.push_back(BoundA());
            }
            return result;
        }

        torch::Tensor A_tensor = A.asTensor();

        // In auto_LiRPA, A matrix is split along dimension (axis + 1)
        // But we need to adjust for the actual dimensions of A_tensor
        // A_tensor might be [spec, ...] without explicit batch dimension
        int split_dim = _axis + 1;

        // Adjust split_dim if it's out of range (batch dim might be missing)
        if (split_dim >= A_tensor.dim() && split_dim > 0) {
            split_dim = _axis;  // Try without the +1 offset
        }

        (void)name; // Suppress unused warning

        // Get sizes along split dimension for each input
        std::vector<int64_t> split_sizes;
        int64_t total_size = A_tensor.size(split_dim);

        // Check if we have complete input size information
        if (_input_sizes.empty() || _input_sizes.size() != _numInputs) {
            // Equal split if sizes not provided or incomplete
            // This is a fallback when shape metadata is not available
            int64_t chunk_size = total_size / _numInputs;
            int64_t remainder = total_size % _numInputs;
            for (unsigned i = 0; i < _numInputs; ++i) {
                // Distribute remainder to first chunks to ensure sum equals total
                split_sizes.push_back(chunk_size + (i < static_cast<unsigned>(remainder) ? 1 : 0));
            }
        } else {
            // Use provided input sizes (sizes along concat axis)
            int64_t sum_sizes = 0;
            for (unsigned size : _input_sizes) {
                sum_sizes += size;
            }

            // Check if input_sizes match the actual A tensor dimension
            if (sum_sizes == total_size) {
                // Perfect match - use input_sizes directly
                for (unsigned size : _input_sizes) {
                    split_sizes.push_back(static_cast<int64_t>(size));
                }
            } else {
                // Mismatch - split proportionally based on input_sizes ratios
                // This happens when the A matrix represents a subset/projection of the output
                double scale = static_cast<double>(total_size) / sum_sizes;
                int64_t accumulated = 0;
                for (unsigned i = 0; i < _input_sizes.size(); ++i) {
                    if (i == _input_sizes.size() - 1) {
                        // Last chunk gets remainder to ensure exact sum
                        split_sizes.push_back(total_size - accumulated);
                    } else {
                        int64_t chunk = static_cast<int64_t>(std::round(_input_sizes[i] * scale));
                        split_sizes.push_back(chunk);
                        accumulated += chunk;
                    }
                }
            }
        }
        
        // Split the tensor
        auto chunks = torch::split(A_tensor, split_sizes, split_dim);

        for (size_t i = 0; i < chunks.size(); ++i) {
            result.push_back(BoundA(chunks[i]));
        }

        return result;
    };
    
    std::vector<BoundA> lA_chunks = splitA(last_lA, "lA");
    std::vector<BoundA> uA_chunks = splitA(last_uA, "uA");
    
    // Assign chunks to output matrices
    for (unsigned i = 0; i < _numInputs && i < lA_chunks.size(); ++i) {
        outputA_matrices[i] = Pair<BoundA, BoundA>(lA_chunks[i], uA_chunks[i]);
    }
    
    // Bias terms pass through unchanged
    auto options = last_lA.defined() && last_lA.isTensor()
        ? last_lA.asTensor().options()
        : (last_uA.defined() && last_uA.isTensor() ? last_uA.asTensor().options()
                                                   : torch::TensorOptions().dtype(torch::kFloat32).device(_device));
    lbias = torch::zeros({1}, options);
    ubias = torch::zeros({1}, options);
}

BoundedTensor<torch::Tensor> BoundedConcatNode::computeIntervalBoundPropagation(
    const Vector<BoundedTensor<torch::Tensor>>& inputBounds) {
    
    if (inputBounds.size() != _numInputs) {
        throw std::runtime_error("BoundedConcatNode: IBP input count mismatch");
    }
    
    if (inputBounds.empty()) {
        throw std::runtime_error("BoundedConcatNode: no inputs for IBP");
    }
    
    // Single input case
    if (inputBounds.size() == 1) {
        return inputBounds[0];
    }
    
    // Collect lower and upper bounds from all inputs
    std::vector<torch::Tensor> lower_bounds;
    std::vector<torch::Tensor> upper_bounds;
    
    for (unsigned i = 0; i < inputBounds.size(); ++i) {
        lower_bounds.push_back(inputBounds[i].lower());
        upper_bounds.push_back(inputBounds[i].upper());
    }
    
    // Make axis non-negative and adjust for actual tensor dimensions
    // In auto_LiRPA, axis is made non-negative in forward() before concatenation
    // The axis refers to the dimension in the actual tensor (which may not have batch dim)
    int64_t axis = _axis;
    if (!lower_bounds.empty()) {
        int64_t ndim = lower_bounds[0].dim();
        
        // Make axis non-negative
        if (axis < 0) {
            axis += ndim;
        }
        
        // If axis is still out of range, it might be because the batch dimension
        // was removed. Try adjusting by -1 if axis > 0
        if (axis >= ndim && axis > 0) {
            axis = axis - 1;
        }
        
        // Ensure axis is valid after adjustment
        if (axis < 0 || axis >= ndim) {
            throw std::runtime_error("BoundedConcatNode: axis out of range");
        }
    }
    
    // Concatenate bounds along the specified axis
    torch::Tensor concat_lower = torch::cat(lower_bounds, axis);
    torch::Tensor concat_upper = torch::cat(upper_bounds, axis);
    
    return BoundedTensor<torch::Tensor>(concat_lower, concat_upper);
}

} // namespace NLR
