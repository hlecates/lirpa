// BoundedSliceNode.cpp - Slice operation with bound propagation
#include "BoundedSliceNode.h"
#include <torch/torch.h>
#include <algorithm>
#include <cmath>

namespace NLR {

// ONNX -inf value used in Slice operations
constexpr int64_t ONNX_NEG_INF = -9223372036854775807LL;

BoundedSliceNode::BoundedSliceNode(int64_t start, int64_t end, int64_t axes, int64_t steps,
                                   const String& name)
    : _start(start), _end(end), _axes(axes), _steps(steps),
      _nodeName(name), _nodeIndex(0), _input_size(0), _output_size(0) {
}

std::pair<int64_t, int64_t> BoundedSliceNode::_fixup_params(
    const std::vector<int64_t>& shape, int64_t start, int64_t end,
    int64_t axes, int64_t steps) const {

    // Handle negative start index
    if (start < 0) {
        start += shape[axes];
    }

    // Handle negative end index
    if (end < 0) {
        if (end == ONNX_NEG_INF) {
            // -inf in ONNX: only possible when step == -1
            end = 0;
        } else {
            end += shape[axes];
        }
    }

    // Handle negative step by swapping start and end
    if (steps == -1) {
        std::swap(start, end);
        end = end + 1;  // Adjust end for proper slicing
    }

    // Clamp end to input shape size
    end = std::min(end, shape[axes]);

    // Ensure start is valid
    start = std::max(start, static_cast<int64_t>(0));

    return {start, end};
}

torch::Tensor BoundedSliceNode::forward(const torch::Tensor& input) {
    // Cache the input shape for backward pass
    _input_shape.clear();
    for (int64_t i = 0; i < input.dim(); ++i) {
        _input_shape.push_back(input.size(i));
    }

    // Adjust axis for tensors that may have had batch dimension removed
    // ONNX models often specify axis with batch dimension included,
    // but during IBP/CROWN the batch dimension may be implicit
    int64_t effective_axes = _axes;
    if (_axes > 0 && _axes >= input.dim()) {
        // The specified axis is out of range - likely the batch dimension was removed
        // Adjust by subtracting 1
        effective_axes = _axes - 1;
    }
    if (effective_axes < 0) {
        effective_axes += input.dim();
    }

    // Fix up parameters based on input shape with adjusted axis
    std::vector<int64_t> shape_for_fixup = _input_shape;
    auto [fixed_start, fixed_end] = _fixup_params(shape_for_fixup, _start, _end, effective_axes, _steps);

    // Compute slice length
    int64_t length = fixed_end - fixed_start;
    if (length <= 0) {
        // Return empty tensor with appropriate shape
        std::vector<int64_t> output_shape = _input_shape;
        output_shape[effective_axes] = 0;
        return torch::zeros(output_shape, input.options());
    }

    // Use torch::narrow for slicing (equivalent to Python slice)
    torch::Tensor result = torch::narrow(input, static_cast<int>(effective_axes),
                                         fixed_start, length);

    // If step is -1, flip the result along the axis
    if (_steps == -1) {
        result = torch::flip(result, {static_cast<int>(effective_axes)});
    }

    return result;
}

torch::Tensor BoundedSliceNode::forward(const std::vector<torch::Tensor>& inputs) {
    // Slice typically has one data input (the tensor to slice)
    // Additional inputs may be constants for start, end, axes, steps (ONNX >= 13)
    // We handle those during parsing, so here we just use the first input
    if (inputs.empty()) {
        throw std::runtime_error("BoundedSliceNode::forward - no inputs provided");
    }
    return forward(inputs[0]);
}

BoundedTensor<torch::Tensor> BoundedSliceNode::computeIntervalBoundPropagation(
    const Vector<BoundedTensor<torch::Tensor>>& inputBounds) {

    if (inputBounds.empty()) {
        throw std::runtime_error("BoundedSliceNode: no inputs for IBP");
    }

    // Get the main input bounds (first input is the data tensor)
    const auto& dataBounds = inputBounds[0];

    // Slice both lower and upper bounds using the same forward operation
    // In auto_LiRPA: return Interval.make_interval(self.forward(*lb), self.forward(*ub))
    torch::Tensor sliced_lower = forward(dataBounds.lower());
    torch::Tensor sliced_upper = forward(dataBounds.upper());

    return BoundedTensor<torch::Tensor>(sliced_lower, sliced_upper);
}

void BoundedSliceNode::boundBackward(
    const BoundA& last_lA,
    const BoundA& last_uA,
    const Vector<BoundedTensor<torch::Tensor>>& inputBounds,
    Vector<Pair<BoundA, BoundA>>& outputA_matrices,
    torch::Tensor& lbias,
    torch::Tensor& ubias) {

    // Ensure we have the input shape cached
    if (_input_shape.empty() && !inputBounds.empty()) {
        // Try to get shape from input bounds
        const auto& dataBounds = inputBounds[0];
        torch::Tensor lower = dataBounds.lower();
        for (int64_t i = 0; i < lower.dim(); ++i) {
            _input_shape.push_back(lower.size(i));
        }
    }

    if (_input_shape.empty()) {
        throw std::runtime_error("BoundedSliceNode: input shape not available for backward pass");
    }

    // Adjust axis for tensors that may have had batch dimension removed
    // (same logic as in forward pass)
    int64_t effective_axes = _axes;
    int64_t input_ndim = static_cast<int64_t>(_input_shape.size());
    if (_axes > 0 && _axes >= input_ndim) {
        effective_axes = _axes - 1;
    }
    if (effective_axes < 0) {
        effective_axes += input_ndim;
    }

    // Fix up parameters with adjusted axis
    std::pair<int64_t, int64_t> fixed_params = _fixup_params(_input_shape, _start, _end, effective_axes, _steps);
    int64_t fixed_start = fixed_params.first;
    int64_t fixed_end = fixed_params.second;

    // Helper lambda for backward propagation on one side (lA or uA)
    // Based on auto_LiRPA's _bound_oneside function
    auto _bound_oneside = [this, fixed_start, fixed_end, effective_axes](const BoundA& A, const char* /*name*/) -> BoundA {
        if (!A.defined() || !A.isTensor()) {
            return BoundA();
        }

        torch::Tensor A_tensor = A.asTensor();

        // Create the output shape for the expanded A matrix
        // A matrix has shape [spec, batch, ...] where ... matches output shape
        // We need to expand it to match input shape

        // Build the target shape: A_shape[:2] + input_shape[:]
        // For backward pass, we preserve the A matrix structure and expand along the slice axis
        std::vector<int64_t> new_A_shape;

        // Copy first two dimensions (spec and batch)
        if (A_tensor.dim() >= 2) {
            new_A_shape.push_back(A_tensor.size(0));
            new_A_shape.push_back(A_tensor.size(1));
        } else if (A_tensor.dim() == 1) {
            new_A_shape.push_back(A_tensor.size(0));
        }

        // Add input shape dimensions
        for (size_t i = 0; i < _input_shape.size(); ++i) {
            new_A_shape.push_back(_input_shape[i]);
        }

        // Create zero tensor with expanded shape
        torch::Tensor new_A = torch::zeros(new_A_shape, A_tensor.options());

        // Compute the dimension in new_A corresponding to the slice axis
        // A matrix has [spec, batch, ...data_dims...]
        // So the slice axis in A is: 2 + effective_axes (if batch is present in data)
        // or 2 + effective_axes - 1 (if batch is not in input_shape)
        int64_t dim = 2 + effective_axes;
        if (dim < 0) {
            dim += new_A.dim();
        }

        // Ensure dim is within valid range
        if (dim < 0 || dim >= new_A.dim()) {
            throw std::runtime_error("BoundedSliceNode: invalid dimension for index_copy");
        }

        // Create indices for the slice region
        torch::Tensor indices = torch::arange(fixed_start, fixed_end,
                                               torch::TensorOptions()
                                                   .dtype(torch::kLong)
                                                   .device(A_tensor.device()));

        // If step is -1, we need to reverse the order (the A matrix comes from
        // the flipped output, so we need to flip back when placing in new_A)
        torch::Tensor source = A_tensor;
        if (_steps == -1) {
            source = torch::flip(A_tensor, {static_cast<int>(dim)});
        }

        // Use index_copy to place the slice data into the correct positions
        new_A = torch::index_copy(new_A, dim, indices, source);

        return BoundA(new_A);
    };

    // Apply backward propagation to both lA and uA
    BoundA lA = _bound_oneside(last_lA, "lA");
    BoundA uA = _bound_oneside(last_uA, "uA");

    // Initialize output A matrices
    // Slice has multiple "inputs" in ONNX (data, starts, ends, axes, steps)
    // but only the first one (data) gets a non-None A matrix
    outputA_matrices.clear();

    // First input (data tensor) gets the computed A matrices
    outputA_matrices.append(Pair<BoundA, BoundA>(lA, uA));

    // Additional inputs (starts, ends, axes, steps) get None A matrices
    // These are typically 4 more inputs in ONNX format
    for (unsigned i = 1; i < 5; ++i) {
        outputA_matrices.append(Pair<BoundA, BoundA>(BoundA(), BoundA()));
    }

    // Bias terms are zero for slice operation (no additive bias)
    auto options = last_lA.defined() && last_lA.isTensor()
        ? last_lA.asTensor().options()
        : (last_uA.defined() && last_uA.isTensor()
           ? last_uA.asTensor().options()
           : torch::TensorOptions().dtype(torch::kFloat32).device(_device));

    lbias = torch::zeros({1}, options);
    ubias = torch::zeros({1}, options);
}

} // namespace NLR
