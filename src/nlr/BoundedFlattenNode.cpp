#include "BoundedFlattenNode.h"

namespace NLR {

BoundedFlattenNode::BoundedFlattenNode(const Operations::FlattenWrapper& flatten_module)
    : _flatten_module(flatten_module) {
    _nodeName = "flatten";  // Set default name
    _nodeIndex = 0;
    _input_size = 0;  // Will be set dynamically
    _output_size = 0; // Will be set dynamically
}

// Standard PyTorch forward pass
torch::Tensor BoundedFlattenNode::forward(const torch::Tensor& input) {
    // Update input/output sizes dynamically
    if (input.dim() > 0) {
        _input_size = input.numel();
        _output_size = input.numel(); // Flatten preserves total number of elements
    }

    // Use the flatten module's forward method
    torch::Tensor output = _flatten_module.forward(input);
    return output;
}

// Auto-LiRPA style boundBackward method
// Flatten operations are identical to Reshape - they don't change the linear relationships, just pass through A matrices
void BoundedFlattenNode::boundBackward(
    const BoundA& last_lA,
    const BoundA& last_uA,
    const Vector<BoundedTensor<torch::Tensor>>& inputBounds,
    Vector<Pair<BoundA, BoundA>>& outputA_matrices,
    torch::Tensor& lbias,
    torch::Tensor& ubias) {

    if (inputBounds.size() < 1) {
        throw std::runtime_error("BoundedFlattenNode expects at least one input");
    }

    if (last_lA.isPatches() || last_uA.isPatches()) {
         throw std::runtime_error("BoundedFlattenNode: Patches propagation not implemented (convert to matrix)");
    }
    
    torch::Tensor lA = last_lA.asTensor();
    torch::Tensor uA = last_uA.asTensor();

    // Flatten operations don't change the linear relationships
    // We need to reshape the A matrices to match the input shape
    auto _bound_oneside = [&](const torch::Tensor& A) -> torch::Tensor {
        if (!A.defined()) {
            return torch::Tensor();
        }

        // Use the stored input shape (excluding batch dimension)
        // If input shape is not set, just pass through A unchanged
        if (_input_shape.empty()) {
            return A;
        }

        // Build new shape: [batch_spec, output_spec, *input_shape[1:]]
        // A has shape [batch_spec, output_spec, flattened_input_dim]
        // We need to reshape the last dimension to match the original input shape
        std::vector<int64_t> new_shape;
        new_shape.push_back(A.size(0));  // batch_spec dimension
        new_shape.push_back(A.size(1));  // output_spec dimension

        // Add input shape dimensions (skip batch dimension at index 0)
        for (size_t i = 1; i < _input_shape.size(); ++i) {
            new_shape.push_back(_input_shape[i]);
        }

        return A.reshape(new_shape);
    };

    // Reshape both A matrices to match input shape
    torch::Tensor reshaped_lA = _bound_oneside(lA);
    torch::Tensor reshaped_uA = _bound_oneside(uA);

    // Pass through the reshaped A matrices
    outputA_matrices.clear();
    outputA_matrices.append(Pair<BoundA, BoundA>(BoundA(reshaped_lA), BoundA(reshaped_uA)));

    // Flatten operations don't add bias - initialize to zeros with correct size
    if (lA.defined()) {
        // Get the output size from the A matrix
        int output_size = lA.size(1); // Second dimension is output size

        if (!lbias.defined()) {
            lbias = torch::zeros({output_size});
        }
    } else {
        if (!lbias.defined()) {
            lbias = torch::zeros({1});
        }
    }

    if (uA.defined()) {
        // Get the output size from the A matrix
        int output_size = uA.size(1); // Second dimension is output size

        if (!ubias.defined()) {
            ubias = torch::zeros({output_size});
        }
    } else {
        if (!ubias.defined()) {
            ubias = torch::zeros({1});
        }
    }
}



// IBP (Interval Bound Propagation): Fast interval-based bound computation for Flatten
BoundedTensor<torch::Tensor> BoundedFlattenNode::computeIntervalBoundPropagation(
    const Vector<BoundedTensor<torch::Tensor>>& inputBounds) {

    if (inputBounds.size() < 1) {
        throw std::runtime_error("Flatten module requires at least one input");
    }

    const auto& inputBoundsPair = inputBounds[0];
    torch::Tensor inputLowerBound = inputBoundsPair.lower();
    torch::Tensor inputUpperBound = inputBoundsPair.upper();

    // Apply flatten to both lower and upper bounds
    torch::Tensor flattenedLower = _flatten_module.forward(inputLowerBound);
    torch::Tensor flattenedUpper = _flatten_module.forward(inputUpperBound);

    return BoundedTensor<torch::Tensor>(flattenedLower, flattenedUpper);
}

void NLR::BoundedFlattenNode::setInputSize(unsigned size) {
    _input_size = size;
}

void NLR::BoundedFlattenNode::setOutputSize(unsigned size) {
    _output_size = size;
}

} // namespace NLR
