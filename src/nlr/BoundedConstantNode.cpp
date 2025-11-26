// Marabou/src/nlr/BoundedConstantNode.cpp
#include "BoundedConstantNode.h"

NLR::BoundedConstantNode::BoundedConstantNode(const torch::Tensor& constantValue, const String& name)
    : _constantValue(constantValue) {
    _nodeName = name;
    _nodeIndex = 0;
}

torch::Tensor NLR::BoundedConstantNode::forward(const torch::Tensor& input) {
    (void)input; // Suppress unused parameter warning
    return _constantValue;
}

void NLR::BoundedConstantNode::boundBackward(
    const BoundA& last_lA,
    const BoundA& last_uA,
    const Vector<BoundedTensor<torch::Tensor>>& inputBounds,
    Vector<Pair<BoundA, BoundA>>& outputA_matrices,
    torch::Tensor& lbias,
    torch::Tensor& ubias
) {
    (void)inputBounds; // Suppress unused parameter warning

    // Constants add to bias, zero A matrices
    // The bias size should match the spec dimension of the incoming A matrix
    
    // If Patches, fallback (throw)
    if (last_lA.isPatches() || last_uA.isPatches()) {
        throw std::runtime_error("BoundedConstantNode: Patches mode not implemented (requires conversion)");
    }
    
    torch::Tensor last_lA_tensor = last_lA.asTensor();
    torch::Tensor last_uA_tensor = last_uA.asTensor();

    // Determine spec dimension from A matrices
    int spec_size = 0;
    if (last_lA_tensor.defined()) {
        spec_size = (last_lA_tensor.dim() >= 2) ? last_lA_tensor.size(1) : last_lA_tensor.size(0);
    } else if (last_uA_tensor.defined()) {
        spec_size = (last_uA_tensor.dim() >= 2) ? last_uA_tensor.size(1) : last_uA_tensor.size(0);
    }

    // Initialize bias tensors if they're not defined
    if (!lbias.defined() && spec_size > 0) {
        lbias = torch::zeros({spec_size});
    }
    if (!ubias.defined() && spec_size > 0) {
        ubias = torch::zeros({spec_size});
    }

    if (last_lA_tensor.defined()) {
        // Compute bias contribution: A @ constant
        // last_lA has shape [batch, spec, *constant_shape]
        // We need to compute matrix multiplication: A @ constant
        // Flatten both A and constant to 2D for proper matrix multiplication

        torch::Tensor A_flat;
        if (last_lA_tensor.dim() == 3) {
            // Shape: [batch, spec, const_size] -> reshape to [batch*spec, const_size]
            A_flat = last_lA_tensor.reshape({last_lA_tensor.size(0) * last_lA_tensor.size(1), last_lA_tensor.size(2)});
        } else if (last_lA_tensor.dim() == 2) {
            // Already 2D [spec, const_size]
            A_flat = last_lA_tensor;
        } else {
            // Flatten all dimensions except the last
            A_flat = last_lA_tensor.flatten(0, -2);
        }

        torch::Tensor constant_flat = _constantValue.flatten();

        // Matrix-vector multiplication: [batch*spec, const_size] @ [const_size] = [batch*spec]
        torch::Tensor new_lbias = torch::matmul(A_flat, constant_flat);

        // Reshape to [batch, spec] if needed, then squeeze to match bias shape
        if (last_lA_tensor.dim() == 3) {
            new_lbias = new_lbias.reshape({last_lA_tensor.size(0), last_lA_tensor.size(1)}).squeeze(0);
        }

        // Add to existing bias
        lbias = lbias + new_lbias;
    }

    if (last_uA_tensor.defined()) {
        // Compute bias contribution: A @ constant
        torch::Tensor A_flat;
        if (last_uA_tensor.dim() == 3) {
            A_flat = last_uA_tensor.reshape({last_uA_tensor.size(0) * last_uA_tensor.size(1), last_uA_tensor.size(2)});
        } else if (last_uA_tensor.dim() == 2) {
            A_flat = last_uA_tensor;
        } else {
            A_flat = last_uA_tensor.flatten(0, -2);
        }

        torch::Tensor constant_flat = _constantValue.flatten();
        torch::Tensor new_ubias = torch::matmul(A_flat, constant_flat);

        if (last_uA_tensor.dim() == 3) {
            new_ubias = new_ubias.reshape({last_uA_tensor.size(0), last_uA_tensor.size(1)}).squeeze(0);
        }

        ubias = ubias + new_ubias;
    }

    // Zero A matrices for inputs (constants have no inputs)
    outputA_matrices.clear();
}

BoundedTensor<torch::Tensor> NLR::BoundedConstantNode::computeIntervalBoundPropagation(
    const Vector<BoundedTensor<torch::Tensor>>& inputBounds) {
    (void)inputBounds; // Suppress unused parameter warning
    
    // Constants have exact bounds (no uncertainty)
    return BoundedTensor<torch::Tensor>(_constantValue, _constantValue);
}

void NLR::BoundedConstantNode::setInputSize(unsigned size) {
    // Constants have no inputs, so this is a no-op
    (void)size; // Suppress unused parameter warning
}

void NLR::BoundedConstantNode::setOutputSize(unsigned size) {
    // For constants, output size is determined by the constant tensor
    // This method is mainly for interface compatibility
    (void)size; // Suppress unused parameter warning
}