#include "BoundedReshapeNode.h"

namespace NLR {

BoundedReshapeNode::BoundedReshapeNode(const Operations::ReshapeWrapper& reshape_module) 
    : _reshape_module(reshape_module) {
    _nodeName = "reshape";  // Set default name
    _nodeIndex = 0;
    _input_size = 0;  // Will be set dynamically
    _output_size = 0; // Will be set dynamically
}

// Standard PyTorch forward pass
torch::Tensor BoundedReshapeNode::forward(const torch::Tensor& input) {
    // Update input/output sizes dynamically
    if (input.dim() > 0) {
        _input_size = input.numel();
        _output_size = input.numel(); // Reshape preserves total number of elements
    }
    
    // Use the reshape module's forward method
    torch::Tensor output = _reshape_module.forward(input);
    
    return output;
}

// Auto-LiRPA style boundBackward method (NEW)
// Reshape operations don't change the linear relationships, just pass through A matrices
void BoundedReshapeNode::boundBackward(
    const BoundA& last_lA, 
    const BoundA& last_uA,
    const Vector<BoundedTensor<torch::Tensor>>& inputBounds,
    Vector<Pair<BoundA, BoundA>>& outputA_matrices,
    torch::Tensor& lbias,
    torch::Tensor& ubias) {
    
    if (inputBounds.size() < 1) {
        throw std::runtime_error("BoundedReshapeNode expects at least one input");
    }
    
    if (last_lA.isPatches() || last_uA.isPatches()) {
         throw std::runtime_error("BoundedReshapeNode: Patches propagation not implemented (convert to matrix)");
    }
    
    torch::Tensor lA = last_lA.asTensor();
    torch::Tensor uA = last_uA.asTensor();
    
    // Reshape operations don't change the linear relationships
    // Simply pass through the A matrices
    outputA_matrices.clear();
    outputA_matrices.append(Pair<BoundA, BoundA>(BoundA(lA), BoundA(uA)));
    
    // Reshape operations don't add bias - initialize to zeros with correct size
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



// IBP (Interval Bound Propagation): Fast interval-based bound computation for Reshape
BoundedTensor<torch::Tensor> BoundedReshapeNode::computeIntervalBoundPropagation(
    const Vector<BoundedTensor<torch::Tensor>>& inputBounds) {
    
    if (inputBounds.size() < 1) {
        throw std::runtime_error("Reshape module requires at least one input");
    }
    
    const auto& inputBoundsPair = inputBounds[0];
    torch::Tensor inputLowerBound = inputBoundsPair.lower();
    torch::Tensor inputUpperBound = inputBoundsPair.upper();
    
    // Apply reshape to both lower and upper bounds
    torch::Tensor reshapedLower = _reshape_module.forward(inputLowerBound);
    torch::Tensor reshapedUpper = _reshape_module.forward(inputUpperBound);
    
    return BoundedTensor<torch::Tensor>(reshapedLower, reshapedUpper);
}

void NLR::BoundedReshapeNode::setInputSize(unsigned size) {
    _input_size = size;
}

void NLR::BoundedReshapeNode::setOutputSize(unsigned size) {
    _output_size = size;
}

} // namespace NLR 