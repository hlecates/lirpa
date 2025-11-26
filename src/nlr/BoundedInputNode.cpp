#include "BoundedInputNode.h"

NLR::BoundedInputNode::BoundedInputNode(unsigned inputIndex, unsigned inputSize, const String& name)
    : _inputIndex(inputIndex) {
    _nodeName = name;
    _nodeIndex = 0;
    _input_size = inputSize;
    _output_size = inputSize;
    
    // Initialize input bounds with default values
    // This is a placeholder for now, these need to be set by torchmodel updating bounds
    torch::Tensor lower = torch::zeros({inputSize}, torch::kFloat32);
    torch::Tensor upper = torch::ones({inputSize}, torch::kFloat32);
    _inputBounds = BoundedTensor<torch::Tensor>(lower, upper);
}

torch::Tensor NLR::BoundedInputNode::forward(const torch::Tensor& input) {
    // Input nodes should return the input tensor they receive
    // This maintains proper tensor dimensions for processing
    if (input.defined() && input.numel() > 0) {
        return input.to(torch::kFloat32);
    }
    
    // If no input provided, return a tensor with the expected shape
    // Use the average of lower and upper bounds as the default value
    torch::Tensor avg = (_inputBounds.lower() + _inputBounds.upper()) / 2.0;
    return avg.unsqueeze(0); // Add batch dimension: [1, input_size]
}

void NLR::BoundedInputNode::boundBackward(
    const BoundA& last_lA,
    const BoundA& last_uA,
    const Vector<BoundedTensor<torch::Tensor>>& inputBounds,
    Vector<Pair<BoundA, BoundA>>& outputA_matrices,
    torch::Tensor& lbias,
    torch::Tensor& ubias
) {
    // Suppress unused parameter warnings
    (void)inputBounds;
    
    // Input nodes should pass through A matrices unchanged
    // They represent the final linear transformation to the input space
    outputA_matrices.clear();
    outputA_matrices.append(Pair<BoundA, BoundA>(last_lA, last_uA));
    
    // Input nodes don't contribute to bias
    // The bias computation will be handled in concretizeBounds using input bounds
    lbias = torch::Tensor();
    ubias = torch::Tensor();
}

BoundedTensor<torch::Tensor> NLR::BoundedInputNode::computeIntervalBoundPropagation(
    const Vector<BoundedTensor<torch::Tensor>>& inputBounds) {
    // Suppress unused parameter warning
    (void)inputBounds; 
    
    // Input nodes have fixed bounds
    return _inputBounds;
}

void NLR::BoundedInputNode::setInputSize(unsigned size) {
    _input_size = size;
}

void NLR::BoundedInputNode::setOutputSize(unsigned size) {
    // For input nodes, output size equals input size
    _output_size = size;
}

void NLR::BoundedInputNode::setInputBounds(const BoundedTensor<torch::Tensor>& bounds) {
    _inputBounds = bounds;
}

BoundedTensor<torch::Tensor> NLR::BoundedInputNode::getInputBounds() const {
    return _inputBounds;
}