#ifndef __BOUNDED_IDENTITY_NODE_H__
#define __BOUNDED_IDENTITY_NODE_H__

#include "BoundedTorchNode.h"

namespace NLR {

class BoundedIdentityNode : public NLR::BoundedTorchNode {
public:
    BoundedIdentityNode(const torch::nn::Identity& identity_module);
    
    // Node identification
    NLR::NodeType getNodeType() const override { return NLR::NodeType::IDENTITY; }
    String getNodeName() const override { return _nodeName; }
    unsigned getNodeIndex() const override { return _nodeIndex; }
    
    // Node state
    void setNodeIndex(unsigned index) override { _nodeIndex = index; }
    void setNodeName(const String& name) override { _nodeName = name; }
    
    // Standard PyTorch forward pass
    torch::Tensor forward(const torch::Tensor& input) override;
    
    // CROWN Backward Mode
    Pair<torch::Tensor, torch::Tensor> computeCrownBackwardPropagation(const torch::Tensor& lastLowerAlpha, 
                                               const torch::Tensor& lastUpperAlpha,
                                               const Vector<BoundedTensor<torch::Tensor>>& inputs);
    
    // IBP
    BoundedTensor<torch::Tensor> computeIntervalBoundPropagation(
        const Vector<BoundedTensor<torch::Tensor>>& inputs) override;
    
    // Auto-LiRPA style boundBackward method (NEW)
    // Returns: A matrices for inputs, lower bias, upper bias
    void boundBackward(
        const BoundA& last_lA, 
        const BoundA& last_uA,
        const Vector<BoundedTensor<torch::Tensor>>& inputBounds,
        Vector<Pair<BoundA, BoundA>>& outputA_matrices,
        torch::Tensor& lbias,
        torch::Tensor& ubias
    ) override;
    
    // Module information
    unsigned getInputSize() const override;
    unsigned getOutputSize() const override;
    bool isPerturbed() const override { return false; }
    
    // Size setters for initialization
    void setInputSize(unsigned size) override;
    void setOutputSize(unsigned size) override;

private:
    torch::nn::Identity _identity_module;
};

} // namespace NLR

#endif // __BOUNDED_IDENTITY_NODE_H__