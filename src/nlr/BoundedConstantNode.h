// Marabou/src/nlr/BoundedConstantNode.h
#ifndef __BOUNDED_CONSTANT_NODE_H__
#define __BOUNDED_CONSTANT_NODE_H__

#include "BoundedTorchNode.h"

namespace NLR {

class BoundedConstantNode : public NLR::BoundedTorchNode {
public:
    BoundedConstantNode(const torch::Tensor& constantValue, const String& name = "");
    
    // Node identification
    NLR::NodeType getNodeType() const override { return NLR::NodeType::CONSTANT; }
    String getNodeName() const override { return _nodeName; }
    unsigned getNodeIndex() const override { return _nodeIndex; }
    
    // Forward pass
    torch::Tensor forward(const torch::Tensor& input) override;
    
    // Backward bound propagation (constants add to bias, zero A matrices)
    void boundBackward(
        const BoundA& last_lA,
        const BoundA& last_uA,
        const Vector<BoundedTensor<torch::Tensor>>& inputBounds,
        Vector<Pair<BoundA, BoundA>>& outputA_matrices,
        torch::Tensor& lbias,
        torch::Tensor& ubias
    ) override;
    
    // IBP (constants have exact bounds)
    BoundedTensor<torch::Tensor> computeIntervalBoundPropagation(
        const Vector<BoundedTensor<torch::Tensor>>& inputBounds) override;
    
    // Node information
    unsigned getInputSize() const override { return 0; }  // Constants have no inputs
    unsigned getOutputSize() const override { return _constantValue.numel(); }
    bool isPerturbed() const override { return false; }  // Constants are not perturbed
    
    // Size setters for initialization
    void setInputSize(unsigned size) override;
    void setOutputSize(unsigned size) override;

    // Node state
    void setNodeIndex(unsigned index) override { _nodeIndex = index; }
    void setNodeName(const String& name) override { _nodeName = name; }
    
    // Constant-specific methods
    torch::Tensor getConstantValue() const { return _constantValue; }

private:
    torch::Tensor _constantValue;
};

} // namespace NLR

#endif // __BOUNDED_CONSTANT_NODE_H__