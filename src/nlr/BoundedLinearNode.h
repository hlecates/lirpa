// Marabou/src/nlr/bounded_modules/TorchLinearModule.h
#ifndef __BOUNDED_LINEAR_NODE_H__
#define __BOUNDED_LINEAR_NODE_H__

#include "BoundedTorchNode.h"

namespace NLR {

class BoundedLinearNode : public NLR::BoundedTorchNode {
public:
    BoundedLinearNode(const torch::nn::Linear& linearModule, 
        float alpha = 1.0f, const String& name = "");
    
    // Node identification
    NLR::NodeType getNodeType() const override { return NLR::NodeType::LINEAR; }
    String getNodeName() const override { return _nodeName; }
    unsigned getNodeIndex() const override { return _nodeIndex; }
    
    // Forward pass
    torch::Tensor forward(const torch::Tensor& input) override;
    
    // Backward bound propagation
    void boundBackward(
        const BoundA& last_lA,
        const BoundA& last_uA,
        const Vector<BoundedTensor<torch::Tensor>>& inputBounds,
        Vector<Pair<BoundA, BoundA>>& outputA_matrices,
        torch::Tensor& lbias,
        torch::Tensor& ubias
    ) override;
    
    // IBP
    BoundedTensor<torch::Tensor> computeIntervalBoundPropagation(
        const Vector<BoundedTensor<torch::Tensor>>& inputBounds) override;
    
    // Node information
    unsigned getInputSize() const override;
    unsigned getOutputSize() const override;
    bool isPerturbed() const override { return true; }
    
    // Size validation methods
    bool hasInputSize() const { return _input_size > 0; }
    bool hasOutputSize() const { return _output_size > 0; }
    
    // Size setters for initialization
    void setInputSize(unsigned size) override;
    void setOutputSize(unsigned size) override;
    
    
    // Node state
    void setNodeIndex(unsigned index) override { _nodeIndex = index; }
    void setNodeName(const String& name) override { _nodeName = name; }
    
    // IBP computation methods
    torch::Tensor computeLinearIBPLowerBound(const torch::Tensor& inputLowerBound, const torch::Tensor& inputUpperBound);
    torch::Tensor computeLinearIBPUpperBound(const torch::Tensor& inputLowerBound, const torch::Tensor& inputUpperBound);
    
    // Access to the linear module
    const torch::nn::Linear& getLinearModule() const { return _linearModule; }

private:
    torch::nn::Linear _linearModule;
    float _alpha;
};

} // namespace NLR

#endif // __BOUNDED_LINEAR_NODE_H__
