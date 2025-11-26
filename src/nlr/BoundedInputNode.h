// Marabou/src/nlr/BoundedInputNode.h
#ifndef __BOUNDED_INPUT_NODE_H__
#define __BOUNDED_INPUT_NODE_H__

#include "BoundedTorchNode.h"

namespace NLR {

class BoundedInputNode : public NLR::BoundedTorchNode {
public:
    BoundedInputNode(unsigned inputIndex, unsigned inputSize, const String& name = "");
    
    // Node identification
    NLR::NodeType getNodeType() const override { return NLR::NodeType::INPUT; }
    String getNodeName() const override { return _nodeName; }
    unsigned getNodeIndex() const override { return _nodeIndex; }
    
    // Forward pass
    torch::Tensor forward(const torch::Tensor& input) override;
    
    // Backward bound propagation (inputs don't propagate further)
    void boundBackward(
        const BoundA& last_lA,
        const BoundA& last_uA,
        const Vector<BoundedTensor<torch::Tensor>>& inputBounds,
        Vector<Pair<BoundA, BoundA>>& outputA_matrices,
        torch::Tensor& lbias,
        torch::Tensor& ubias
    ) override;
    
    // IBP (inputs have fixed bounds)
    BoundedTensor<torch::Tensor> computeIntervalBoundPropagation(
        const Vector<BoundedTensor<torch::Tensor>>& inputBounds) override;
    
    // Node information
    unsigned getInputSize() const override { return _input_size; }
    unsigned getOutputSize() const override { return _output_size; }
    bool isPerturbed() const override { return true; }  // Inputs are perturbed
    
    // Size setters for initialization
    void setInputSize(unsigned size) override;
    void setOutputSize(unsigned size) override;

    // Node state
    void setNodeIndex(unsigned index) override { _nodeIndex = index; }
    void setNodeName(const String& name) override { _nodeName = name; }
    
    // Input-specific methods
    unsigned getInputIndex() const { return _inputIndex; }
    void setInputBounds(const BoundedTensor<torch::Tensor>& bounds);
    BoundedTensor<torch::Tensor> getInputBounds() const;

private:
    unsigned _inputIndex;
    BoundedTensor<torch::Tensor> _inputBounds;
};

} // namespace NLR

#endif // __BOUNDED_INPUT_NODE_H__