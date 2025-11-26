#ifndef __BOUNDED_ADD_NODE_H__
#define __BOUNDED_ADD_NODE_H__

#include "BoundedTorchNode.h"

namespace NLR {

class BoundedAddNode : public BoundedTorchNode {
public:
    BoundedAddNode();

    // Node identification
    NLR::NodeType getNodeType() const override { return NLR::NodeType::ADD; }
    String getNodeName() const override { return _nodeName; }
    unsigned getNodeIndex() const override { return _nodeIndex; }

    // Standard PyTorch forward pass - single input (not typical for Add)
    torch::Tensor forward(const torch::Tensor& input) override;

    // Multi-input forward pass for Add: x + y
    torch::Tensor forward(const std::vector<torch::Tensor>& inputs) override;

    // CROWN Backward Mode bound propagation
    void boundBackward(
        const BoundA& last_lA,
        const BoundA& last_uA,
        const Vector<BoundedTensor<torch::Tensor>>& inputBounds,
        Vector<Pair<BoundA, BoundA>>& outputA_matrices,
        torch::Tensor& lbias,
        torch::Tensor& ubias
    ) override;

    // IBP (Interval Bound Propagation)
    BoundedTensor<torch::Tensor> computeIntervalBoundPropagation(
        const Vector<BoundedTensor<torch::Tensor>>& inputBounds) override;

    // Module information
    unsigned getInputSize() const override { return _input_size; }
    unsigned getOutputSize() const override { return _output_size; }
    bool isPerturbed() const override { return true; }
    String getModuleType() const { return "Add"; }

    // Size setters for initialization
    void setInputSize(unsigned size) override;
    void setOutputSize(unsigned size) override;

    // Node state
    void setNodeIndex(unsigned index) override { _nodeIndex = index; }
    void setNodeName(const String& name) override { _nodeName = name; }

    // Store the constant value for Add operations with constants
    void setConstantValue(const torch::Tensor& constant) { _constantValue = constant; }
    bool hasConstant() const { return _constantValue.defined(); }
    torch::Tensor getConstantValue() const { return _constantValue; }

private:
    torch::Tensor _constantValue;  // Store constant for x + constant operations

    // Helper function for broadcasting in backward pass
    torch::Tensor broadcast_backward(const torch::Tensor& last_A, const BoundedTensor<torch::Tensor>& input) const;
};

} // namespace NLR

#endif // __BOUNDED_ADD_NODE_H__
