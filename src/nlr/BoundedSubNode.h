#ifndef __BOUNDED_SUB_NODE_H__
#define __BOUNDED_SUB_NODE_H__

#include "BoundedTorchNode.h"

namespace NLR {

class BoundedSubNode : public BoundedTorchNode {
public:
    BoundedSubNode();

    // Node identification
    NLR::NodeType getNodeType() const override { return NLR::NodeType::SUB; }
    String getNodeName() const override { return _nodeName; }
    unsigned getNodeIndex() const override { return _nodeIndex; }

    // Standard PyTorch forward pass - single input (not typical for Sub)
    torch::Tensor forward(const torch::Tensor& input) override;

    // Multi-input forward pass for Sub: x - y
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
    String getModuleType() const { return "Sub"; }

    // Size setters for initialization
    void setInputSize(unsigned size) override;
    void setOutputSize(unsigned size) override;

    // Node state
    void setNodeIndex(unsigned index) override { _nodeIndex = index; }
    void setNodeName(const String& name) override { _nodeName = name; }

    // Store the constant value for Sub operations with constants
    void setConstantValue(const torch::Tensor& constant, bool isSecondOperand = true) {
        _constantValue = constant;
        _constantIsSecond = isSecondOperand;
    }
    bool hasConstant() const { return _constantValue.defined(); }
    torch::Tensor getConstantValue() const { return _constantValue; }
    bool isConstantSecondOperand() const { return _constantIsSecond; }

private:
    torch::Tensor _constantValue;  // Store constant for x - constant operations
    bool _constantIsSecond = true;  // Whether constant is the second operand (x - c) vs (c - x)
};

} // namespace NLR

#endif // __BOUNDED_SUB_NODE_H__
