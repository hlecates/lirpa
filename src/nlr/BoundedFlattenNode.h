#ifndef __BOUNDED_FLATTEN_NODE_H__
#define __BOUNDED_FLATTEN_NODE_H__

#include "BoundedTorchNode.h"
#include "../input_parsers/Operations.h"

namespace NLR {

class BoundedFlattenNode : public BoundedTorchNode {
public:
    BoundedFlattenNode(const Operations::FlattenWrapper& flatten_module);

    // Node identification
    NLR::NodeType getNodeType() const override { return NLR::NodeType::FLATTEN; }
    String getNodeName() const override { return _nodeName; }
    unsigned getNodeIndex() const override { return _nodeIndex; }

    // Standard PyTorch forward pass
    torch::Tensor forward(const torch::Tensor& input) override;

    // CROWN Backward Mode
    Pair<torch::Tensor, torch::Tensor> computeCrownBackwardPropagation(const torch::Tensor& lastLowerAlpha,
                                               const torch::Tensor& lastUpperAlpha,
                                               const Vector<BoundedTensor<torch::Tensor>>& inputs);

    // IBP
    BoundedTensor<torch::Tensor> computeIntervalBoundPropagation(
        const Vector<BoundedTensor<torch::Tensor>>& inputs) override;

    // Auto-LiRPA style boundBackward method
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
    unsigned getInputSize() const override { return _input_size; }
    unsigned getOutputSize() const override { return _output_size; }
    bool isPerturbed() const override { return true; }
    String getModuleType() const { return "Flatten"; }

    // Size setters for initialization
    void setInputSize(unsigned size) override;
    void setOutputSize(unsigned size) override;

    // Node state
    void setNodeIndex(unsigned index) override { _nodeIndex = index; }
    void setNodeName(const String& name) override { _nodeName = name; }

    // Shape information for backward propagation
    void setInputShape(const std::vector<int64_t>& shape) { _input_shape = shape; }
    const std::vector<int64_t>& getInputShape() const { return _input_shape; }

private:
    Operations::FlattenWrapper _flatten_module;
    std::vector<int64_t> _input_shape;  // Store original input shape for backward propagation
};

} // namespace NLR

#endif // __BOUNDED_FLATTEN_NODE_H__
