// BoundedConvNode.h - Convolution layer with bound propagation
#ifndef __BOUNDED_CONV_NODE_H__
#define __BOUNDED_CONV_NODE_H__

#include "BoundedTorchNode.h"
#include "conv/ConvolutionMode.h"
#include <vector>

namespace NLR {

class BoundedConvNode : public BoundedTorchNode {
public:
    // Constructor for Conv2d
    BoundedConvNode(const torch::nn::Conv2d& convModule,
                     ConvMode mode = ConvMode::MATRIX,
                     const String& name = "");

    // Node identification
    NodeType getNodeType() const override { return NodeType::CONV; }
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

    // IBP computation
    BoundedTensor<torch::Tensor> computeIntervalBoundPropagation(
        const Vector<BoundedTensor<torch::Tensor>>& inputBounds) override;

    // Node information
    unsigned getInputSize() const override;
    unsigned getOutputSize() const override;
    bool isPerturbed() const override { return true; }
    
    // Infer output size based on input size (assumes square input/kernels for now if shape unknown)
    unsigned inferOutputSize(unsigned inputSize) const;

    // Size setters for initialization
    void setInputSize(unsigned size) override;
    void setOutputSize(unsigned size) override;

    // Node state
    void setNodeIndex(unsigned index) override { _nodeIndex = index; }
    void setNodeName(const String& name) override { _nodeName = name; }

    // Convolution specific getters
    std::vector<int> getPadding() const { return padding; }
    std::vector<int> getStride() const { return stride; }
    std::vector<int> getDilation() const { return dilation; }
    int getGroups() const { return groups; }
    bool hasBias() const { return has_bias; }
    ConvMode getMode() const { return mode; }

    // Set optimization flags
    void setReluFollowed(bool followed) { relu_followed = followed; }
    bool isReluFollowed() const { return relu_followed; }

private:
    // Helper methods
    void initializeFromConv2d(const torch::nn::Conv2d& convModule);

    // Backward bound computation helpers
    BoundA boundOneSide(const BoundA& last_A,
                        const torch::Tensor& weight,
                        const torch::Tensor& bias,
                        torch::Tensor& sum_bias);

    // Compute output padding for transpose convolution
    std::vector<int> computeOutputPadding(const std::vector<int>& input_shape,
                                           const std::vector<int>& output_shape,
                                           const torch::Tensor& weight) const;

    // Conv module for 2D convolution
    torch::nn::Conv2d conv2d{nullptr};

    // Convolution parameters (mirrors auto_LiRPA structure)
    std::vector<int> padding;      // Padding for each dimension
    std::vector<int> stride;       // Stride for each dimension
    std::vector<int> dilation;     // Dilation for each dimension
    int groups;                    // Number of groups for grouped convolution
    bool has_bias;                 // Whether bias is present

    // Mode and optimization flags
    ConvMode mode;                 // MATRIX or PATCHES mode
    bool relu_followed;            // Whether this conv is followed by ReLU
    bool patches_start;            // Whether to start patches mode

    // Cached shapes
    std::vector<int> input_shape;   // Shape of input tensor
    std::vector<int> output_shape;  // Shape of output tensor
};

} // namespace NLR

#endif // __BOUNDED_CONV_NODE_H__
