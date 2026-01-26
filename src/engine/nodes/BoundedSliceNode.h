// BoundedSliceNode.h - Slice operation with bound propagation
#ifndef __BOUNDED_SLICE_NODE_H__
#define __BOUNDED_SLICE_NODE_H__

#include "BoundedTorchNode.h"
#include <vector>

namespace NLR {

/**
 * BoundedSliceNode implements the ONNX Slice operation with support for
 * interval bound propagation (IBP) and CROWN backward bound propagation.
 *
 * Based on auto_LiRPA's BoundSlice implementation (slice_concat.py).
 */
class BoundedSliceNode : public BoundedTorchNode {
public:
    /**
     * Constructor for BoundedSliceNode.
     * @param start Starting index for slicing
     * @param end Ending index for slicing (exclusive)
     * @param axes The axis along which to slice
     * @param steps Step size (only 1 or -1 supported)
     * @param name Optional node name
     */
    BoundedSliceNode(int64_t start, int64_t end, int64_t axes, int64_t steps = 1,
                     const String& name = "");

    // Node identification
    NodeType getNodeType() const override { return NodeType::SLICE; }
    String getNodeName() const override { return _nodeName; }
    unsigned getNodeIndex() const override { return _nodeIndex; }

    // Forward pass - slice the input tensor
    torch::Tensor forward(const torch::Tensor& input) override;
    torch::Tensor forward(const std::vector<torch::Tensor>& inputs) override;

    // Backward bound propagation - expand A matrix with zeros
    void boundBackward(
        const BoundA& last_lA,
        const BoundA& last_uA,
        const Vector<BoundedTensor<torch::Tensor>>& inputBounds,
        Vector<Pair<BoundA, BoundA>>& outputA_matrices,
        torch::Tensor& lbias,
        torch::Tensor& ubias
    ) override;

    // IBP computation - slice bounds
    BoundedTensor<torch::Tensor> computeIntervalBoundPropagation(
        const Vector<BoundedTensor<torch::Tensor>>& inputBounds) override;

    // Node information
    unsigned getInputSize() const override { return _input_size; }
    unsigned getOutputSize() const override { return _output_size; }
    bool isPerturbed() const override { return true; }

    // Size setters
    void setInputSize(unsigned size) override { _input_size = size; }
    void setOutputSize(unsigned size) override { _output_size = size; }

    // Node state
    void setNodeIndex(unsigned index) override { _nodeIndex = index; }
    void setNodeName(const String& name) override { _nodeName = name; }

    // Slice-specific accessors
    int64_t getStart() const { return _start; }
    int64_t getEnd() const { return _end; }
    int64_t getAxes() const { return _axes; }
    int64_t getSteps() const { return _steps; }

    // Store the input shape for backward propagation
    void setInputShape(const std::vector<int64_t>& shape) { _input_shape = shape; }
    const std::vector<int64_t>& getInputShape() const { return _input_shape; }

private:
    /**
     * Fix up slice parameters to handle negative indices and ONNX special values.
     * Based on auto_LiRPA's _fixup_params method.
     *
     * @param shape The input tensor shape
     * @param start Starting index (may be negative)
     * @param end Ending index (may be negative or -inf)
     * @param axes The axis to slice along
     * @param steps Step size (1 or -1)
     * @return Pair of (fixed_start, fixed_end)
     */
    std::pair<int64_t, int64_t> _fixup_params(const std::vector<int64_t>& shape,
                                               int64_t start, int64_t end,
                                               int64_t axes, int64_t steps) const;

    int64_t _start;         // Starting index
    int64_t _end;           // Ending index (exclusive)
    int64_t _axes;          // Axis along which to slice
    int64_t _steps;         // Step size (1 or -1)

    std::vector<int64_t> _input_shape;  // Cached input shape for backward pass

    String _nodeName;
    unsigned _nodeIndex;
    unsigned _input_size;
    unsigned _output_size;
};

} // namespace NLR

#endif // __BOUNDED_SLICE_NODE_H__
