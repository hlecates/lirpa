// Marabou/src/nlr/BoundedTorchNode.h
#ifndef __BOUNDED_TORCH_NODE_H__
#define __BOUNDED_TORCH_NODE_H__

#include "Map.h"
#include "Vector.h"
#include "Pair.h"
#include "MString.h"
#include "BoundedTensor.h"
#include "BoundResult.h"

// Undefine Warning macro to avoid conflict with PyTorch
#ifdef Warning
#undef Warning
#endif

#include <torch/torch.h>
#include <memory>

// Redefine Warning macro for CVC4 compatibility
#ifndef Warning
#define Warning (! ::CVC4::WarningChannel.isOn()) ? ::CVC4::nullCvc4Stream : ::CVC4::WarningChannel
#endif

namespace NLR {

enum class NodeType { INPUT, CONSTANT, LINEAR, RELU, RESHAPE, IDENTITY, SUB, FLATTEN, ADD, CONV, BATCHNORM, SIGMOID, CONVTRANSPOSE, CONCAT, SLICE };

class BoundedTorchNode : public torch::nn::Module {
public:
    virtual ~BoundedTorchNode() = default;
    
    // Node ID getters
    virtual NodeType getNodeType() const = 0;
    virtual String getNodeName() const = 0;
    virtual unsigned getNodeIndex() const = 0;

    // Standard PyTorch forward pass (evaluation mode without bound computation)
    virtual torch::Tensor forward(const torch::Tensor& input) = 0;
    virtual torch::Tensor forward(const std::vector<torch::Tensor>& inputs) {
        if (inputs.size() == 1) {
            return forward(inputs[0]);
        } else {
            throw std::runtime_error("Multi-input forward not implemented for this module");
        }
    }
    
    virtual BoundedTensor<torch::Tensor> computeIntervalBoundPropagation(
        const Vector<BoundedTensor<torch::Tensor>>& inputBounds) = 0;
    
    virtual void boundBackward(
        const BoundA& last_lA,
        const BoundA& last_uA,
        const Vector<BoundedTensor<torch::Tensor>>& inputBounds,
        Vector<Pair<BoundA, BoundA>>& outputA_matrices,
        torch::Tensor& lbias,
        torch::Tensor& ubias
    ) = 0;

    // Node Info
    virtual unsigned getInputSize() const = 0;
    virtual unsigned getOutputSize() const = 0;
    virtual bool isPerturbed() const = 0;
    
    // Size setters for initialization
    virtual void setInputSize(unsigned size) = 0;
    virtual void setOutputSize(unsigned size) = 0;

    // Node state
    virtual void setNodeIndex(unsigned index) = 0;
    virtual void setNodeName(const String& name) = 0;

    virtual void moveToDevice(const torch::Device& device) {
        _device = device;
        this->to(device);
    }

    // Bound accessors (matching auto_LiRPA's property pattern)
    const torch::Tensor& getLower() const { return _lower; }
    const torch::Tensor& getUpper() const { return _upper; }

    void setLower(const torch::Tensor& value) {
        _lower = value;
        _isLowerBoundCurrent = true;
    }

    void setUpper(const torch::Tensor& value) {
        _upper = value;
        _isUpperBoundCurrent = true;
    }

    void setBounds(const torch::Tensor& lower, const torch::Tensor& upper) {
        _lower = lower;
        _upper = upper;
        _isLowerBoundCurrent = true;
        _isUpperBoundCurrent = true;
    }

    void clearBounds() {
        _lower = torch::Tensor();
        _upper = torch::Tensor();
        _isLowerBoundCurrent = false;
        _isUpperBoundCurrent = false;
    }

    bool isLowerBoundCurrent() const { return _isLowerBoundCurrent; }
    bool isUpperBoundCurrent() const { return _isUpperBoundCurrent; }
    bool hasBounds() const { return _isLowerBoundCurrent && _isUpperBoundCurrent; }

    // Which inputs require bounds for this node (e.g., ReLU needs bounds on input 0)
    const Vector<unsigned>& getRequiresInputBounds() const { return _requiresInputBounds; }

    // Whether this node can use IBP for intermediate bounds (e.g., ReLU can)
    bool isIBPIntermediate() const { return _ibpIntermediate; }

protected:
    unsigned _nodeIndex;
    String _nodeName;
    unsigned _input_size;
    unsigned _output_size;
    torch::Device _device{torch::kCPU};

    // Bound storage (models auto_LiRPA's node.lower/node.upper pattern)
    torch::Tensor _lower;
    torch::Tensor _upper;
    bool _isLowerBoundCurrent{false};
    bool _isUpperBoundCurrent{false};

    // Lazy computation flags (set in derived class constructors)
    Vector<unsigned> _requiresInputBounds;  // Which input indices need bounds
    bool _ibpIntermediate{false};           // Can use IBP for this node's bounds
};

} // namespace NLR

#endif // __TORCH_MODULE_BOUNDED_H__
