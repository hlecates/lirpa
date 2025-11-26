#ifndef __BOUNDED_RELU_NODE_H__
#define __BOUNDED_RELU_NODE_H__

#include "BoundedAlphaOptimizedNode.h"

namespace NLR {

class BoundedReLUNode : public NLR::BoundedAlphaOptimizeNode {
public:
    BoundedReLUNode(const torch::nn::ReLU& reluModule, const String& name = "");
    
    // Node identification
    NLR::NodeType getNodeType() const override { return NLR::NodeType::RELU; }
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
    
    // Module information
    unsigned getInputSize() const override;
    unsigned getOutputSize() const override;
    bool isPerturbed() const override { return true; }  // ReLU is perturbed (depends on input)
    
    // Size setters for initialization
    void setInputSize(unsigned size) override;
    void setOutputSize(unsigned size) override;

    // Node state
    void setNodeIndex(unsigned index) override { _nodeIndex = index; }
    void setNodeName(const String& name) override { _nodeName = name; }
    
    // Relaxation result structure (following auto_LiRPA approach)
    struct RelaxationResult {
        torch::Tensor d_lower;    // Lower bound slopes
        torch::Tensor d_upper;    // Upper bound slopes
        torch::Tensor bias_lower; // Lower bound biases
        torch::Tensor bias_upper; // Upper bound biases

        // Additional alpha-based slopes (for optimization mode)
        torch::Tensor lb_lower_d, ub_lower_d;  // Lower bound slopes for lA/uA
        torch::Tensor lb_upper_d, ub_upper_d;  // Upper bound slopes for lA/uA
    };
    
    // Unified backward relaxation method (following auto_LiRPA approach)
    RelaxationResult _backwardRelaxation(const BoundA& last_lA, const BoundA& last_uA,
                                        const torch::Tensor& input_lower, const torch::Tensor& input_upper);
    
    // Alpha-aware relaxation computation
    void computeAlphaRelaxation(
        const torch::Tensor& last_lA,
        const torch::Tensor& last_uA,
        const torch::Tensor& input_lower,
        const torch::Tensor& input_upper,
        torch::Tensor& d_lower,
        torch::Tensor& d_upper,
        torch::Tensor& bias_lower,
        torch::Tensor& bias_upper) override;

private:
    std::shared_ptr<torch::nn::ReLU> _reluModule; 
    
    // Helper methods following auto_LiRPA approach
    std::pair<torch::Tensor, torch::Tensor> _reluUpperBound(const torch::Tensor& lb, const torch::Tensor& ub);
    torch::Tensor _computeStandardCROWNLowerBound(const torch::Tensor& input_lower, const torch::Tensor& input_upper);
    torch::Tensor getAlphaForBound(bool isLowerBound, int boundType) const;
    void _maskAlpha(const torch::Tensor& input_lower, const torch::Tensor& input_upper, const torch::Tensor& upper_d, RelaxationResult& result);
    
    // Helper to maybe unfold patches
    torch::Tensor maybe_unfold_patches(const torch::Tensor& d_tensor, const BoundA& last_A);

    // Storage for CROWN upper slopes (for alpha initialization)
    torch::Tensor init_upper_d;

    // Current spec dimension from last_lA (used for per-spec alpha)
    int _currentSpecDim{1};

public:
    // CROWN slope access for alpha initialization (following auto_LiRPA approach)
    torch::Tensor getCROWNSlope(bool isLowerBound) const override;
    
};

} // namespace NLR

#endif // __BOUNDED_RELU_NODE_H__
