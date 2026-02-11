#ifndef __BOUNDED_SIGMOID_NODE_H__
#define __BOUNDED_SIGMOID_NODE_H__

#include "BoundedAlphaOptimizedNode.h"
#include "configuration/LunaConfiguration.h"

namespace NLR {

class BoundedSigmoidNode : public NLR::BoundedAlphaOptimizeNode {
public:
    BoundedSigmoidNode(const torch::nn::Sigmoid& sigmoidModule, const String& name = "");
    
    // Node identification
    NLR::NodeType getNodeType() const override { return NLR::NodeType::SIGMOID; }
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
    bool isPerturbed() const override { return true; }  // Sigmoid is perturbed (depends on input)
    
    // Size setters for initialization
    void setInputSize(unsigned size) override;
    void setOutputSize(unsigned size) override;

    // Node state
    void setNodeIndex(unsigned index) override { _nodeIndex = index; }
    void setNodeName(const String& name) override { _nodeName = name; }
    void moveToDevice(const torch::Device& device) override;
    
    // Relaxation result structure (following auto_LiRPA approach)
    struct RelaxationResult {
        torch::Tensor d_lower;    // Lower bound slopes
        torch::Tensor d_upper;    // Upper bound slopes
        torch::Tensor bias_lower; // Lower bound biases
        torch::Tensor bias_upper; // Upper bound biases

        // Additional alpha-based slopes (for optimization mode)
        torch::Tensor lb_lower_d, ub_lower_d;  // Lower bound slopes for lA/uA
        torch::Tensor lb_upper_d, ub_upper_d;  // Upper bound slopes for lA/uA
        // Optional per-path alpha-based biases (for closer auto_LiRPA matching)
        torch::Tensor lb_lower_b, ub_lower_b;  // Lower-bound bias for lA/uA paths
        torch::Tensor lb_upper_b, ub_upper_b;  // Upper-bound bias for lA/uA paths
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

    // CROWN slope access for alpha initialization (following auto_LiRPA approach)
    torch::Tensor getCROWNSlope(bool isLowerBound) const override;

    // Expose precomputed tangent defaults for alpha initialization.
    std::pair<torch::Tensor, torch::Tensor> getDefaultTangentPoints(
        const torch::Tensor& lower, const torch::Tensor& upper);

private:
    std::shared_ptr<torch::nn::Sigmoid> _sigmoidModule;
    
    // Precomputed lookup tables (following Python BoundTanh implementation)
    torch::Tensor d_lower;      // Precomputed tangent points for lower bounds
    torch::Tensor d_upper;      // Precomputed tangent points for upper bounds
    torch::Tensor dfunc_values;  // Precomputed derivative values
    
    // Configuration for lookup tables
    double step_pre = 0.01;
    int num_points_pre = 0;
    double x_limit = 20.0;  // SIGMOID_CUTOFF_CONSTANT from LunaConfiguration
    bool _lookupTablesInitialized = false;
    
    // Helper methods for lookup table generation and retrieval
    void precomputeRelaxation(double required_limit = -1.0);
    void precomputeDfuncValues();
    torch::Tensor retrieveFromPrecompute(const torch::Tensor& precomputed_d, 
                                         const torch::Tensor& input_bound, 
                                         const torch::Tensor& default_d);
    std::pair<torch::Tensor, torch::Tensor> generateDLowerUpper(const torch::Tensor& lower, 
                                                                 const torch::Tensor& upper);
    std::pair<torch::Tensor, torch::Tensor> retrieveDFromK(const torch::Tensor& k);
    
    // Sigmoid function helpers
    torch::Tensor sigmoidFunc(const torch::Tensor& x);
    torch::Tensor dsigmoidFunc(const torch::Tensor& x);
    
    // Bound relaxation implementation (matching Python bound_relax_impl)
    void boundRelaxImpl(const torch::Tensor& input_lower, const torch::Tensor& input_upper);
    
    // Helper to maybe unfold patches
    torch::Tensor maybe_unfold_patches(const torch::Tensor& d_tensor, const BoundA& last_A);
    
    // Storage for CROWN slopes (for alpha initialization)
    torch::Tensor init_lower_d;
    torch::Tensor init_upper_d;
    
    // Current spec dimension from last_lA (used for per-spec alpha)
    int _currentSpecDim{1};
    
    // Masks for different input bound cases
    torch::Tensor mask_pos;   // input lower >= 0
    torch::Tensor mask_neg;   // input upper <= 0
    torch::Tensor mask_both;  // crosses zero (lower < 0 and upper > 0)
    
    // Linear relaxation coefficients (following Python add_linear_relaxation)
    torch::Tensor lw, lb;  // Lower bound weight and bias
    torch::Tensor uw, ub;  // Upper bound weight and bias
};

} // namespace NLR

#endif // __BOUNDED_SIGMOID_NODE_H__

