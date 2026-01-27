#include "BoundedSigmoidNode.h"
#include "AlphaCROWNAnalysis.h"
#include "configuration/LirpaConfiguration.h"
#include "conv/Patches.h"
#include "Debug.h"
#include <algorithm>
#include <cmath>

namespace NLR {

BoundedSigmoidNode::BoundedSigmoidNode(const torch::nn::Sigmoid& sigmoidModule, const String& name)
    : BoundedAlphaOptimizeNode()
    , _sigmoidModule(std::make_shared<torch::nn::Sigmoid>(sigmoidModule))
    , x_limit(LirpaConfiguration::SIGMOID_CUTOFF_CONSTANT) {
    _nodeName = name;
    _nodeIndex = 0;
    _input_size = 0;
    _output_size = 0;
    num_points_pre = static_cast<int>(x_limit / step_pre);
    _lookupTablesInitialized = false;
}

// Forward pass through the sigmoid layer
torch::Tensor BoundedSigmoidNode::forward(const torch::Tensor& input) {
    // Update input/output sizes dynamically
    if (input.dim() > 0) {
        _input_size = input.numel();
        _output_size = input.numel();
    }
    
    // Apply sigmoid transformation
    return (*_sigmoidModule)(input);
}

void BoundedSigmoidNode::moveToDevice(const torch::Device& device)
{
    BoundedAlphaOptimizeNode::moveToDevice(device);
    if (_sigmoidModule) {
        _sigmoidModule->ptr()->to(device);
    }
    if (d_lower.defined()) d_lower = d_lower.to(device);
    if (d_upper.defined()) d_upper = d_upper.to(device);
    if (dfunc_values.defined()) dfunc_values = dfunc_values.to(device);
    if (init_lower_d.defined()) init_lower_d = init_lower_d.to(device);
    if (init_upper_d.defined()) init_upper_d = init_upper_d.to(device);
    if (mask_pos.defined()) mask_pos = mask_pos.to(device);
    if (mask_neg.defined()) mask_neg = mask_neg.to(device);
    if (mask_both.defined()) mask_both = mask_both.to(device);
    if (lw.defined()) lw = lw.to(device);
    if (lb.defined()) lb = lb.to(device);
    if (uw.defined()) uw = uw.to(device);
    if (ub.defined()) ub = ub.to(device);
}

// Sigmoid function
torch::Tensor BoundedSigmoidNode::sigmoidFunc(const torch::Tensor& x) {
    return torch::sigmoid(x);
}

// Derivative of sigmoid: dsigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
torch::Tensor BoundedSigmoidNode::dsigmoidFunc(const torch::Tensor& x) {
    auto s = torch::sigmoid(x);
    return s * (1 - s);
}

// Precompute relaxation lookup tables (matching Python precompute_relaxation)
void BoundedSigmoidNode::precomputeRelaxation() {
    if (_lookupTablesInitialized && d_lower.defined()) {
        return;
    }
    
    x_limit = LirpaConfiguration::SIGMOID_CUTOFF_CONSTANT;
    step_pre = 0.01;
    num_points_pre = static_cast<int>(x_limit / step_pre);
    int max_iter = 100;
    
    // Use configured device for precomputation
    torch::Device device = _device;
    
    // Helper function to check if slope at d is a lower bound at upper
    auto check_lower = [this](const torch::Tensor& upper, const torch::Tensor& d) -> torch::Tensor {
        torch::Tensor k = dsigmoidFunc(d);
        torch::Tensor y_d = sigmoidFunc(d);
        torch::Tensor y_upper = sigmoidFunc(upper);
        // Return True if the slope is a lower bound: k * (upper - d) + func(d) <= func(upper)
        return (k * (upper - d) + y_d) <= y_upper;
    };
    
    // Helper function to check if slope at d is an upper bound at lower
    auto check_upper = [this](const torch::Tensor& lower, const torch::Tensor& d) -> torch::Tensor {
        torch::Tensor k = dsigmoidFunc(d);
        torch::Tensor y_d = sigmoidFunc(d);
        torch::Tensor y_lower = sigmoidFunc(lower);
        // Return True if the slope is an upper bound: k * (lower - d) + func(d) >= func(lower)
        return (k * (lower - d) + y_d) >= y_lower;
    };
    
    // Given an upper bound point (>=0), find a line that is guaranteed to be a lower bound
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    torch::Tensor upper = step_pre * torch::arange(0, num_points_pre + 5, options);
    torch::Tensor r = torch::zeros_like(upper);
    // Initial guess, the tangent line is at -1
    torch::Tensor l = -torch::ones_like(upper);
    
    while (true) {
        torch::Tensor checked = check_lower(upper, l).to(torch::kInt64);
        // If the initial guess is not small enough, double it (-2, -4, etc)
        l = checked * l + (1 - checked) * (l * 2);
        if (checked.sum().item<int64_t>() == l.numel()) {
            break;
        }
    }
    
    // Binary search to tighten the bound
    for (int i = 0; i < max_iter; ++i) {
        torch::Tensor m = (l + r) / 2;
        torch::Tensor checked = check_lower(upper, m).to(torch::kInt64);
        l = checked * m + (1 - checked) * l;
        r = checked * r + (1 - checked) * m;
    }
    
    // At upper, a line with slope l is guaranteed to lower bound the function
    d_lower = l.clone();
    
    // Do the same for upper bounds
    // Given a lower bound point (<=0), find a line that is guaranteed to be an upper bound
    torch::Tensor lower = -step_pre * torch::arange(0, num_points_pre + 5, options);
    l = torch::zeros_like(upper);
    r = torch::ones_like(upper);
    
    while (true) {
        torch::Tensor checked = check_upper(lower, r).to(torch::kInt64);
        r = checked * r + (1 - checked) * (r * 2);
        if (checked.sum().item<int64_t>() == l.numel()) {
            break;
        }
    }
    
    for (int i = 0; i < max_iter; ++i) {
        torch::Tensor m = (l + r) / 2;
        torch::Tensor checked = check_upper(lower, m).to(torch::kInt64);
        l = (1 - checked) * m + checked * l;
        r = (1 - checked) * r + checked * m;
    }
    
    d_upper = r.clone();
    _lookupTablesInitialized = true;
}

// Precompute derivative function values
void BoundedSigmoidNode::precomputeDfuncValues() {
    if (!_lookupTablesInitialized) {
        precomputeRelaxation();
    }
    
    torch::Tensor upper = step_pre * torch::arange(0, num_points_pre + 5, torch::TensorOptions().dtype(torch::kFloat32));
    dfunc_values = dsigmoidFunc(upper);
}

// Retrieve precomputed values based on input bounds (matching Python retrieve_from_precompute)
torch::Tensor BoundedSigmoidNode::retrieveFromPrecompute(const torch::Tensor& precomputed_d, 
                                                         const torch::Tensor& input_bound, 
                                                         const torch::Tensor& default_d) {
    if (!_lookupTablesInitialized || !precomputed_d.defined()) {
        precomputeRelaxation();
    }
    
    // Ensure precomputed_d is on the same device and dtype as input_bound
    torch::Tensor precomputed_d_aligned = precomputed_d.to(input_bound.device()).to(input_bound.dtype());
    torch::Tensor default_d_aligned = default_d.to(input_bound.device()).to(input_bound.dtype());
    
    // Python: index = torch.max(torch.zeros(...), (input_bound / self.step_pre).to(torch.long).reshape(-1)) + 1
    // Divide input bound into number of steps to the inflection point
    torch::Tensor index = torch::max(
        torch::zeros({input_bound.numel()}, torch::TensorOptions().dtype(torch::kInt64).device(input_bound.device())),
        (input_bound / step_pre).to(torch::kInt64).reshape(-1)
    ) + 1;
    
    // Python: if index.max() >= precomputed_d.numel(): use default, else use index_select
    // If precompute range is smaller than input, tangent points will be taken from default
    if (index.max().item<int64_t>() >= precomputed_d_aligned.numel()) {
        // Use default value for out-of-range indices
        // Python: mask = (index < precomputed_d.numel()).view(input_bound.shape)
        torch::Tensor mask = (index < precomputed_d_aligned.numel()).view(input_bound.sizes());
        // Python: clamped_index = index.clamp(max=precomputed_d.numel() - 1)
        torch::Tensor clamped_index = torch::clamp(index, 0, static_cast<int64_t>(precomputed_d_aligned.numel() - 1));
        // Python: selected = torch.index_select(precomputed_d, 0, clamped_index).view(input_bound.shape)
        torch::Tensor selected = torch::index_select(precomputed_d_aligned, 0, clamped_index).view(input_bound.sizes());
        // Python: return torch.where(mask, selected, default_d).view(input_bound.shape)
        return torch::where(mask, selected, default_d_aligned).view(input_bound.sizes());
    } else {
        // Python: return torch.index_select(precomputed_d, 0, index).view(input_bound.shape)
        return torch::index_select(precomputed_d_aligned, 0, index).view(input_bound.sizes());
    }
}

// Generate valid lower/upper bound slopes using lookup tables
std::pair<torch::Tensor, torch::Tensor> BoundedSigmoidNode::generateDLowerUpper(const torch::Tensor& lower, 
                                                                               const torch::Tensor& upper) {
    if (!_lookupTablesInitialized || !d_lower.defined()) {
        precomputeRelaxation();
    }
    
    // Ensure lookup tables are on the same device and dtype as input tensors
    if (!d_lower.defined() || d_lower.device() != lower.device() || d_lower.dtype() != lower.dtype()) {
        if (!d_lower.defined()) {
            precomputeRelaxation();
        }
        d_lower = d_lower.to(lower.device()).to(lower.dtype());
        d_upper = d_upper.to(lower.device()).to(lower.dtype());
    }
    
    // Indices of neurons with input upper bound >=0, whose optimal slope to
    // lower bound the function was pre-computed
    torch::Tensor d_lower_result = retrieveFromPrecompute(d_lower, upper, lower);
    
    // Indices of neurons with lower bound <=0, whose optimal slope to upper
    // bound the function was pre-computed
    torch::Tensor d_upper_result = retrieveFromPrecompute(d_upper, -lower, upper);
    
    return std::make_pair(d_lower_result, d_upper_result);
}

// Retrieve tangent point from slope (for same-slope relaxation)
std::pair<torch::Tensor, torch::Tensor> BoundedSigmoidNode::retrieveDFromK(const torch::Tensor& k) {
    if (!_lookupTablesInitialized || !dfunc_values.defined()) {
        precomputeDfuncValues();
    }
    
    // Ensure dfunc_values is on the same device and dtype as k
    if (!dfunc_values.defined() || dfunc_values.device() != k.device() || dfunc_values.dtype() != k.dtype()) {
        if (!dfunc_values.defined()) {
            precomputeDfuncValues();
        }
        dfunc_values = dfunc_values.to(k.device()).to(k.dtype());
    }
    
    // Search for the index where dfunc_values matches k
    // Python: d_indices = torch.searchsorted(torch.flip(self.dfunc_values, [0]), k, right=False)
    torch::Tensor dfunc_flipped = torch::flip(dfunc_values, {0});
    torch::Tensor d_indices = torch::searchsorted(dfunc_flipped, k, /*right=*/false);
    d_indices = num_points_pre - d_indices + 4;
    
    torch::Tensor d_left = d_indices * step_pre;
    torch::Tensor d_right = d_left + step_pre;
    torch::Tensor y_left = sigmoidFunc(d_left);
    torch::Tensor y_right = sigmoidFunc(d_right);
    
    torch::Tensor clamped_indices = torch::clamp(d_indices, 0, static_cast<int64_t>(dfunc_values.size(0) - 1));
    torch::Tensor k_left = dfunc_values.index_select(0, clamped_indices);
    torch::Tensor k_right_indices = torch::clamp(d_indices + 1, 0, static_cast<int64_t>(dfunc_values.size(0) - 1));
    torch::Tensor k_right = dfunc_values.index_select(0, k_right_indices);
    
    // Choose the intersection of two tangent lines
    torch::Tensor denominator = (k_left - k_right).clamp_min(1e-8);
    torch::Tensor d_return = (k_left * d_left - k_right * d_right - y_left + y_right) / denominator;
    
    torch::Tensor mask_almost_the_same = (k_left - k_right).abs() < 1e-5;
    d_return = torch::where(mask_almost_the_same, d_left, d_return);
    torch::Tensor y_d = k_left * (d_return - d_left) + y_left;
    
    return std::make_pair(d_return, y_d);
}

// IBP (Interval Bound Propagation): Fast interval-based bound computation for sigmoid
BoundedTensor<torch::Tensor> BoundedSigmoidNode::computeIntervalBoundPropagation(
    const Vector<BoundedTensor<torch::Tensor>>& inputBounds) {
    
    if (inputBounds.size() < 1) {
        throw std::runtime_error("Sigmoid module requires at least one input");
    }
    
    const auto& inputBoundsPair = inputBounds[0];
    torch::Tensor inputLowerBound = inputBoundsPair.lower();
    torch::Tensor inputUpperBound = inputBoundsPair.upper();
    
    // Set input size from the input tensor during IBP
    if (_input_size == 0 && inputLowerBound.defined()) {
        _input_size = inputLowerBound.numel();
    }
    
    // Sigmoid: y = sigmoid(x), which is monotonic
    // Lower bound: sigmoid(lower input)
    // Upper bound: sigmoid(upper input)
    torch::Tensor lowerBound = sigmoidFunc(inputLowerBound);
    torch::Tensor upperBound = sigmoidFunc(inputUpperBound);
    
    // Set output size from the computed bounds during IBP
    if (_output_size == 0 && lowerBound.defined()) {
        _output_size = lowerBound.numel();
    }

    return BoundedTensor<torch::Tensor>(lowerBound, upperBound);
}

// Node information
unsigned BoundedSigmoidNode::getInputSize() const {
    return _input_size;
}

unsigned BoundedSigmoidNode::getOutputSize() const {
    // If output size is not set, try to infer from input size
    if (_output_size == 0 && _input_size > 0) {
        return _input_size; // Sigmoid preserves input size
    }
    // If still 0, return a reasonable default for testing
    if (_output_size == 0) {
        return 2; // Default size for testing
    }
    return _output_size;
}

void BoundedSigmoidNode::setInputSize(unsigned size) {
    _input_size = size;
}

void BoundedSigmoidNode::setOutputSize(unsigned size) {
    _output_size = size;
}

// Bound relaxation implementation (matching Python bound_relax_impl)
void BoundedSigmoidNode::boundRelaxImpl(const torch::Tensor& input_lower, const torch::Tensor& input_upper) {
    // Initialize masks
    mask_pos = input_lower >= 0;
    mask_neg = input_upper <= 0;
    mask_both = torch::logical_not(torch::logical_or(mask_pos, mask_neg));
    
    // Initialize linear relaxation coefficients
    lw = torch::zeros_like(input_lower);
    lb = torch::zeros_like(input_lower);
    uw = torch::zeros_like(input_lower);
    ub = torch::zeros_like(input_lower);
    
    torch::Tensor y_l = sigmoidFunc(input_lower);
    torch::Tensor y_u = sigmoidFunc(input_upper);
    
    
    // k_direct is the slope of the line directly connecting (lower, func(lower)), (upper, func(upper))
    torch::Tensor k_direct = (y_u - y_l) / (input_upper - input_lower).clamp_min(1e-8);
    torch::Tensor mask_almost_the_same = (input_upper - input_lower).abs() < 1e-4;
    k_direct = torch::where(mask_almost_the_same, dsigmoidFunc(input_lower), k_direct);
    
    
    // Upper bound for the case of input lower bound <= 0, is always the direct line
    uw = torch::where(mask_neg, k_direct, uw);
    ub = torch::where(mask_neg, y_l - k_direct * input_lower, ub);
    
    // Lower bound for the case of input upper bound >= 0, is always the direct line
    lw = torch::where(mask_pos, k_direct, lw);
    lb = torch::where(mask_pos, y_l - k_direct * input_lower, lb);
    
    // Generate precomputed tangent points
    auto [d_lower_precomputed, d_upper_precomputed] = generateDLowerUpper(input_lower, input_upper);
    
    
    // Check if direct line can be used for mask_both cases
    torch::Tensor k_lower = dsigmoidFunc(input_lower);
    torch::Tensor k_upper = dsigmoidFunc(input_upper);
    torch::Tensor mask_direct_lower = torch::logical_and(mask_both, k_direct < k_lower);
    torch::Tensor mask_direct_upper = torch::logical_and(mask_both, k_direct < k_upper);
    
    
    // Handle mask_both cases with direct line when valid
    lw = torch::where(mask_direct_lower, k_direct, lw);
    lb = torch::where(mask_direct_lower, y_l - k_direct * input_lower, lb);
    
    uw = torch::where(mask_direct_upper, k_direct, uw);
    ub = torch::where(mask_direct_upper, y_l - k_direct * input_lower, ub);
    
    // Handle mask_both cases with precomputed tangent lines
    torch::Tensor mask_both_lower = torch::logical_and(mask_both, torch::logical_not(mask_direct_lower));
    torch::Tensor mask_both_upper = torch::logical_and(mask_both, torch::logical_not(mask_direct_upper));
    
    torch::Tensor k_lower_tangent = dsigmoidFunc(d_lower_precomputed);
    torch::Tensor y_lower_tangent = sigmoidFunc(d_lower_precomputed);
    lw = torch::where(mask_both_lower, k_lower_tangent, lw);
    lb = torch::where(mask_both_lower, y_lower_tangent - k_lower_tangent * d_lower_precomputed, lb);
    
    torch::Tensor k_upper_tangent = dsigmoidFunc(d_upper_precomputed);
    torch::Tensor y_upper_tangent = sigmoidFunc(d_upper_precomputed);
    uw = torch::where(mask_both_upper, k_upper_tangent, uw);
    ub = torch::where(mask_both_upper, y_upper_tangent - k_upper_tangent * d_upper_precomputed, ub);
    
    // Handle mask_neg lower bound (middle point slope) - Python line 385
    torch::Tensor m = (input_lower + input_upper) / 2;
    torch::Tensor y_m = sigmoidFunc(m);
    torch::Tensor k_m = dsigmoidFunc(m);
    lw = torch::where(mask_neg, k_m, lw);
    lb = torch::where(mask_neg, y_m - k_m * m, lb);
    
    // Handle mask_pos upper bound (middle point slope) - Python line 388
    uw = torch::where(mask_pos, k_m, uw);
    ub = torch::where(mask_pos, y_m - k_m * m, ub);
    
}

// Unified backward relaxation method (following auto_LiRPA approach)
BoundedSigmoidNode::RelaxationResult BoundedSigmoidNode::_backwardRelaxation(
    const BoundA& /* last_lA */, const BoundA& /* last_uA */,
    const torch::Tensor& input_lower, const torch::Tensor& input_upper)
{
    RelaxationResult result;
    
    // Compute bound relaxation
    boundRelaxImpl(input_lower, input_upper);
    
    // Extract slopes and biases from relaxation
    result.d_lower = lw;
    result.d_upper = uw;
    result.bias_lower = lb;
    result.bias_upper = ub;
    
    // Store slopes for alpha initialization if needed
    init_lower_d = lw.detach().clone();
    init_upper_d = uw.detach().clone();
    
    return result;
}

// Helper to maybe unfold patches
torch::Tensor BoundedSigmoidNode::maybe_unfold_patches(const torch::Tensor& d_tensor, const BoundA& last_A) {
    if (!d_tensor.defined() || !last_A.isPatches()) {
        return d_tensor;
    }
    auto patches = last_A.asPatches();
    
    // d_tensor shape: [N, C, H, W]
    // Needs to unfold to match patches kernel [C, kh, kw] at each location
    
    if (d_tensor.dim() == 3) {
        // [C, H, W] -> [1, C, H, W]
        return maybe_unfold_patches(d_tensor.unsqueeze(0), last_A);
    }
    
    // Use inplace_unfold
    torch::Tensor d_unfolded = inplace_unfold(d_tensor, 
        {patches->patches.size(-2), patches->patches.size(-1)}, 
        patches->stride, patches->padding, patches->inserted_zeros, patches->output_padding);
    
    // Permute to match patches: [out_c, batch, out_h, out_w, C, kh, kw]
    return d_unfolded.permute({0, 1, 2, 3, 4, 5}).unsqueeze(0);
}

// Auto-LiRPA style boundBackward method
void BoundedSigmoidNode::boundBackward(
    const BoundA& last_lA,
    const BoundA& last_uA,
    const Vector<BoundedTensor<torch::Tensor>>& inputBounds,
    Vector<Pair<BoundA, BoundA>>& outputA_matrices,
    torch::Tensor& lbias,
    torch::Tensor& ubias) {

    if (inputBounds.size() < 1) {
        throw std::runtime_error("BoundedSigmoidNode expects at least one input");
    }

    const auto& inputBound = inputBounds[0];
    torch::Tensor input_lower = inputBound.lower();
    torch::Tensor input_upper = inputBound.upper();

    // Extract live spec dimension from last_lA or last_uA for per-spec alpha
    int specDim = 1;
    auto inferSpecDim = [](const BoundA& A) -> int {
        if (!A.defined() || !A.isTensor()) return 1;
        torch::Tensor t = A.asTensor();
        if (!t.defined()) return 1;
        // FIXED: A matrix conventions - spec dim is ALWAYS the first dimension (size(0))
        // - 3D A: [S, B, ...] => spec dim is size(0)
        // - 2D A: [S, ...]    => spec dim is size(0)
        if (t.dim() >= 2) return (int)t.size(0);
        return 1;
    };
    if (last_lA.defined() && last_lA.isTensor()) {
        specDim = inferSpecDim(last_lA);
    } else if (last_uA.defined() && last_uA.isTensor()) {
        specDim = inferSpecDim(last_uA);
    }
    _currentSpecDim = specDim;

    // Call the unified backward relaxation method
    auto relaxation_result = _backwardRelaxation(last_lA, last_uA, input_lower, input_upper);

    // Helper lambdas
    auto expand_like = [](torch::Tensor v, const torch::Tensor& A) {
        if (!v.defined() || !A.defined()) return v;

        // Handle the common "flat-to-conv" mismatch by reshaping v when its numel matches
        if (v.numel() > 0 && A.dim() >= 3) {
            int64_t payload_numel = 1;
            for (int d = 2; d < A.dim(); ++d) payload_numel *= A.size(d);
            if (v.numel() == payload_numel && v.dim() == 1) {
                std::vector<int64_t> payload_shape;
                for (int d = 2; d < A.dim(); ++d) payload_shape.push_back(A.size(d));
                v = v.view(payload_shape).unsqueeze(0).unsqueeze(0);
            } else if (v.numel() == payload_numel && v.dim() == 2 && v.size(0) == 1) {
                std::vector<int64_t> payload_shape;
                for (int d = 2; d < A.dim(); ++d) payload_shape.push_back(A.size(d));
                v = v.view(payload_shape).unsqueeze(0).unsqueeze(0);
            }
        }

        // Align rank
        if (A.dim() >= 2 && v.dim() == A.dim() - 2) {
            v = v.unsqueeze(0).unsqueeze(0);
        } else if (A.dim() >= 1 && v.dim() == A.dim() - 1) {
            v = v.unsqueeze(0);
        }

        try {
            return v.expand_as(A);
        } catch (...) {
            return v;
        }
    };

    auto reduce_bias_like_A = [&](const torch::Tensor& term, const torch::Tensor& A) {
        if (!term.defined() || !A.defined()) return term;
        if (A.dim() >= 3) {
            std::vector<int64_t> dims;
            for (int64_t d = 2; d < term.dim(); ++d) dims.push_back(d);
            return dims.empty() ? term : term.sum(dims);
        }
        if (A.dim() == 2) {
            return (term.dim() >= 2) ? term.sum({1}) : term;
        }
        return term;
    };

    BoundA new_lA, new_uA;

    // ----- LOWER path -----
    if (last_lA.defined()) {
        auto d_l = relaxation_result.d_lower;
        auto b_l = relaxation_result.bias_lower;

        if (last_lA.isTensor()) {
            torch::Tensor lA = last_lA.asTensor();
            auto A_l = lA * expand_like(d_l, lA);
            new_lA = BoundA(A_l);
            auto add_lbias = reduce_bias_like_A(lA * expand_like(b_l, lA), lA);
            lbias = lbias.defined() ? (lbias + add_lbias) : add_lbias;
        } else {
            // Patches mode
            auto patches = last_lA.asPatches();
            torch::Tensor d_l_unfolded = maybe_unfold_patches(d_l, last_lA);
            torch::Tensor P = patches->patches;
            torch::Tensor P_new = P * d_l_unfolded;
            new_lA = BoundA(patches->create_similar(P_new));
            
            // Bias handling for patches
            if (b_l.dim() == 3) b_l = b_l.unsqueeze(0);
            torch::Tensor b_l_unfolded = inplace_unfold(b_l, 
                {patches->patches.size(-2), patches->patches.size(-1)}, 
                patches->stride, patches->padding, patches->inserted_zeros, patches->output_padding);
            torch::Tensor b_l_ready = b_l_unfolded.permute({0, 1, 2, 3, 4, 5}).unsqueeze(0);
            torch::Tensor total_bias = (P * b_l_ready).sum({-3, -2, -1});
            total_bias = total_bias.permute({1, 0, 2, 3});
            lbias = lbias.defined() ? (lbias + total_bias) : total_bias;
        }
    }

    // ----- UPPER path -----
    if (last_uA.defined()) {
        auto d_u = relaxation_result.d_upper;
        auto b_u = relaxation_result.bias_upper;

        if (last_uA.isTensor()) {
            torch::Tensor uA = last_uA.asTensor();
            auto A_u = uA * expand_like(d_u, uA);
            new_uA = BoundA(A_u);
            auto add_ubias = reduce_bias_like_A(uA * expand_like(b_u, uA), uA);
            ubias = ubias.defined() ? (ubias + add_ubias) : add_ubias;
        } else {
            // Patches mode
            auto patches = last_uA.asPatches();
            torch::Tensor d_u_unfolded = maybe_unfold_patches(d_u, last_uA);
            torch::Tensor P = patches->patches;
            torch::Tensor P_new = P * d_u_unfolded;
            new_uA = BoundA(patches->create_similar(P_new));
            
            // Bias handling for patches
            if (b_u.dim() == 3) b_u = b_u.unsqueeze(0);
            torch::Tensor b_u_unfolded = inplace_unfold(b_u, 
                {patches->patches.size(-2), patches->patches.size(-1)}, 
                patches->stride, patches->padding, patches->inserted_zeros, patches->output_padding);
            torch::Tensor b_u_ready = b_u_unfolded.permute({0, 1, 2, 3, 4, 5}).unsqueeze(0);
            torch::Tensor total_bias = (P * b_u_ready).sum({-3, -2, -1});
            total_bias = total_bias.permute({1, 0, 2, 3});
            ubias = ubias.defined() ? (ubias + total_bias) : total_bias;
        }
    }

    // Ensure outputA_matrices has the right structure
    if (outputA_matrices.size() == 0) {
        outputA_matrices.append(Pair<BoundA, BoundA>(new_lA, new_uA));
    }
}

// Alpha-aware relaxation computation (implementing BoundedAlphaOptimizeNode interface)
void BoundedSigmoidNode::computeAlphaRelaxation(
    const torch::Tensor& last_lA,
    const torch::Tensor& last_uA,
    const torch::Tensor& input_lower,
    const torch::Tensor& input_upper,
    torch::Tensor& d_lower,
    torch::Tensor& d_upper,
    torch::Tensor& bias_lower,
    torch::Tensor& bias_upper) {
    
    // Use the unified backward relaxation method
    auto result = _backwardRelaxation(BoundA(last_lA), BoundA(last_uA), input_lower, input_upper);
    
    // Extract the computed slopes and biases
    d_lower = result.d_lower;
    d_upper = result.d_upper;
    bias_lower = result.bias_lower;
    bias_upper = result.bias_upper;
}

// Get CROWN slopes for alpha initialization 
torch::Tensor BoundedSigmoidNode::getCROWNSlope(bool isLowerBound) const
{
    if (isLowerBound) {
        if (!init_lower_d.defined() || init_lower_d.numel() == 0) {
            // Return default lower slopes if CROWN slopes not available
            return torch::full({getOutputSize()}, 0.25f, torch::kFloat32);
        }
        return init_lower_d;
    } else {
        if (!init_upper_d.defined() || init_upper_d.numel() == 0) {
            // Return default upper slopes if CROWN slopes not available
            return torch::full({getOutputSize()}, 0.25f, torch::kFloat32);
        }
        return init_upper_d;
    }
}

} // namespace NLR

