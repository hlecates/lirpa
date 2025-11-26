// Marabou/src/nlr/bounded_modules/BoundedLinearNode.cpp
#include "BoundedLinearNode.h"

namespace NLR {

NLR::BoundedLinearNode::BoundedLinearNode(const torch::nn::Linear& linearModule,
    float alpha, const String& name)
    : _linearModule(linearModule),
      _alpha(alpha) {

    _nodeName = name;
    _nodeIndex = 0;
    _input_size = 0;
    _output_size = 0;

    // Try to set sizes from weight matrix during construction
    if (_linearModule && _linearModule->weight.defined()) {
        auto weight = _linearModule->weight;
        setInputSize(weight.size(1));  // Weight matrix columns
        setOutputSize(weight.size(0)); // Weight matrix rows

        // Diagnostic: Verify weight tensor properties for Alpha-CROWN compatibility
        if (!weight.requires_grad()) {
            printf("[WARNING] BoundedLinearNode: Weight tensor does not have requires_grad=True\n");
            printf("[WARNING] This will prevent Alpha-CROWN optimization from working correctly.\n");
        }
        if (!weight.is_contiguous()) {
            printf("[WARNING] BoundedLinearNode: Weight tensor is not contiguous.\n");
            printf("[WARNING] This may cause performance issues or gradient flow problems.\n");
        }
        if (weight.dtype() != torch::kFloat32) {
            printf("[WARNING] BoundedLinearNode: Weight tensor dtype is %s, expected Float32.\n",
                   torch::toString(weight.dtype()).c_str());
        }

        // Similar checks for bias if it exists
        if (_linearModule->bias.defined()) {
            auto bias = _linearModule->bias;
            if (!bias.requires_grad()) {
                printf("[WARNING] BoundedLinearNode: Bias tensor does not have requires_grad=True\n");
            }
            if (!bias.is_contiguous()) {
                printf("[WARNING] BoundedLinearNode: Bias tensor is not contiguous.\n");
            }
            if (bias.dtype() != torch::kFloat32) {
                printf("[WARNING] BoundedLinearNode: Bias tensor dtype is %s, expected Float32.\n",
                       torch::toString(bias.dtype()).c_str());
            }
        }
    }
}

// Forward pass through the linear layer
torch::Tensor BoundedLinearNode::forward(const torch::Tensor& input) {
    // Update input/output sizes dynamically if needed
    if (input.dim() > 0) {
        _input_size = input.numel();
        _output_size = _linearModule->weight.size(0);
    }
    
    // Convert input and weight to float32 for consistency
    // Ensure contiguous memory layout for numerical consistency
    torch::Tensor inputFloat = input.to(torch::kFloat32).contiguous();
    torch::Tensor weight = _linearModule->weight.to(torch::kFloat32).contiguous();

    // Apply linear transformation: y = alpha * (W * x + b)
    // Use explicit float32 operations to match auto_LiRPA
    torch::Tensor weight_t = weight.t().contiguous();
    torch::Tensor matmul_result = torch::matmul(inputFloat, weight_t);
    torch::Tensor alpha_scaled = _alpha * matmul_result;
    
    // Add bias if defined
    torch::Tensor result = alpha_scaled;
    if (_linearModule->bias.defined()) {
        torch::Tensor bias = _linearModule->bias.to(torch::kFloat32);
        result = result + bias;
    }
    
    return result;
}

void BoundedLinearNode::boundBackward(
    const BoundA& last_lA,
    const BoundA& last_uA,
    const Vector<BoundedTensor<torch::Tensor>>& inputBounds,
    Vector<Pair<BoundA, BoundA>>& outputA_matrices,
    torch::Tensor& lbias,
    torch::Tensor& ubias) {

    if (inputBounds.size() < 1) {
        throw std::runtime_error("BoundedLinearNode expects at least one input");
    }

    if (!last_lA.isTensor() || !last_uA.isTensor()) {
        // Fallback or error for Patches
        throw std::runtime_error("BoundedLinearNode: Patches mode propagation not implemented (requires conversion to matrix)");
    }

    torch::Tensor last_lA_tensor = last_lA.asTensor();
    torch::Tensor last_uA_tensor = last_uA.asTensor();

    // Extract weight and bias from the linear module
    auto weight = _linearModule->weight.to(torch::kFloat32);
    auto bias = _linearModule->bias.defined() ? _linearModule->bias.to(torch::kFloat32) : torch::Tensor();

    // Scale weight by alpha
    weight = _alpha * weight;
    
    // For linear layers, A matrices are computed as: A = last_A @ weight
    // where last_A represents the transformation from final output to current layer input
    // and weight represents the transformation from current layer input to current layer output
    
    // Compute A matrices for linear layer
    torch::Tensor lA = torch::matmul(last_lA_tensor, weight);
    torch::Tensor uA = torch::matmul(last_uA_tensor, weight);
    
    outputA_matrices.append(Pair<BoundA, BoundA>(BoundA(lA), BoundA(uA)));
    
    // Compute bias contribution 
    // The key insight: bias terms must be transformed to output space dimensions
    if (bias.defined()) {
        if (last_lA_tensor.defined() && last_lA_tensor.numel() > 0) {
            // Transform bias using A matrix multiplication
            // For our case: last_lA: [1, final_output_size, current_layer_output_size]
            // bias: [current_layer_output_size]
            // We need to reshape bias to match the A matrix dimensions
            // last_lA: [1, final_output_size, current_layer_output_size]
            // bias: [current_layer_output_size] -> [1, current_layer_output_size, 1]
            
            // Reshape bias for matrix multiplication
            torch::Tensor bias_reshaped = bias.unsqueeze(0).unsqueeze(-1); // [1, current_layer_output_size, 1]
            
            // Matrix multiplication: [1, final_output_size, current_layer_output_size] @ [1, current_layer_output_size, 1]
            // This gives us: [1, final_output_size, 1]
            torch::Tensor transformed_lbias = torch::matmul(last_lA_tensor, bias_reshaped).squeeze(-1).squeeze(0); // [final_output_size]
            torch::Tensor transformed_ubias = torch::matmul(last_uA_tensor, bias_reshaped).squeeze(-1).squeeze(0); // [final_output_size]
            
            lbias = transformed_lbias;
            ubias = transformed_ubias;
        } else {
            // If no A matrix, do not accumulate bias (not valid for CROWN backward)
            lbias = torch::Tensor();
            ubias = torch::Tensor();
        }
    }
}

BoundedTensor<torch::Tensor> BoundedLinearNode::computeIntervalBoundPropagation(
    const Vector<BoundedTensor<torch::Tensor>>& inputBounds) {
    
    if (inputBounds.size() < 1) {
        throw std::runtime_error("Linear module requires at least one input");
    }
    
    const auto& inputBoundsPair = inputBounds[0];
    torch::Tensor inputLowerBound = inputBoundsPair.lower().to(torch::kFloat32);
    torch::Tensor inputUpperBound = inputBoundsPair.upper().to(torch::kFloat32);
    
    // Set input size from input bounds if not already set
    if (_input_size == 0 && inputLowerBound.defined()) {
        setInputSize(inputLowerBound.numel());
    }
    
    // Extract weight and bias
    auto weight = _linearModule->weight.to(torch::kFloat32);
    auto bias = _linearModule->bias.defined() ? _linearModule->bias.to(torch::kFloat32) : torch::Tensor();
    
    // Scale weight by alpha
    weight = _alpha * weight;
    
    // Compute IBP bounds: y = alpha * (W * x + b)
    torch::Tensor lowerBound = computeLinearIBPLowerBound(inputLowerBound, inputUpperBound);
    torch::Tensor upperBound = computeLinearIBPUpperBound(inputLowerBound, inputUpperBound);
    
    // Add bias if defined
    if (bias.defined()) {
        torch::Tensor bias_scaled = _alpha * bias;
        lowerBound = lowerBound + bias_scaled;
        upperBound = upperBound + bias_scaled;
    }
    
    // Set output size from computed bounds if not already set
    if (_output_size == 0 && lowerBound.defined()) {
        setOutputSize(lowerBound.numel());
    }
    
    return BoundedTensor<torch::Tensor>(lowerBound, upperBound);
}

// Node information
unsigned BoundedLinearNode::getInputSize() const {
    if (_input_size > 0) {
        return _input_size;
    }
    
    // Fallback: try to infer from weight matrix
    if (_linearModule && _linearModule->weight.defined()) {
        return _linearModule->weight.size(1); // Input size is weight matrix columns
    }
    
    return 0;
}

unsigned BoundedLinearNode::getOutputSize() const {
    if (_output_size > 0) {
        return _output_size;
    }
    
    // Fallback: try to infer from weight matrix
    if (_linearModule && _linearModule->weight.defined()) {
        return _linearModule->weight.size(0); // Output size is weight matrix rows
    }
    
    return 0;
}

void BoundedLinearNode::setInputSize(unsigned size) {
    if (size > 0) {
        _input_size = size;
    }
}

void BoundedLinearNode::setOutputSize(unsigned size) {
    if (size > 0) {
        _output_size = size;
    }
}

// IBP computation methods
torch::Tensor BoundedLinearNode::computeLinearIBPLowerBound(const torch::Tensor& inputLowerBound, const torch::Tensor& inputUpperBound) {
    auto weight = _linearModule->weight.to(torch::kFloat32);
    weight = _alpha * weight;
    
    // Convert input tensors to float32 if needed
    torch::Tensor inputLower = inputLowerBound.to(torch::kFloat32);
    torch::Tensor inputUpper = inputUpperBound.to(torch::kFloat32);
    
    // For linear layers, IBP is straightforward
    // y_lower = W_positive * x_lower + W_negative * x_upper
    torch::Tensor W_positive = torch::clamp(weight, 0);
    torch::Tensor W_negative = torch::clamp(weight, std::numeric_limits<float>::lowest(), 0);
    
    // Fix: Use weight directly, not transpose
    torch::Tensor term1 = torch::matmul(inputLower, weight.t());
    torch::Tensor term2 = torch::matmul(inputUpper, weight.t());
    
    // For lower bound: use positive weights with lower input, negative weights with upper input
    torch::Tensor positive_contribution = torch::matmul(inputLower, W_positive.t());
    torch::Tensor negative_contribution = torch::matmul(inputUpper, W_negative.t());
    
    return positive_contribution + negative_contribution;
}

torch::Tensor BoundedLinearNode::computeLinearIBPUpperBound(const torch::Tensor& inputLowerBound, const torch::Tensor& inputUpperBound) {
    auto weight = _linearModule->weight.to(torch::kFloat32);
    weight = _alpha * weight;
    
    // Convert input tensors to float32 if needed
    torch::Tensor inputLower = inputLowerBound.to(torch::kFloat32);
    torch::Tensor inputUpper = inputUpperBound.to(torch::kFloat32);
    
    // For linear layers, IBP is straightforward
    // y_upper = W_positive * x_upper + W_negative * x_lower
    torch::Tensor W_positive = torch::clamp(weight, 0);
    torch::Tensor W_negative = torch::clamp(weight, std::numeric_limits<float>::lowest(), 0);
    
    // Fix: Use weight directly, not transpose
    torch::Tensor term1 = torch::matmul(inputUpper, weight.t());
    torch::Tensor term2 = torch::matmul(inputLower, weight.t());
    
    // For upper bound: use positive weights with upper input, negative weights with lower input
    torch::Tensor positive_contribution = torch::matmul(inputUpper, W_positive.t());
    torch::Tensor negative_contribution = torch::matmul(inputLower, W_negative.t());
    
    return positive_contribution + negative_contribution;
}

} // namespace NLR
