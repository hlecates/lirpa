#include "BoundedConvNode.h"
#include "conv/MatrixConvolution.h"
#include "conv/Patches.h"
#include <torch/nn/functional.h>
#include <stdexcept>
#include <cmath>

namespace NLR {

// Constructor for Conv1d
BoundedConvNode::BoundedConvNode(const torch::nn::Conv1d& convModule,
                                 ConvMode mode,
                                 const String& name)
    : conv1d(convModule), mode(mode), conv_dim(1) {

    _nodeName = name;
    _nodeIndex = 0;
    _input_size = 0;
    _output_size = 0;
    relu_followed = false;
    patches_start = true;

    initializeFromConv1d(convModule);
}

// Constructor for Conv2d
BoundedConvNode::BoundedConvNode(const torch::nn::Conv2d& convModule,
                                 ConvMode mode,
                                 const String& name)
    : conv2d(convModule), mode(mode), conv_dim(2) {

    _nodeName = name;
    _nodeIndex = 0;
    _input_size = 0;
    _output_size = 0;
    relu_followed = false;
    patches_start = true;

    initializeFromConv2d(convModule);
}

void BoundedConvNode::initializeFromConv1d(const torch::nn::Conv1d& convModule) {
    if (!convModule) {
        throw std::runtime_error("Conv1d module is null");
    }

    // Extract convolution parameters
    auto options = convModule->options;

    // Set padding (1D has single value)
    if (std::holds_alternative<torch::ExpandingArray<1>>(options.padding())) {
        auto pad_array = std::get<torch::ExpandingArray<1>>(options.padding());
        padding = {static_cast<int>((*pad_array)[0])};
    } else {
        // Handle other padding types (kValid, kSame) for now default to 0
        padding = {0};
    }

    // Set stride
    stride = {static_cast<int>((*options.stride())[0])};

    // Set dilation
    dilation = {static_cast<int>((*options.dilation())[0])};

    // Set groups
    groups = options.groups();

    // Check for bias
    has_bias = options.bias();

    // Try to set sizes from weight matrix during construction
    if (conv1d && conv1d->weight.defined()) {
        auto weight = conv1d->weight;
        // Weight shape: [out_channels, in_channels/groups, kernel_length]

        // Fix weight tensor properties for Alpha-CROWN compatibility
        bool weightNeedsFix = false;
        if (!weight.requires_grad() || !weight.is_contiguous() || weight.dtype() != torch::kFloat32) {
            weightNeedsFix = true;
            if (!weight.requires_grad()) {
            }
            if (!weight.is_contiguous()) {
            }
            if (weight.dtype() != torch::kFloat32) {
            }
        }
        
        // Automatically fix weight tensor
            if (weightNeedsFix) {
                conv1d->weight = weight.contiguous().to(torch::kFloat32).requires_grad_(false);  // Network weights are constants;
            }
        
        // Similar checks and fixes for bias if it exists
        if (has_bias && conv1d->bias.defined()) {
            auto bias = conv1d->bias;
            bool biasNeedsFix = false;
            if (!bias.requires_grad() || !bias.is_contiguous() || bias.dtype() != torch::kFloat32) {
                biasNeedsFix = true;
                if (!bias.requires_grad()) {
                }
                if (!bias.is_contiguous()) {
                }
                if (bias.dtype() != torch::kFloat32) {
                }
            }
            
            // Automatically fix bias tensor
            if (biasNeedsFix) {
                conv1d->bias = bias.contiguous().to(torch::kFloat32).requires_grad_(false);  // Network biases are constants;
            }
        }
    }
}

void BoundedConvNode::initializeFromConv2d(const torch::nn::Conv2d& convModule) {
    if (!convModule) {
        throw std::runtime_error("Conv2d module is null");
    }

    // Extract convolution parameters
    auto options = convModule->options;

    // Set padding (ensure same padding on both sides for now)
    if (std::holds_alternative<torch::ExpandingArray<2>>(options.padding())) {
        auto pad_array = std::get<torch::ExpandingArray<2>>(options.padding());
        padding = {static_cast<int>((*pad_array)[0]),
                   static_cast<int>((*pad_array)[1])};
    } else {
        // Handle other padding types (kValid, kSame) for now default to 0
        padding = {0, 0};
    }

    // Set stride
    stride = {static_cast<int>((*options.stride())[0]),
              static_cast<int>((*options.stride())[1])};

    // Set dilation
    dilation = {static_cast<int>((*options.dilation())[0]),
               static_cast<int>((*options.dilation())[1])};

    // Set groups
    groups = options.groups();

    // Check for bias
    has_bias = options.bias();

    // Try to set sizes from weight matrix during construction
    // Note: conv2d has been initialized from convModule in the constructor's initializer list
    if (conv2d && conv2d->weight.defined()) {
        auto weight = conv2d->weight;
        // Weight shape: [out_channels, in_channels/groups, kernel_h, kernel_w]
        // int out_channels = weight.size(0);
        // int in_channels = weight.size(1) * groups;

        // Fix weight tensor properties for Alpha-CROWN compatibility
        bool weightNeedsFix = false;
        if (!weight.requires_grad() || !weight.is_contiguous() || weight.dtype() != torch::kFloat32) {
            weightNeedsFix = true;
            if (!weight.requires_grad()) {
            }
            if (!weight.is_contiguous()) {
            }
            if (weight.dtype() != torch::kFloat32) {
            }
        }
        
        // Automatically fix weight tensor: convert to Float32, make contiguous, and enable gradients
        // NOTE: Do NOT use detach() as it breaks the computation graph needed for Alpha-CROWN
        if (weightNeedsFix) {
            conv2d->weight = weight.contiguous().to(torch::kFloat32).requires_grad_(false);  // Network weights are constants;
        }
        
        // Similar checks and fixes for bias if it exists
        if (has_bias && conv2d->bias.defined()) {
            auto bias = conv2d->bias;
            bool biasNeedsFix = false;
            if (!bias.requires_grad() || !bias.is_contiguous() || bias.dtype() != torch::kFloat32) {
                biasNeedsFix = true;
                if (!bias.requires_grad()) {
                }
                if (!bias.is_contiguous()) {
                }
                if (bias.dtype() != torch::kFloat32) {
                }
            }
            
            // Automatically fix bias tensor
            // NOTE: Do NOT use detach() as it breaks the computation graph needed for Alpha-CROWN
            if (biasNeedsFix) {
                conv2d->bias = bias.contiguous().to(torch::kFloat32).requires_grad_(false);  // Network biases are constants;
            }
        }
    }
}

// Forward pass
torch::Tensor BoundedConvNode::forward(const torch::Tensor& input) {
    // Convert input to float32 and ensure contiguous
    torch::Tensor inputFloat = input.to(torch::kFloat32).contiguous();

    // Update input/output shapes only if not already set from ONNX metadata
    // (ONNX parsing sets the proper 4D shape, forward() might receive flattened input)
    if (input_shape.empty()) {
        for (int i = 0; i < input.dim(); ++i) {
            input_shape.push_back(input.size(i));
        }
    }

    torch::Tensor output;
    
    if (conv_dim == 1) {
        // 1D convolution
        if (!conv1d) {
            throw std::runtime_error("Conv1d module not initialized");
        }

        // Get weight and bias
        torch::Tensor weight = conv1d->weight.to(torch::kFloat32).contiguous();
        torch::Tensor bias = has_bias ? conv1d->bias.to(torch::kFloat32) : torch::Tensor();

        // Direct convolution for 1D (matrix mode not implemented for 1D)
        std::vector<int64_t> stride_64(stride.begin(), stride.end());
        std::vector<int64_t> padding_64(padding.begin(), padding.end());
        std::vector<int64_t> dilation_64(dilation.begin(), dilation.end());

        output = torch::nn::functional::conv1d(
            inputFloat, weight,
            torch::nn::functional::Conv1dFuncOptions()
                .bias(bias)
                .stride(stride_64)
                .padding(padding_64)
                .dilation(dilation_64)
                .groups(groups)
        );
    } else {
        // 2D convolution
        if (!conv2d) {
            throw std::runtime_error("Conv2d module not initialized");
        }

        // Get weight and bias
        torch::Tensor weight = conv2d->weight.to(torch::kFloat32).contiguous();
        torch::Tensor bias = has_bias ? conv2d->bias.to(torch::kFloat32) : torch::Tensor();

        if (mode == ConvMode::MATRIX) {
            // Matrix mode using im2col
            std::vector<int> kernel_size = {static_cast<int>(weight.size(2)),
                                           static_cast<int>(weight.size(3))};

            // Compute output shape
            std::vector<int> spatial_output = MatrixConvolution::computeConvOutputShape(
                {static_cast<int>(input.size(2)), static_cast<int>(input.size(3))},
                kernel_size, stride, padding, dilation
            );

            // Perform im2col transformation
            torch::Tensor input_matrix = MatrixConvolution::im2col(
                inputFloat, kernel_size, stride, padding, dilation, groups
            );

            // Matrix multiplication
            output = MatrixConvolution::matrixConvForward(
                input_matrix, weight, bias, spatial_output
            );
        } else {
            // Direct convolution
            std::vector<int64_t> stride_64(stride.begin(), stride.end());
            std::vector<int64_t> padding_64(padding.begin(), padding.end());
            std::vector<int64_t> dilation_64(dilation.begin(), dilation.end());

            output = torch::nn::functional::conv2d(
                inputFloat, weight,
                torch::nn::functional::Conv2dFuncOptions()
                    .bias(bias)
                    .stride(stride_64)
                    .padding(padding_64)
                    .dilation(dilation_64)
                    .groups(groups)
            );
        }
    }

    // Update output shape
    output_shape.clear();
    for (int i = 0; i < output.dim(); ++i) {
        output_shape.push_back(output.size(i));
    }

    // Update sizes for getInputSize/getOutputSize
    _input_size = input.numel() / input.size(0);  // Size per batch
    _output_size = output.numel() / output.size(0);  // Size per batch

    return output;
}

void BoundedConvNode::moveToDevice(const torch::Device& device)
{
    BoundedTorchNode::moveToDevice(device);
    if (conv1d) {
        conv1d->to(device);
    }
    if (conv2d) {
        conv2d->to(device);
    }
}

// Backward bound propagation
void BoundedConvNode::boundBackward(
    const BoundA& last_lA,
    const BoundA& last_uA,
    const Vector<BoundedTensor<torch::Tensor>>& inputBounds,
    Vector<Pair<BoundA, BoundA>>& outputA_matrices,
    torch::Tensor& lbias,
    torch::Tensor& ubias) {
        
    // Fix for missing input_shape - Forward pass is not always run before doing CROWN, hence not always dynamically set
    if (input_shape.empty()) {
        if (inputBounds.size() < 1) {
             throw std::runtime_error("BoundedConvNode: input_shape empty and no input bounds provided");
        }
        
        auto& lb = inputBounds[0].lower();
        int64_t total_input_size = lb.numel();
        
        if (conv_dim == 1) {
            // 1D convolution: weight shape [out_c, in_c, k_l]
            torch::Tensor weight = conv1d->weight;
            int64_t in_channels_per_group = weight.size(1);
            int64_t in_channels = in_channels_per_group * groups;
            
            // Assume [Batch, C, L]
            int64_t L = total_input_size / in_channels;
            input_shape = {1, static_cast<int>(in_channels), static_cast<int>(L)};
            
            // Compute output shape for 1D
            int64_t kernel_length = weight.size(2);
            int64_t out_l = (L + 2 * padding[0] - dilation[0] * (kernel_length - 1) - 1) / stride[0] + 1;
            output_shape = {1, static_cast<int>(weight.size(0)), static_cast<int>(out_l)};
            
            _input_size = total_input_size;
            _output_size = weight.size(0) * out_l;
        } else {
            // 2D convolution: weight shape [out_c, in_c, k_h, k_w]
            torch::Tensor weight = conv2d->weight;
            int64_t in_channels_per_group = weight.size(1);
            int64_t in_channels = in_channels_per_group * groups;
            int64_t kernel_h = weight.size(2);
            int64_t kernel_w = weight.size(3);
            
            // Assume [Batch, C, H, W]
            int64_t spatial_size = total_input_size / in_channels; // H * W
            
            // Find a valid factorization H * W = spatial_size where H >= kernel_h and W >= kernel_w
            int64_t H = 0, W = 0;
            
            // First try square layout
            int64_t sqrt_spatial = static_cast<int64_t>(std::sqrt(spatial_size));
            if (sqrt_spatial * sqrt_spatial == spatial_size && 
                sqrt_spatial >= kernel_h && sqrt_spatial >= kernel_w) {
                H = sqrt_spatial;
                W = sqrt_spatial;
            } else {
                // Find a valid non-square factorization
                // Prefer factorizations closer to square that satisfy kernel constraints
                int64_t best_H = 0, best_W = 0;
                int64_t best_diff = INT64_MAX;
                
                for (int64_t h = 1; h * h <= spatial_size; ++h) {
                    if (spatial_size % h == 0) {
                        int64_t w = spatial_size / h;
                        // Check both orientations: (h, w) and (w, h)
                        if (h >= kernel_h && w >= kernel_w) {
                            int64_t diff = std::abs(h - w);
                            if (diff < best_diff) {
                                best_diff = diff;
                                best_H = h;
                                best_W = w;
                            }
                        }
                        if (w >= kernel_h && h >= kernel_w) {
                            int64_t diff = std::abs(h - w);
                            if (diff < best_diff) {
                                best_diff = diff;
                                best_H = w;
                                best_W = h;
                            }
                        }
                    }
                }
                
                if (best_H > 0) {
                    H = best_H;
                    W = best_W;
                } else {
                    // No valid factorization found - use 1D-like layout
                    // Try [1, spatial_size] or [spatial_size, 1]
                    if (spatial_size >= kernel_w && 1 >= kernel_h) {
                        H = 1;
                        W = spatial_size;
                    } else if (spatial_size >= kernel_h && 1 >= kernel_w) {
                        H = spatial_size;
                        W = 1;
                    } else {
                        // Last resort: use closest square and let the convolution potentially fail
                        H = sqrt_spatial > 0 ? sqrt_spatial : 1;
                        W = spatial_size / H;
                    }
                }
            }
            
            input_shape = {1, static_cast<int>(in_channels), static_cast<int>(H), static_cast<int>(W)};
            
            // Also compute output shape
             std::vector<int> kernel_size_vec = {static_cast<int>(kernel_h),
                                          static_cast<int>(kernel_w)};
            std::vector<int> spatial_output = MatrixConvolution::computeConvOutputShape(
                {static_cast<int>(H), static_cast<int>(W)},
                kernel_size_vec, stride, padding, dilation
            );
            
            output_shape = {1, static_cast<int>(weight.size(0)), spatial_output[0], spatial_output[1]};
            
            _input_size = total_input_size;
            _output_size = weight.size(0) * spatial_output[0] * spatial_output[1];
        }
    }

    if (inputBounds.size() < 1) {
        throw std::runtime_error("BoundedConvNode expects at least one input");
    }

    // Get weight and bias
    torch::Tensor weight, bias;
    if (conv_dim == 1) {
        if (!conv1d) {
            throw std::runtime_error("Conv1d module not initialized");
        }
        weight = conv1d->weight.to(torch::kFloat32);
        bias = has_bias ? conv1d->bias.to(torch::kFloat32) : torch::Tensor();
    } else {
        if (!conv2d) {
            throw std::runtime_error("Conv2d module not initialized");
        }
        weight = conv2d->weight.to(torch::kFloat32);
        bias = has_bias ? conv2d->bias.to(torch::kFloat32) : torch::Tensor();
    }

    // Compute bounds for lower and upper
    torch::Tensor lA_bias_contrib, uA_bias_contrib;
    BoundA lA_x = boundOneSide(last_lA, weight, bias, lA_bias_contrib);
    BoundA uA_x = boundOneSide(last_uA, weight, bias, uA_bias_contrib);

    // Flatten output matrices if input bounds are flat (e.g. [3072])
    if (inputBounds.size() > 0 && inputBounds[0].lower().dim() == 1) {
        if (lA_x.isTensor()) {
            torch::Tensor t = lA_x.asTensor();
            if (t.dim() == 4) { // [B, C, H, W]
                 lA_x = BoundA(t.reshape({t.size(0), -1}));
            } else if (t.dim() == 5) { // [B, S, C, H, W]
                 lA_x = BoundA(t.reshape({t.size(0), t.size(1), -1}));
            }
        }
        if (uA_x.isTensor()) {
             torch::Tensor t = uA_x.asTensor();
             if (t.dim() == 4) {
                 uA_x = BoundA(t.reshape({t.size(0), -1}));
            } else if (t.dim() == 5) {
                 uA_x = BoundA(t.reshape({t.size(0), t.size(1), -1}));
            }
        }
    }

    // Prepare output matrices
    outputA_matrices.clear();
    outputA_matrices.append(Pair<BoundA, BoundA>(lA_x, uA_x));

    // Handle bias accumulation
    lbias = lA_bias_contrib;
    ubias = uA_bias_contrib;
}

BoundA BoundedConvNode::boundOneSide(const BoundA& last_A,
                                            const torch::Tensor& weight,
                                            const torch::Tensor& bias,
                                            torch::Tensor& sum_bias) {
    if (!last_A.defined()) {
        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(_device);
        sum_bias = torch::zeros({1}, options); // Should match expected bias shape logic
        return BoundA();
    }

    if (last_A.isTensor()) {
        // Matrix mode (Tensor) logic
        torch::Tensor last_A_tensor = last_A.asTensor();
        
        // Use transpose convolution for backward pass
        // Compute output padding for transpose convolution

        std::vector<int> output_padding = computeOutputPadding(input_shape, output_shape, weight);

        // Reshape last_A for transpose convolution
        auto shape = last_A_tensor.sizes().vec();
        
        torch::Tensor reshaped_last_A;
        bool was_flat = false;

        if (conv_dim == 1) {
            // 1D convolution reshaping
            if (shape.size() == 4) {
                 // [S, B, C, L]
                 reshaped_last_A = last_A_tensor.reshape({shape[0] * shape[1], shape[2], shape[3]});
            } else if (shape.size() == 3 && output_shape.size() >= 3) {
                 // [batch, spec, flat] -> [batch*spec, C, L]
                 reshaped_last_A = last_A_tensor.reshape({shape[0] * shape[1], output_shape[1], output_shape[2]});
                 was_flat = true;
            } else if (shape.size() == 2 && output_shape.size() >= 3) {
                 // [batch, flat] -> [batch, C, L]
                 reshaped_last_A = last_A_tensor.reshape({shape[0], output_shape[1], output_shape[2]});
                 was_flat = true;
            } else {
                 reshaped_last_A = last_A_tensor;
            }
        } else {
            // 2D convolution reshaping
            if (shape.size() == 5) {
                 // Reshape can be zero-copy if strides allow
                 reshaped_last_A = last_A_tensor.reshape({shape[0] * shape[1], shape[2], shape[3], shape[4]});
            } else if (shape.size() == 3 && output_shape.size() >= 4) {
                 // [batch, spec, flat] -> [batch*spec, C, H, W]
                 reshaped_last_A = last_A_tensor.reshape({shape[0] * shape[1], output_shape[1], output_shape[2], output_shape[3]});
                 was_flat = true;
            } else if (shape.size() == 2 && output_shape.size() >= 4) {
                 // [batch, flat] -> [batch, C, H, W]
                 reshaped_last_A = last_A_tensor.reshape({shape[0], output_shape[1], output_shape[2], output_shape[3]});
                 was_flat = true;
            } else {
                 reshaped_last_A = last_A_tensor;
            }
        }

        // Convert to int64_t
        //Could cache the stride, dilation etc since they are constant -- but def not the bottleneck rn
        std::vector<int64_t> stride_64(stride.begin(), stride.end());
        std::vector<int64_t> padding_64(padding.begin(), padding.end());
        std::vector<int64_t> dilation_64(dilation.begin(), dilation.end());
        std::vector<int64_t> output_padding_64(output_padding.begin(), output_padding.end());

        // Apply transpose convolution
        torch::Tensor next_A;
        if (conv_dim == 1) {
            next_A = torch::nn::functional::conv_transpose1d(
                reshaped_last_A, weight,
                torch::nn::functional::ConvTranspose1dFuncOptions()
                    .stride(stride_64)
                    .padding(padding_64)
                    .dilation(dilation_64)
                    .groups(groups)
                    .output_padding(output_padding_64)
            );
        } else {
            next_A = torch::nn::functional::conv_transpose2d(
                reshaped_last_A, weight,
                torch::nn::functional::ConvTranspose2dFuncOptions()
                    .stride(stride_64)
                    .padding(padding_64)
                    .dilation(dilation_64)
                    .groups(groups)
                    .output_padding(output_padding_64)
            );
        }

        // Reshape back
        if (conv_dim == 1) {
            // 1D convolution reshape back
            if (shape.size() == 4) {
                // [S*B, C, L] -> [S, B, C, L]
                next_A = next_A.view({shape[0], shape[1], next_A.size(1), next_A.size(2)});
            } else if (shape.size() == 3 && output_shape.size() >= 3) {
                // [B*S, C, L] -> [B, S, C, L]
                next_A = next_A.view({shape[0], shape[1], next_A.size(1), next_A.size(2)});
            } else if (shape.size() == 2 && output_shape.size() >= 3) {
                // [B, C, L] stays as is
                next_A = next_A.view({shape[0], next_A.size(1), next_A.size(2)});
            }
        } else {
            // 2D convolution reshape back
            if (shape.size() == 5) {
                next_A = next_A.view({shape[0], shape[1], next_A.size(1), next_A.size(2), next_A.size(3)});
            } else if (shape.size() == 3 && output_shape.size() >= 4) {
                // We reshaped last_A from [B, S, flat] -> [B*S, C, H, W] for conv_transpose2d.
                // Reshape back to [B, S, C, H, W] so downstream nodes preserve the spec dimension.
                next_A = next_A.view({shape[0], shape[1], next_A.size(1), next_A.size(2), next_A.size(3)});
            } else if (shape.size() == 2 && output_shape.size() >= 4) {
                // [B, flat] -> [B, C, H, W]
                next_A = next_A.view({shape[0], next_A.size(1), next_A.size(2), next_A.size(3)});
            }
        }
        
        // Handle bias
        // Each spatial dim across channels receives the same conv bias contribution
        // So below we sum across the spatial dims (ie the L or H,W) of each indexes A value, then multiply by the bias contribution for that channel
        //      eg rather than doing (a_1 * bias) + ... + (a_n * bias) we do (a_1 + ... + a_n) * bias
        // Then since we need a single scalar in final concretization we sum the bias value across the channels to get the conv's sum of constant contributions
        if (has_bias && bias.defined()) {
            if (conv_dim == 1) {
                // 1D convolution bias handling
                if (shape.size() == 4) {
                    // [S, B, C, L] where C is output channels of this conv
                    // Sum over spatial dimension L first: [S, B, C]
                    torch::Tensor sum_spatial = last_A_tensor.sum({3});
                    torch::Tensor bias_expanded = bias.unsqueeze(0).unsqueeze(0); // [1, 1, C]
                    torch::Tensor product = sum_spatial * bias_expanded; // [S, B, C]
                    sum_bias = product.sum(-1); // [S, B]
                } else if (shape.size() == 3) {
                    // Handle [B, C, L] or [S, B, flat]
                    if (output_shape.size() >= 3 && !was_flat) {
                        torch::Tensor sum_spatial = last_A_tensor.sum({2}); // [B, C]
                        torch::Tensor bias_expanded = bias.unsqueeze(0); // [1, C]
                        torch::Tensor product = sum_spatial * bias_expanded; // [B, C]
                        sum_bias = product.sum(-1); // [B]
                    } else if (output_shape.size() >= 3) {
                        // Reshape from flat to [S, B, C, L]
                        torch::Tensor reshaped = last_A_tensor.reshape({shape[0], shape[1],
                                                                       output_shape[1], output_shape[2]});
                        torch::Tensor sum_spatial = reshaped.sum({3}); // [S, B, C]
                        torch::Tensor bias_expanded = bias.unsqueeze(0).unsqueeze(0); // [1, 1, C]
                        torch::Tensor product = sum_spatial * bias_expanded; // [S, B, C]
                        sum_bias = product.sum(-1); // [S, B]
                    } else {
                        sum_bias = torch::zeros({shape[0], shape[1]}, last_A_tensor.options());
                    }
                } else if (shape.size() == 2) {
                    // [B, flat] - need to reshape
                    if (output_shape.size() >= 3) {
                        torch::Tensor reshaped = last_A_tensor.reshape({shape[0], output_shape[1], output_shape[2]});
                        torch::Tensor sum_spatial = reshaped.sum({2}); // [B, C]
                        torch::Tensor bias_expanded = bias.unsqueeze(0); // [1, C]
                        torch::Tensor product = sum_spatial * bias_expanded; // [B, C]
                        sum_bias = product.sum(-1).unsqueeze(0); // [1, B] for consistency
                    } else {
                        sum_bias = torch::zeros({1, shape[0]}, last_A_tensor.options());
                    }
                } else {
                    sum_bias = torch::zeros({1}, last_A_tensor.options());
                }
            } else {
                // 2D convolution bias handling
                if (shape.size() == 5) {
                    // [S, B, C, H, W] where C is output channels of this conv
                    // Sum over spatial dimensions H, W first: [S, B, C]
                    torch::Tensor sum_spatial = last_A_tensor.sum({3, 4});
                    torch::Tensor bias_expanded = bias.unsqueeze(0).unsqueeze(0); // [1, 1, C]
                    torch::Tensor product = sum_spatial * bias_expanded; // [S, B, C]
                    sum_bias = product.sum(-1); // [S, B]
                } else if (shape.size() == 4) {
                    // [B, C, H, W] - no spec dimension
                    torch::Tensor sum_spatial = last_A_tensor.sum({2, 3}); // [B, C]
                    torch::Tensor bias_expanded = bias.unsqueeze(0); // [1, C]
                    torch::Tensor product = sum_spatial * bias_expanded; // [B, C]
                    sum_bias = product.sum(-1); // [B]
                } else if (shape.size() == 3) {
                    // [S, B, flat] or [B, S, flat] - need to reshape
                    if (output_shape.size() >= 4) {
                        // Reshape to spatial dimensions
                        torch::Tensor reshaped = last_A_tensor.reshape({shape[0], shape[1],
                                                                       output_shape[1], output_shape[2], output_shape[3]});
                        torch::Tensor sum_spatial = reshaped.sum({3, 4}); // [S, B, C]
                        torch::Tensor bias_expanded = bias.unsqueeze(0).unsqueeze(0); // [1, 1, C]
                        torch::Tensor product = sum_spatial * bias_expanded; // [S, B, C]
                        sum_bias = product.sum(-1); // [S, B]
                    } else {
                        // Fallback
                        sum_bias = torch::zeros({shape[0], shape[1]}, last_A_tensor.options());
                    }
                } else if (shape.size() == 2) {
                    // [B, flat] - need to reshape
                    if (output_shape.size() >= 4) {
                        torch::Tensor reshaped = last_A_tensor.reshape({shape[0],
                                                                       output_shape[1], output_shape[2], output_shape[3]});
                        torch::Tensor sum_spatial = reshaped.sum({2, 3}); // [B, C]
                        torch::Tensor bias_expanded = bias.unsqueeze(0); // [1, C]
                        torch::Tensor product = sum_spatial * bias_expanded; // [B, C]
                        sum_bias = product.sum(-1).unsqueeze(0); // [1, B] for consistency
                    } else {
                        sum_bias = torch::zeros({1, shape[0]}, last_A_tensor.options());
                    }
                } else {
                    // Fallback
                    sum_bias = torch::zeros({1}, last_A_tensor.options());
                }
            }
        } else {
            // No bias term: bias contribution is exactly 0, but the tensor shape must match
            // the (spec, batch) convention used by the rest of the CROWN pipeline.
            if (shape.size() == 5 || shape.size() == 4) {
                // [S, B, C, H, W] -> [S, B] or [S, B, C, L] -> [S, B]
                sum_bias = torch::zeros({shape[0], shape[1]}, last_A_tensor.options());
            } else if (shape.size() == 4 || shape.size() == 3) {
                // [B, C, H, W] -> [B] or [B, C, L] -> [B]
                sum_bias = torch::zeros({shape[0]}, last_A_tensor.options());
            } else if (shape.size() == 3 || shape.size() == 2) {
                // [S, B, flat] or [B, S, flat] -> return a 2D (spec,batch)-like bias
                sum_bias = torch::zeros({shape[0], shape[1]}, last_A_tensor.options());
            } else if (shape.size() == 2) {
                // [B, flat] or [S, flat] -> follow the existing convention used when bias exists: [1, B]
                sum_bias = torch::zeros({1, shape[0]}, last_A_tensor.options());
            } else {
                sum_bias = torch::zeros({1}, last_A_tensor.options());
            }
        }

        return BoundA(next_A);
    } else {
        // Patches mode
        auto last_patches = last_A.asPatches();
        
        torch::Tensor pieces;
        
        if (last_patches->identity == 0) {
            torch::Tensor patches_tensor;
            if (!relu_followed) {
                std::vector<int64_t> output_shape_vec;
                for(int s : output_shape) output_shape_vec.push_back(s);
                
                torch::Tensor mask = create_valid_mask(
                    output_shape_vec,
                    last_patches->patches.device(),
                    weight.scalar_type(),
                    {last_patches->patches.size(-2), last_patches->patches.size(-1)}, // kernel size from patches shape
                    last_patches->stride,
                    last_patches->inserted_zeros,
                    last_patches->padding,
                    last_patches->output_padding,
                    last_patches->unstable_idx
                );
                patches_tensor = last_patches->patches * mask;
            } else {
                patches_tensor = last_patches->patches;
            }
            
            if (has_bias && bias.defined()) {
                
                torch::Tensor bias_view = bias.view({-1, 1, 1}); // [c, 1, 1]
                
                sum_bias = (patches_tensor * bias_view).sum({-3, -2, -1});
            } else {
                sum_bias = torch::zeros({1}, patches_tensor.options()); 
            }
            
            // flattened_patches = patches.reshape(-1, patches.size(-3), patches.size(-2), patches.size(-1))
            int64_t C = patches_tensor.size(-3);
            int64_t H = patches_tensor.size(-2);
            int64_t W = patches_tensor.size(-1);
            
            torch::Tensor flattened = patches_tensor.reshape({-1, C, H, W});
            
            // pieces = F.conv_transpose2d(flattened, insert_zeros(weight, inserted_zeros), stride=stride)
            torch::Tensor weight_processed = insert_zeros(weight, last_patches->inserted_zeros);
            
            std::vector<int64_t> stride_64(stride.begin(), stride.end());
            pieces = torch::nn::functional::conv_transpose2d(
                flattened, weight_processed,
                torch::nn::functional::ConvTranspose2dFuncOptions().stride(stride_64)
            );
            
            // Reshape pieces back
            // pieces = pieces.view(*patches.shape[:-3], pieces.size(-3), pieces.size(-2), pieces.size(-1))
            std::vector<int64_t> new_shape;
            for(int i=0; i<patches_tensor.dim()-3; ++i) new_shape.push_back(patches_tensor.size(i));
            new_shape.push_back(pieces.size(-3));
            new_shape.push_back(pieces.size(-2));
            new_shape.push_back(pieces.size(-1));
            
            pieces = pieces.view(new_shape);
            
        } else if (last_patches->identity == 1) {
            // Identity patches
            // weight: [out_c, in_c, k_h, k_w]
            
            if (last_patches->unstable_idx.has_value()) {
                throw std::runtime_error("BoundedConvNode: Sparse identity patches not implemented");
            } else {
                pieces = weight.view({weight.size(0), 1, 1, 1, weight.size(1), weight.size(2), weight.size(3)});
                // Expand
                std::vector<int64_t> expand_dims = {
                    weight.size(0), 
                    last_patches->output_shape[1], // batch
                    last_patches->output_shape[2], // out_h
                    last_patches->output_shape[3], // out_w
                    weight.size(1), weight.size(2), weight.size(3)
                };
                pieces = pieces.expand(expand_dims);
                
                // Bias
                if (has_bias) {
                    sum_bias = bias.view({-1, 1, 1, 1}).expand({
                        weight.size(0), last_patches->output_shape[1], last_patches->output_shape[2], last_patches->output_shape[3]
                    });
                } else {
                    sum_bias = torch::zeros({1}, weight.options());
                }
            }
        }
        
        // compute_patches_stride_padding
        std::vector<int64_t> new_padding_vec, new_stride_vec, new_output_padding_vec;
        
        std::vector<int64_t> p_pad = last_patches->padding;
        std::vector<int64_t> p_str = last_patches->stride;
        std::vector<int64_t> o_pad = {static_cast<int64_t>(padding[0]), static_cast<int64_t>(padding[1]), static_cast<int64_t>(padding[0]), static_cast<int64_t>(padding[1])}; // unify_shape(padding)
        std::vector<int64_t> o_str = {static_cast<int64_t>(stride[0]), static_cast<int64_t>(stride[1])};
        std::vector<int64_t> out_pad_prev = last_patches->output_padding;
        std::vector<int64_t> in_shape_vec;
        for(int s : input_shape) in_shape_vec.push_back(s);
        
        compute_patches_stride_padding(in_shape_vec, p_pad, p_str, o_pad, o_str, last_patches->inserted_zeros, out_pad_prev,
                                       new_padding_vec, new_stride_vec, new_output_padding_vec);
        
    
        if (last_patches->inserted_zeros == 0 && !is_shape_used(new_output_padding_vec) && 
            pieces.size(-1) > input_shape[3]) {
            
            // Patches too large, convert to matrix
            // return patches_to_matrix(...)
           
            (void)0;
        }
        
        return BoundA(last_patches->create_similar(
            pieces,
            new_stride_vec,
            new_padding_vec,
            new_output_padding_vec,
            0, 
            std::nullopt, // identity
            std::nullopt // input_shape
        ));
    }
}

std::vector<int> BoundedConvNode::computeOutputPadding(const std::vector<int>& input_shape,
                                                       const std::vector<int>& output_shape,
                                                       const torch::Tensor& weight) const {
    
    if (conv_dim == 1) {
        // 1D convolution
        int kernel_l = weight.size(2);
        int needed_l = input_shape[2];
        int current_l = (output_shape[2] - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kernel_l - 1) + 1;
        int output_padding0 = needed_l - current_l;
        
        // Ensure non-negative
        if (output_padding0 < 0) {
            output_padding0 = 0;
        }
        
        return {output_padding0};
    } else {
        // 2D convolution
        int kernel_h = weight.size(2);
        int kernel_w = weight.size(3);
        
        int needed_h = input_shape[2];
        int current_h = (output_shape[2] - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kernel_h - 1) + 1;
        int output_padding0 = needed_h - current_h;
        
        int needed_w = input_shape[3];
        int current_w = (output_shape[3] - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kernel_w - 1) + 1;
        int output_padding1 = needed_w - current_w;

        // Ensure non-negative
        if (output_padding0 < 0) {
            output_padding0 = 0;
        }
        
        if (output_padding1 < 0) {
            output_padding1 = 0;
        }

        return {output_padding0, output_padding1};
    }
}

// IBP computation
BoundedTensor<torch::Tensor> BoundedConvNode::computeIntervalBoundPropagation(
    const Vector<BoundedTensor<torch::Tensor>>& inputBounds) {

    if (inputBounds.empty()) {
        throw std::runtime_error("No input bounds provided for IBP");
    }

    const BoundedTensor<torch::Tensor>& input = inputBounds[0];

    // Get weight and bias
    torch::Tensor weight, bias;
    if (conv_dim == 1) {
        weight = conv1d->weight.to(torch::kFloat32);
        bias = has_bias ? conv1d->bias.to(torch::kFloat32) : torch::Tensor();
    } else {
        weight = conv2d->weight.to(torch::kFloat32);
        bias = has_bias ? conv2d->bias.to(torch::kFloat32) : torch::Tensor();
    }

    // Reshape input bounds if needed for convolution
    torch::Tensor input_lower = input.lower();
    torch::Tensor input_upper = input.upper();

    // Check if we need to reshape the input from flattened to proper conv shape
    if (input_lower.dim() == 1 || (input_lower.dim() == 2 && input_lower.size(0) == 1)) {
        // Input is flattened, need to reshape to [C, L] / [N, C, L] (1D) or [C, H, W] / [N, C, H, W] (2D)

        // Determine the proper shape based on stored input_shape or weight dimensions
        if (!input_shape.empty() && input_shape.size() >= 3) {
            // Use stored input shape from forward pass
            if (conv_dim == 1 && input_shape.size() == 3) {
                // [N, C, L]
                input_lower = input_lower.reshape({input_shape[0], input_shape[1], input_shape[2]});
                input_upper = input_upper.reshape({input_shape[0], input_shape[1], input_shape[2]});
            } else if (input_shape.size() == 4) {
                // [N, C, H, W]
                input_lower = input_lower.reshape({input_shape[0], input_shape[1], input_shape[2], input_shape[3]});
                input_upper = input_upper.reshape({input_shape[0], input_shape[1], input_shape[2], input_shape[3]});
            } else if (input_shape.size() == 3) {
                // [C, H, W] or [C, L]
                input_lower = input_lower.reshape({input_shape[0], input_shape[1], input_shape[2]});
                input_upper = input_upper.reshape({input_shape[0], input_shape[1], input_shape[2]});
            }
        } else if (conv_dim == 1) {
            // 1D convolution shape inference
            int in_channels = weight.size(1) * groups;  // weight is [out_channels, in_channels/groups, L]
            int total_elements = input_lower.numel();
            
            if (total_elements % in_channels == 0) {
                int L = total_elements / in_channels;
                input_lower = input_lower.reshape({in_channels, L});
                input_upper = input_upper.reshape({in_channels, L});
            } else {
                throw std::runtime_error("Input size incompatible with Conv1d weight dimensions");
            }
        } else {
            // Try to infer shape from weight dimensions and input size
            int in_channels = weight.size(1) * groups;  // weight is [out_channels, in_channels/groups, H, W]
            int total_elements = input_lower.numel();

            // For CIFAR-10: 3072 = 3 * 32 * 32
            if (total_elements == 3072 && in_channels == 3) {
                input_lower = input_lower.reshape({3, 32, 32});
                input_upper = input_upper.reshape({3, 32, 32});
            }
            // For other common sizes, compute H and W
            else if (total_elements % in_channels == 0) {
                int spatial_size = total_elements / in_channels;
                int kernel_h = weight.size(2);
                int kernel_w = weight.size(3);
                int sqrt_spatial = static_cast<int>(std::sqrt(spatial_size));

                // Check if perfect square AND satisfies kernel constraints
                if (sqrt_spatial * sqrt_spatial == spatial_size && 
                    sqrt_spatial >= kernel_h && sqrt_spatial >= kernel_w) {
                    // Perfect square that works with kernel - use as is
                    input_lower = input_lower.reshape({in_channels, sqrt_spatial, sqrt_spatial});
                    input_upper = input_upper.reshape({in_channels, sqrt_spatial, sqrt_spatial});
                } else {
                    // Need to find a valid spatial layout that satisfies kernel constraints
                    // Try to factorize spatial_size into H*W that satisfies H >= kernel_h and W >= kernel_w
                    int best_H = 0, best_W = 0;
                    int best_diff = INT32_MAX;
                    
                    // Try factorizations of spatial_size
                    for (int h = 1; h * h <= spatial_size; ++h) {
                        if (spatial_size % h == 0) {
                            int w = spatial_size / h;
                            // Check both orientations: (h, w) and (w, h)
                            if (h >= kernel_h && w >= kernel_w) {
                                int diff = std::abs(h - w);
                                if (diff < best_diff) {
                                    best_diff = diff;
                                    best_H = h;
                                    best_W = w;
                                }
                            }
                            if (w >= kernel_h && h >= kernel_w) {
                                int diff = std::abs(h - w);
                                if (diff < best_diff) {
                                    best_diff = diff;
                                    best_H = w;
                                    best_W = h;
                                }
                            }
                        }
                    }
                    
                    // If no valid factorization found, try 1D layouts
                    if (best_H == 0) {
                        if (kernel_h == 1 || spatial_size >= kernel_w) {
                            // Use [C, 1, W] layout
                            best_H = 1;
                            best_W = spatial_size;
                        } else if (kernel_w == 1 || spatial_size >= kernel_h) {
                            // Use [C, H, 1] layout
                            best_H = spatial_size;
                            best_W = 1;
                        } else {
                            // Last resort - use [C, 1, W] and hope for the best
                            best_H = 1;
                            best_W = spatial_size;
                        }
                    }

                    input_lower = input_lower.reshape({in_channels, best_H, best_W});
                    input_upper = input_upper.reshape({in_channels, best_H, best_W});
                }
            } else {
                throw std::runtime_error("Input size incompatible with convolution weight dimensions");
            }
        }
    }

    // Add batch dimension if needed
    if (conv_dim == 1 && input_lower.dim() == 2) {
        // Conv1d expects 3D input [N, C, L]
        input_lower = input_lower.unsqueeze(0);
        input_upper = input_upper.unsqueeze(0);
    } else if (conv_dim == 2 && input_lower.dim() == 3) {
        // Conv2d expects 4D input [N, C, H, W]
        input_lower = input_lower.unsqueeze(0);
        input_upper = input_upper.unsqueeze(0);
    }

    // Split weight into positive and negative parts
    torch::Tensor weight_pos = torch::clamp_min(weight, 0);
    torch::Tensor weight_neg = torch::clamp_max(weight, 0);

    // Convert vectors to int64_t for LibTorch
    std::vector<int64_t> stride_64(stride.begin(), stride.end());
    std::vector<int64_t> padding_64(padding.begin(), padding.end());
    std::vector<int64_t> dilation_64(dilation.begin(), dilation.end());

    torch::Tensor lower_bound, upper_bound;
    
    if (conv_dim == 1) {
        // Compute lower and upper bounds for 1D convolution
        // Lower bound: positive weights * lower input + negative weights * upper input
        lower_bound = torch::nn::functional::conv1d(
            input_lower, weight_pos,
            torch::nn::functional::Conv1dFuncOptions()
                .stride(stride_64)
                .padding(padding_64)
                .dilation(dilation_64)
                .groups(groups)
        ) + torch::nn::functional::conv1d(
            input_upper, weight_neg,
            torch::nn::functional::Conv1dFuncOptions()
                .stride(stride_64)
                .padding(padding_64)
                .dilation(dilation_64)
                .groups(groups)
        );

        // Upper bound: positive weights * upper input + negative weights * lower input
        upper_bound = torch::nn::functional::conv1d(
            input_upper, weight_pos,
            torch::nn::functional::Conv1dFuncOptions()
                .stride(stride_64)
                .padding(padding_64)
                .dilation(dilation_64)
                .groups(groups)
        ) + torch::nn::functional::conv1d(
            input_lower, weight_neg,
            torch::nn::functional::Conv1dFuncOptions()
                .stride(stride_64)
                .padding(padding_64)
                .dilation(dilation_64)
                .groups(groups)
        );
        
        // Add bias if present
        if (has_bias && bias.defined()) {
            lower_bound = lower_bound + bias.unsqueeze(0).unsqueeze(-1);
            upper_bound = upper_bound + bias.unsqueeze(0).unsqueeze(-1);
        }
    } else {
        // Compute lower and upper bounds for 2D convolution
        // Lower bound: positive weights * lower input + negative weights * upper input
        lower_bound = torch::nn::functional::conv2d(
            input_lower, weight_pos,
            torch::nn::functional::Conv2dFuncOptions()
                .stride(stride_64)
                .padding(padding_64)
                .dilation(dilation_64)
                .groups(groups)
        ) + torch::nn::functional::conv2d(
            input_upper, weight_neg,
            torch::nn::functional::Conv2dFuncOptions()
                .stride(stride_64)
                .padding(padding_64)
                .dilation(dilation_64)
                .groups(groups)
        );

        // Upper bound: positive weights * upper input + negative weights * lower input
        upper_bound = torch::nn::functional::conv2d(
            input_upper, weight_pos,
            torch::nn::functional::Conv2dFuncOptions()
                .stride(stride_64)
                .padding(padding_64)
                .dilation(dilation_64)
                .groups(groups)
        ) + torch::nn::functional::conv2d(
            input_lower, weight_neg,
            torch::nn::functional::Conv2dFuncOptions()
                .stride(stride_64)
                .padding(padding_64)
                .dilation(dilation_64)
                .groups(groups)
        );

        // Add bias if present
        if (has_bias && bias.defined()) {
            lower_bound = lower_bound + bias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1);
            upper_bound = upper_bound + bias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1);
        }
    }

    // Flatten the output if the original input was flattened
    if (input.lower().dim() == 1 || (input.lower().dim() == 2 && input.lower().size(0) == 1)) {
        // Flatten back to match the expected format
        lower_bound = lower_bound.flatten();
        upper_bound = upper_bound.flatten();
    }

    return BoundedTensor<torch::Tensor>(lower_bound, upper_bound);
}

// Size getters and setters
unsigned BoundedConvNode::getInputSize() const {
    return _input_size;
}

unsigned BoundedConvNode::getOutputSize() const {
    return _output_size;
}

void BoundedConvNode::setInputSize(unsigned size) {
    _input_size = size;
}

void BoundedConvNode::setOutputSize(unsigned size) {
    _output_size = size;
}

unsigned BoundedConvNode::inferOutputSize(unsigned inputSize) const {
    if (conv_dim == 1) {
        if (!conv1d) return 0;
        torch::Tensor weight = conv1d->weight;
        
        int64_t out_channels = weight.size(0);
        int64_t in_channels_per_group = weight.size(1);
        int64_t in_channels = in_channels_per_group * groups;
        
        if (in_channels == 0) return 0;
        
        int64_t L = inputSize / in_channels;
        if (L <= 0) return 0;
        
        // Compute output length for 1D convolution
        int64_t kernel_length = weight.size(2);
        int64_t out_l = (L + 2 * padding[0] - dilation[0] * (kernel_length - 1) - 1) / stride[0] + 1;
        
        return static_cast<unsigned>(out_channels * out_l);
    } else {
        if (!conv2d) return 0;
        torch::Tensor weight = conv2d->weight;
        
        int64_t out_channels = weight.size(0);
        int64_t in_channels_per_group = weight.size(1);
        int64_t in_channels = in_channels_per_group * groups;
        
        if (in_channels == 0) return 0;
        
        int64_t spatial_dim_sq = inputSize / in_channels;
        if (spatial_dim_sq <= 0) return 0;
        
        int64_t H = static_cast<int64_t>(std::sqrt(spatial_dim_sq));
        int64_t W = H;
        
        // Verify assumption roughly
        if (in_channels * H * W != inputSize) {
        }
        
        std::vector<int> kernel_size = {static_cast<int>(weight.size(2)),
                                       static_cast<int>(weight.size(3))};
                                       
        std::vector<int> spatial_output = MatrixConvolution::computeConvOutputShape(
            {static_cast<int>(H), static_cast<int>(W)},
            kernel_size, stride, padding, dilation
        );
        
        if (spatial_output.size() < 2) return 0;
        
        return static_cast<unsigned>(out_channels * spatial_output[0] * spatial_output[1]);
    }
}

} // namespace NLR
