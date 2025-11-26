// BoundedConvNode.cpp - Implementation of convolution layer with bound propagation
#include "BoundedConvNode.h"
#include "conv/MatrixConvolution.h"
#include "Patches.h" // Include patches definition
#include <torch/nn/functional.h>
#include <stdexcept>
#include <cmath>
#include <cstdio>

namespace NLR {

// Constructor for Conv2d
BoundedConvNode::BoundedConvNode(const torch::nn::Conv2d& convModule,
                                 ConvMode mode,
                                 const String& name)
    : conv2d(convModule), mode(mode) {

    _nodeName = name;
    _nodeIndex = 0;
    _input_size = 0;
    _output_size = 0;
    relu_followed = false;
    patches_start = true;

    initializeFromConv2d(convModule);
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
        // Handle other padding types (kValid, kSame) - default to 0
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
    if (convModule->weight.defined()) {
        auto weight = convModule->weight;
        // Weight shape: [out_channels, in_channels/groups, kernel_h, kernel_w]
        // int out_channels = weight.size(0);
        // int in_channels = weight.size(1) * groups;

        // Note: Full input/output sizes will be determined dynamically during forward pass
        // based on actual input tensor dimensions

        // Diagnostic: Verify weight tensor properties
        if (!weight.requires_grad()) {
            printf("[WARNING] BoundedConvNode: Weight tensor does not have requires_grad=True\n");
        }
        if (!weight.is_contiguous()) {
            printf("[WARNING] BoundedConvNode: Weight tensor is not contiguous.\n");
        }
        if (weight.dtype() != torch::kFloat32) {
            printf("[WARNING] BoundedConvNode: Weight tensor dtype is not Float32.\n");
        }
    }
}


// Forward pass
torch::Tensor BoundedConvNode::forward(const torch::Tensor& input) {
    // Convert input to float32 and ensure contiguous
    torch::Tensor inputFloat = input.to(torch::kFloat32).contiguous();

    // Update input/output shapes
    input_shape.clear();
    for (int i = 0; i < input.dim(); ++i) {
        input_shape.push_back(input.size(i));
    }

    // 2D convolution only
    if (!conv2d) {
        throw std::runtime_error("Conv2d module not initialized");
    }

    // Get weight and bias
    torch::Tensor weight = conv2d->weight.to(torch::kFloat32).contiguous();
    torch::Tensor bias = has_bias ? conv2d->bias.to(torch::kFloat32) : torch::Tensor();

    torch::Tensor output;

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
        // Direct convolution (fallback or patches mode preparation)
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

// Backward bound propagation
void BoundedConvNode::boundBackward(
    const BoundA& last_lA,
    const BoundA& last_uA,
    const Vector<BoundedTensor<torch::Tensor>>& inputBounds,
    Vector<Pair<BoundA, BoundA>>& outputA_matrices,
    torch::Tensor& lbias,
    torch::Tensor& ubias) {

    // TODO: Fix Conv bound propagation - currently causes segfault
    // The issue is that Conv nodes need proper spatial dimension tracking

    // Fix for missing input_shape (when forward pass wasn't run)
    if (input_shape.empty()) {
        if (inputBounds.size() < 1) {
             throw std::runtime_error("BoundedConvNode: input_shape empty and no input bounds provided");
        }
        
        auto& lb = inputBounds[0].lower();
        // Infer shape from total size and weights
        // weight: [out_c, in_c, k_h, k_w]
        torch::Tensor weight = conv2d->weight;
        int64_t in_channels_per_group = weight.size(1);
        int64_t in_channels = in_channels_per_group * groups;
        int64_t total_input_size = lb.numel();
        
        // Assume [Batch, C, H, W]
        // If lb is [N], assume Batch=1
        int64_t spatial_dim_sq = total_input_size / in_channels; // H * W
        int64_t H = static_cast<int64_t>(std::sqrt(spatial_dim_sq));
        int64_t W = H;
        
        if (in_channels * H * W != total_input_size) {
             printf("WARNING: BoundedConvNode inferred shape mismatch: total=%lld, C=%lld, H=%lld, W=%lld\n", 
                    (long long)total_input_size, (long long)in_channels, (long long)H, (long long)W);
        }
        
        input_shape = {1, static_cast<int>(in_channels), static_cast<int>(H), static_cast<int>(W)};
        
        // Also compute output shape
         std::vector<int> kernel_size = {static_cast<int>(weight.size(2)),
                                       static_cast<int>(weight.size(3))};
        std::vector<int> spatial_output = MatrixConvolution::computeConvOutputShape(
            {static_cast<int>(H), static_cast<int>(W)},
            kernel_size, stride, padding, dilation
        );
        
        output_shape = {1, static_cast<int>(weight.size(0)), spatial_output[0], spatial_output[1]};
        
        _input_size = total_input_size;
        _output_size = weight.size(0) * spatial_output[0] * spatial_output[1];
    }

    if (inputBounds.size() < 1) {
        throw std::runtime_error("BoundedConvNode expects at least one input");
    }

    // Get weight and bias from Conv2d
    if (!conv2d) {
        throw std::runtime_error("Conv2d module not initialized");
    }
    torch::Tensor weight = conv2d->weight.to(torch::kFloat32);
    torch::Tensor bias = has_bias ? conv2d->bias.to(torch::kFloat32) : torch::Tensor();

    // Debug output for conv bounds
    printf("[Conv Debug] Node index: %u, weight shape: [%lld, %lld, %lld, %lld]\n",
           _nodeIndex, (long long)weight.size(0), (long long)weight.size(1),
           (long long)weight.size(2), (long long)weight.size(3));
    printf("[Conv Debug] Input shape: [");
    for(size_t i = 0; i < input_shape.size(); ++i) {
        if(i > 0) printf(", ");
        printf("%d", input_shape[i]);
    }
    printf("], Output shape: [");
    for(size_t i = 0; i < output_shape.size(); ++i) {
        if(i > 0) printf(", ");
        printf("%d", output_shape[i]);
    }
    printf("]\n");

    // Compute bounds for lower and upper
    torch::Tensor lA_bias_contrib, uA_bias_contrib;
    BoundA lA_x = boundOneSide(last_lA, weight, bias, lA_bias_contrib);
    BoundA uA_x = boundOneSide(last_uA, weight, bias, uA_bias_contrib);

    // Debug bias contributions
    if (lA_bias_contrib.defined()) {
        printf("[Conv Debug] lA_bias_contrib shape: [");
        auto shape = lA_bias_contrib.sizes();
        for(size_t i = 0; i < shape.size(); ++i) {
            if(i > 0) printf(", ");
            printf("%lld", (long long)shape[i]);
        }
        printf("], values (first 5): ");
        auto flat = lA_bias_contrib.flatten();
        for(int i = 0; i < std::min(5, (int)flat.numel()); ++i) {
            if(i > 0) printf(", ");
            printf("%.6f", flat[i].item<float>());
        }
        printf("\n");
    }
    
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

    // Add placeholders for weight and bias (if they are perturbed, which we don't support yet)
    if (has_bias) {
        // Just placeholders, not actual bounds propagation to weights/bias
    }
}

BoundA BoundedConvNode::boundOneSide(const BoundA& last_A,
                                            const torch::Tensor& weight,
                                            const torch::Tensor& bias,
                                            torch::Tensor& sum_bias) {
    if (!last_A.defined()) {
        sum_bias = torch::zeros({1}); // Should match expected bias shape logic
        return BoundA();
    }

    if (last_A.isTensor()) {
        // Matrix mode (Tensor) logic
        torch::Tensor last_A_tensor = last_A.asTensor();
        
        // For 2D convolution, use transpose convolution for backward pass
        // Compute output padding for transpose convolution
        std::vector<int> output_padding = computeOutputPadding(input_shape, output_shape, weight);

        // Reshape last_A for transpose convolution
        // last_A shape: [batch, spec, C, H, W] or [spec, C, H, W]
        // We need to treat (batch * spec) as batch dimension for conv_transpose2d
        auto shape = last_A_tensor.sizes().vec();
        
        torch::Tensor reshaped_last_A;
        bool was_flat = false;

        if (shape.size() == 5) {
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

        // Convert to int64_t for LibTorch
        std::vector<int64_t> stride_64(stride.begin(), stride.end());
        std::vector<int64_t> padding_64(padding.begin(), padding.end());
        std::vector<int64_t> dilation_64(dilation.begin(), dilation.end());
        std::vector<int64_t> output_padding_64(output_padding.begin(), output_padding.end());

        // Apply transpose convolution
        torch::Tensor next_A = torch::nn::functional::conv_transpose2d(
            reshaped_last_A, weight,
            torch::nn::functional::ConvTranspose2dFuncOptions()
                .stride(stride_64)
                .padding(padding_64)
                .dilation(dilation_64)
                .groups(groups)
                .output_padding(output_padding_64)
        );

        // Reshape back
        if (shape.size() == 5) {
            next_A = next_A.view({shape[0], shape[1], next_A.size(1), next_A.size(2), next_A.size(3)});
        }
        
        // Handle bias
        if (has_bias && bias.defined()) {
            // From auto_LiRPA: sum_bias = torch.einsum('sb...,...->sb', last_A, bias)
            // For conv2d, last_A comes from output layer and has shape matching output channels
            // bias has shape [out_channels] of this conv layer

            if (shape.size() == 5) {
                // [S, B, C, H, W] where C is output channels of this conv
                // Sum over spatial dimensions H, W first: [S, B, C]
                torch::Tensor sum_spatial = last_A_tensor.sum({3, 4});

                // Now multiply by bias and sum over C
                // sum_spatial: [S, B, C]
                // bias: [C]
                // Result: [S, B]

                // Use unsqueeze and multiplication for correct broadcasting
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
        } else {
            if (shape.size() >= 2)
                sum_bias = torch::zeros({shape[0], shape[1]}, last_A_tensor.options());
            else
                sum_bias = torch::zeros({1}, last_A_tensor.options());
        }

        return BoundA(next_A);
    } else {
        // Patches mode
        auto last_patches = last_A.asPatches();
        
        // assert self.conv_dim == 2 (implied)
        
        // Python logic:
        // if last_A.identity == 0:
        //    if not self.relu_followed:
        //       pad patches
        //    patches = last_A.patches
        //    sum_bias = ...
        //    flatten patches, conv_transpose2d, reshape
        // elif last_A.identity == 1:
        //    pieces = weight.view(...)
        //    sum_bias = ...
        
        torch::Tensor pieces;
        
        if (last_patches->identity == 0) {
            torch::Tensor patches_tensor;
            if (!relu_followed) {
                // Pad patches
                // one_d_unfolded_r = create_valid_mask(...)
                // patches = last_A.patches * one_d_unfolded_r
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
                // sum_bias = torch.einsum('sb...chw,c->sb...', patches, bias)
                // Patches: [out_c, batch, out_h, out_w, c, h, w]
                // bias: [c] (input channels? No, bias is output channels of current layer, which corresponds to 'c' in patches?)
                // Wait, in backward pass, `bias` corresponds to the bias of *this* layer.
                // But we are propagating backwards.
                // The bias of this layer contributes to the bounds of the *next* layer (which passed us last_A).
                // `last_A` maps output of *this* layer to final output.
                // So `last_A` * `bias` is the contribution.
                // `last_A` has input dimension equal to *this* layer's output dimension.
                // Patches structure:
                // [batch, out_c, out_h, out_w, in_c, k_h, k_w] (from patches.py doc)
                // But patches.py says: "shape of Patches.patches is [batch_size, num_of_patches, out_channel, in_channel, M, M]"
                // And logic says: "Patches either has [out_c, batch, out_h, out_w, c, h, w] or ..."
                
                // In `bound_backward` (python):
                // sum_bias = torch.einsum('sb...chw,c->sb...', patches, x[2].lower)
                // x[2] is bias.
                // patches has 'c' dimension matching bias size?
                // patches: [..., c, h, w]. `c` is input channel of the patch (which is output channel of this layer).
                // Yes.
                
                // Calculate sum_bias using reduction.
                // Multiply patches by bias broadcasted to [..., c, 1, 1]
                // Then sum over c, h, w.
                
                torch::Tensor bias_view = bias.view({-1, 1, 1}); // [c, 1, 1]
                // We need to broadcast bias to match patches prefix
                // patches: [..., C, H, W]
                // bias: [C]
                // patches * bias_view -> [..., C, H, W]
                // sum over C, H, W
                
                sum_bias = (patches_tensor * bias_view).sum({-3, -2, -1});
            } else {
                sum_bias = torch::zeros({1}, patches_tensor.options()); // Shape?
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
            // pieces = weight.view(...)
            // weight: [out_c, in_c, k_h, k_w]
            
            if (last_patches->unstable_idx.has_value()) {
                // Sparse
                // pieces = weight[last_A.unstable_idx[0]]
                // Expand batch dim
                // ...
                // For C++, implementing full sparse logic might be tedious.
                // Let's assume dense for now or simple case.
                throw std::runtime_error("BoundedConvNode: Sparse identity patches not implemented");
            } else {
                // weight size(0) == last_A.shape[0]? (output channels)
                // pieces = weight.view(weight.size(0), 1, 1, 1, weight.size(1), weight.size(2), weight.size(3))
                //          .expand(-1, *last_A.shape[1:4], -1, -1, -1)
                // pieces: [out_c, batch, out_h, out_w, in_c, k_h, k_w]
                
                // bias calculation
                // ...
                
                // TODO: Implement full logic.
                // For now, just use fallback to matrix if identity patches.
                // But wait, identity patches is optimizing.
                
                // pieces = weight.view(weight.size(0), 1, 1, 1, weight.size(1), weight.size(2), weight.size(3))
                // This assumes standard patches shape [out_c, batch, out_h, out_w, c, h, w]
                
                pieces = weight.view({weight.size(0), 1, 1, 1, weight.size(1), weight.size(2), weight.size(3)});
                // Expand
                std::vector<int64_t> expand_dims = {
                    weight.size(0), 
                    last_patches->output_shape[1], // batch? shape[1]
                    last_patches->output_shape[2], // out_h
                    last_patches->output_shape[3], // out_w
                    weight.size(1), weight.size(2), weight.size(3)
                };
                pieces = pieces.expand(expand_dims);
                
                // Bias
                if (has_bias) {
                    // sum_bias = bias.view(-1, 1, 1, 1).expand(...)
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
        
        // Check if patches too large (fallback condition)
        // if (inserted_zeros == 0 and not is_shape_used(output_padding) and pieces.shape[-1] > self.input_shape[-1]):
        //    convert to matrix
        
        if (last_patches->inserted_zeros == 0 && !is_shape_used(new_output_padding_vec) && 
            pieces.size(-1) > input_shape[3]) { // assume input_shape[3] is width
            
            // Patches too large, convert to matrix
            // return patches_to_matrix(...)
            
            // As patches_to_matrix is throwing, we just throw/warn here.
            printf("BoundedConvNode: Patches too large, but conversion not implemented. Continuing with patches (might be slow or wrong).\n");
        }
        
        return BoundA(last_patches->create_similar(
            pieces,
            new_stride_vec,
            new_padding_vec,
            new_output_padding_vec,
            0, // inserted_zeros = 0 (calculated in python? No, python says: "padding, stride, output_padding = compute...")
            // Actually python logic line 224:
            // padding, stride, output_padding = compute_patches_stride_padding(...)
            
            // And "inserted_zeros = last_A.inserted_zeros if last_A is not None else 0"
            // Wait, `compute_patches_stride_padding` returns new stride/padding.
            // Does it return new inserted_zeros? No.
            // inserted_zeros is kept?
            // Python line 242: `new_patches = last_A.create_similar(pieces, stride=stride, padding=padding, output_padding=output_padding, identity=0, input_shape=self.input_shape)`
            // It doesn't pass `inserted_zeros`. So it uses default/previous?
            // `create_similar` uses `self.inserted_zeros` if not passed.
            // So it preserves `inserted_zeros`?
            
            // Wait, in `BoundConv`, `inserted_zeros` is passed to `compute_patches_stride_padding`.
            // But `create_similar` doesn't update it.
            // So `inserted_zeros` stays same?
            
            // Ah, `BoundConvTranspose` updates `inserted_zeros`. `BoundConv` does not (it is standard conv).
            // Standard conv doesn't introduce inserted zeros.
            
            std::nullopt, // identity
            std::nullopt // input_shape
        ));
    }
}

std::vector<int> BoundedConvNode::computeOutputPadding(const std::vector<int>& input_shape,
                                                       const std::vector<int>& output_shape,
                                                       const torch::Tensor& weight) const {
    // Based on auto_LiRPA's formula for 2D convolution
    // output_padding = input_shape - (output_shape - 1) * stride + 2 * padding + 1 + (kernel_size - 1) * dilation
    
    // Calculate standard output size for convolution to see what the padding needs to be to match input_shape
    // when we do conv_transpose2d
    
    // The standard formula for output size of conv is:
    // out = floor((in + 2*pad - dilation*(kernel-1) - 1)/stride + 1)
    
    // The formula for output size of conv_transpose is:
    // out = (in - 1)*stride - 2*pad + dilation*(kernel-1) + output_padding + 1
    
    // We want the output of conv_transpose2d (which corresponds to input_shape of the original conv)
    // to match input_shape.
    // So: input_shape = (output_shape - 1)*stride - 2*padding + dilation*(kernel-1) + output_padding + 1
    // => output_padding = input_shape - ((output_shape - 1)*stride - 2*padding + dilation*(kernel-1) + 1)
    
    // Note: The formula in auto_LiRPA might be slightly different or I copied it wrong above.
    // Let's verify against PyTorch docs.
    // H_out = (H_in - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kernel_size[0] - 1) + output_padding[0] + 1
    // We want H_out to be equal to input_shape[2] (Height).
    // H_in is output_shape[2].
    
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
        // If negative, clamp to 0
        output_padding0 = 0;
    }
    
    if (output_padding1 < 0) {
        output_padding1 = 0;
    }

    return {output_padding0, output_padding1};
}

// IBP computation
BoundedTensor<torch::Tensor> BoundedConvNode::computeIntervalBoundPropagation(
    const Vector<BoundedTensor<torch::Tensor>>& inputBounds) {

    if (inputBounds.empty()) {
        throw std::runtime_error("No input bounds provided for IBP");
    }

    const BoundedTensor<torch::Tensor>& input = inputBounds[0];

    // Get weight and bias from Conv2d
    torch::Tensor weight = conv2d->weight.to(torch::kFloat32);
    torch::Tensor bias = has_bias ? conv2d->bias.to(torch::kFloat32) : torch::Tensor();

    // Split weight into positive and negative parts
    torch::Tensor weight_pos = torch::clamp_min(weight, 0);
    torch::Tensor weight_neg = torch::clamp_max(weight, 0);

    // Convert vectors to int64_t for LibTorch
    std::vector<int64_t> stride_64(stride.begin(), stride.end());
    std::vector<int64_t> padding_64(padding.begin(), padding.end());
    std::vector<int64_t> dilation_64(dilation.begin(), dilation.end());

    // Compute lower and upper bounds for 2D convolution
    // Lower bound: positive weights * lower input + negative weights * upper input
    torch::Tensor lower_bound = torch::nn::functional::conv2d(
        input.lower(), weight_pos,
        torch::nn::functional::Conv2dFuncOptions()
            .stride(stride_64)
            .padding(padding_64)
            .dilation(dilation_64)
            .groups(groups)
    ) + torch::nn::functional::conv2d(
        input.upper(), weight_neg,
        torch::nn::functional::Conv2dFuncOptions()
            .stride(stride_64)
            .padding(padding_64)
            .dilation(dilation_64)
            .groups(groups)
    );

    // Upper bound: positive weights * upper input + negative weights * lower input
    torch::Tensor upper_bound = torch::nn::functional::conv2d(
        input.upper(), weight_pos,
        torch::nn::functional::Conv2dFuncOptions()
            .stride(stride_64)
            .padding(padding_64)
            .dilation(dilation_64)
            .groups(groups)
    ) + torch::nn::functional::conv2d(
        input.lower(), weight_neg,
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
         // Try to see if it's not square? 
         // For now, assume square if we can't tell otherwise.
         // If it's not square, this heuristic fails. 
         // But auto_LiRPA generally assumes shapes are known or can be propagated.
         printf("BoundedConvNode::inferOutputSize warning: inputSize %u not matching [C=%lld, H=%lld, W=%lld]\n", 
                inputSize, (long long)in_channels, (long long)H, (long long)W);
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

} // namespace NLR
