// MatrixConvolution.cpp - Implementation of matrix-based convolution operations
#include "MatrixConvolution.h"
#include <stdexcept>

namespace NLR {

torch::Tensor MatrixConvolution::im2col(
    const torch::Tensor& input,
    const std::vector<int>& kernel_size,
    const std::vector<int>& stride,
    const std::vector<int>& padding,
    const std::vector<int>& dilation,
    int groups) {

    // Input shape: [batch_size, in_channels, height, width]
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_height = input.size(2);
    int input_width = input.size(3);

    // Add padding to input if needed
    torch::Tensor padded_input = input;
    if (padding[0] > 0 || padding[1] > 0) {
        padded_input = torch::nn::functional::pad(
            input,
            torch::nn::functional::PadFuncOptions({padding[1], padding[1], padding[0], padding[0]})
        );
    }

    int padded_height = padded_input.size(2);
    int padded_width = padded_input.size(3);

    // Calculate output dimensions
    int output_height = (padded_height - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1;
    int output_width = (padded_width - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1;

    // Use unfold to create the im2col matrix
    // First unfold along height dimension
    torch::Tensor unfolded = padded_input.unfold(2, kernel_size[0], stride[0]);
    // Then unfold along width dimension
    unfolded = unfolded.unfold(3, kernel_size[1], stride[1]);

    // Reshape to get im2col format
    // Shape: [batch_size, in_channels * kernel_h * kernel_w, output_h * output_w]
    unfolded = unfolded.permute({0, 1, 4, 5, 2, 3});
    unfolded = unfolded.reshape({
        batch_size,
        in_channels * kernel_size[0] * kernel_size[1],
        output_height * output_width
    });

    return unfolded;
}


torch::Tensor MatrixConvolution::col2im(
    const torch::Tensor& col,
    const std::vector<int>& output_size,
    const std::vector<int>& kernel_size,
    const std::vector<int>& stride,
    const std::vector<int>& padding,
    const std::vector<int>& dilation,
    int groups) {

    // This is a simplified implementation
    // In practice, we'll use torch::nn::functional::conv_transpose2d
    // for the backward pass which is more efficient

    int batch_size = col.size(0);
    int out_height = output_size[0];
    int out_width = output_size[1];
    int out_channels = output_size[2];

    // Create output tensor
    torch::Tensor output = torch::zeros({batch_size, out_channels, out_height, out_width},
                                         col.options());

    // Note: Full col2im implementation would require careful index mapping
    // For now, we'll rely on PyTorch's built-in operations in the actual backward pass

    return output;
}

std::vector<int> MatrixConvolution::computeConvOutputShape(
    const std::vector<int>& input_shape,
    const std::vector<int>& kernel_size,
    const std::vector<int>& stride,
    const std::vector<int>& padding,
    const std::vector<int>& dilation) {

    std::vector<int> output_shape;

    for (size_t i = 0; i < input_shape.size(); ++i) {
        int output_dim = (input_shape[i] + 2 * padding[i] - dilation[i] * (kernel_size[i] - 1) - 1) / stride[i] + 1;
        output_shape.push_back(output_dim);
    }

    return output_shape;
}

std::vector<int> MatrixConvolution::computeTransposeOutputPadding(
    const std::vector<int>& input_shape,
    const std::vector<int>& output_shape,
    const std::vector<int>& kernel_size,
    const std::vector<int>& stride,
    const std::vector<int>& padding,
    const std::vector<int>& dilation) {

    std::vector<int> output_padding;

    // Based on auto_LiRPA's implementation
    for (size_t i = 0; i < input_shape.size() - 2; ++i) {
        int computed_padding = input_shape[i + 2] -
                               ((output_shape[i] - 1) * stride[i] + 2 * padding[i] + 1 +
                                (kernel_size[i] - 1) * dilation[i]);
        output_padding.push_back(computed_padding);
    }

    return output_padding;
}

torch::Tensor MatrixConvolution::matrixConvForward(
    const torch::Tensor& input_matrix,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const std::vector<int>& output_shape) {

    // Input matrix shape: [batch_size, in_channels * kernel_h * kernel_w, output_h * output_w]
    // Weight shape: [out_channels, in_channels, kernel_h, kernel_w]

    int batch_size = input_matrix.size(0);
    int out_channels = weight.size(0);
    int in_channels = weight.size(1);
    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);

    // Reshape weight for matrix multiplication
    torch::Tensor weight_matrix = weight.view({out_channels, in_channels * kernel_h * kernel_w});

    // Perform matrix multiplication
    // Result shape: [batch_size, out_channels, output_h * output_w]
    torch::Tensor output = torch::bmm(
        weight_matrix.unsqueeze(0).expand({batch_size, -1, -1}),
        input_matrix
    );

    // Add bias if present
    if (bias.defined()) {
        output = output + bias.unsqueeze(0).unsqueeze(2);
    }

    // Reshape to spatial dimensions
    output = output.view({batch_size, out_channels, output_shape[0], output_shape[1]});

    return output;
}

torch::Tensor MatrixConvolution::matrixConvBackward(
    const torch::Tensor& grad_output,
    const torch::Tensor& weight,
    const std::vector<int>& input_shape,
    const std::vector<int>& stride,
    const std::vector<int>& padding,
    const std::vector<int>& dilation,
    const std::vector<int>& output_padding) {

    // For backward pass, we use transpose convolution
    // This is more efficient than implementing col2im from scratch

    // Convert to int64_t for LibTorch
    std::vector<int64_t> stride_64(stride.begin(), stride.end());
    std::vector<int64_t> padding_64(padding.begin(), padding.end());
    std::vector<int64_t> output_padding_64(output_padding.begin(), output_padding.end());
    std::vector<int64_t> dilation_64(dilation.begin(), dilation.end());

    // Use PyTorch's built-in conv_transpose2d for efficiency
    torch::Tensor grad_input = torch::nn::functional::conv_transpose2d(
        grad_output,
        weight,
        torch::nn::functional::ConvTranspose2dFuncOptions()
            .stride(stride_64)
            .padding(padding_64)
            .output_padding(output_padding_64)
            .dilation(dilation_64)
    );

    return grad_input;
}

} // namespace NLR