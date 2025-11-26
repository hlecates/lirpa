// MatrixConvolution.h - Matrix-based convolution operations
#ifndef __MATRIX_CONVOLUTION_H__
#define __MATRIX_CONVOLUTION_H__

#include <torch/torch.h>
#include <vector>

namespace NLR {

class MatrixConvolution {
public:
    // Im2col transformation - converts input patches to columns for matrix multiplication
    // Based on auto_LiRPA's matrix mode implementation
    static torch::Tensor im2col(
        const torch::Tensor& input,
        const std::vector<int>& kernel_size,
        const std::vector<int>& stride,
        const std::vector<int>& padding,
        const std::vector<int>& dilation,
        int groups = 1
    );

    // Col2im transformation - reverse of im2col, used in backward pass
    static torch::Tensor col2im(
        const torch::Tensor& col,
        const std::vector<int>& output_size,
        const std::vector<int>& kernel_size,
        const std::vector<int>& stride,
        const std::vector<int>& padding,
        const std::vector<int>& dilation,
        int groups = 1
    );

    // Compute output shape for convolution
    static std::vector<int> computeConvOutputShape(
        const std::vector<int>& input_shape,
        const std::vector<int>& kernel_size,
        const std::vector<int>& stride,
        const std::vector<int>& padding,
        const std::vector<int>& dilation
    );

    // Compute output padding for transpose convolution
    static std::vector<int> computeTransposeOutputPadding(
        const std::vector<int>& input_shape,
        const std::vector<int>& output_shape,
        const std::vector<int>& kernel_size,
        const std::vector<int>& stride,
        const std::vector<int>& padding,
        const std::vector<int>& dilation
    );

    // Matrix multiplication for convolution forward pass
    static torch::Tensor matrixConvForward(
        const torch::Tensor& input_matrix,   // After im2col
        const torch::Tensor& weight,          // Conv weight
        const torch::Tensor& bias,           // Conv bias (optional)
        const std::vector<int>& output_shape  // Expected output shape
    );

    // Matrix multiplication for convolution backward pass (transpose conv)
    static torch::Tensor matrixConvBackward(
        const torch::Tensor& grad_output,     // Gradient from next layer
        const torch::Tensor& weight,          // Conv weight
        const std::vector<int>& input_shape,  // Original input shape
        const std::vector<int>& stride,
        const std::vector<int>& padding,
        const std::vector<int>& dilation,
        const std::vector<int>& output_padding
    );

private:
    // Helper function to unfold input tensor for im2col
    static torch::Tensor unfoldInput(
        const torch::Tensor& input,
        const std::vector<int>& kernel_size,
        const std::vector<int>& stride,
        const std::vector<int>& padding,
        const std::vector<int>& dilation
    );

    // Helper function to fold columns back to spatial dimensions
    static torch::Tensor foldColumns(
        const torch::Tensor& col,
        const std::vector<int>& output_size,
        const std::vector<int>& kernel_size,
        const std::vector<int>& stride,
        const std::vector<int>& padding
    );
};

} // namespace NLR

#endif // __MATRIX_CONVOLUTION_H__