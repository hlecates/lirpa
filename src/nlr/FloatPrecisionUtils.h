#ifndef __FloatPrecisionUtils_h__
#define __FloatPrecisionUtils_h__

#include <torch/torch.h>

namespace NLR {

// Utility functions to ensure numerical consistency with auto_LiRPA
class FloatPrecisionUtils {
public:
    // Ensure tensor is float32 and contiguous
    static torch::Tensor ensureFloat32(const torch::Tensor& t) {
        if (t.dtype() != torch::kFloat32) {
            return t.to(torch::kFloat32).contiguous();
        }
        if (!t.is_contiguous()) {
            return t.contiguous();
        }
        return t;
    }

    // Ensure all operations match auto_LiRPA's numerical behavior
    static torch::Tensor stableMatmul(const torch::Tensor& a, const torch::Tensor& b) {
        // Ensure both inputs are float32
        torch::Tensor a_f32 = ensureFloat32(a);
        torch::Tensor b_f32 = ensureFloat32(b);

        // Perform matmul with float32 precision
        torch::Tensor result = torch::matmul(a_f32, b_f32);

        // Ensure result is also float32 (should be automatic, but be explicit)
        return ensureFloat32(result);
    }

    // Stable division with epsilon matching auto_LiRPA
    static torch::Tensor stableDivision(const torch::Tensor& numerator,
                                         const torch::Tensor& denominator,
                                         double epsilon = 1e-8) {
        torch::Tensor num_f32 = ensureFloat32(numerator);
        torch::Tensor denom_f32 = ensureFloat32(denominator);

        // Add epsilon to denominator to avoid division by zero
        // This matches auto_LiRPA's approach
        torch::Tensor safe_denom = denom_f32.clamp_min(epsilon);

        return num_f32 / safe_denom;
    }

    // Ensure bounds are computed with consistent precision
    static std::pair<torch::Tensor, torch::Tensor> ensureFloat32Bounds(
        const torch::Tensor& lower, const torch::Tensor& upper) {
        return {ensureFloat32(lower), ensureFloat32(upper)};
    }
};

} // namespace NLR

#endif // __FloatPrecisionUtils_h__