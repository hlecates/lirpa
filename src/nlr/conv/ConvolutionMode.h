// ConvolutionMode.h - Mode selection for convolution operations
#ifndef __CONVOLUTION_MODE_H__
#define __CONVOLUTION_MODE_H__

namespace NLR {

// Convolution computation mode (mirrors auto_LiRPA's conv_mode)
enum class ConvMode {
    MATRIX,    // Matrix mode using im2col transformation (default)
    PATCHES    // Patches mode for more efficient bound propagation (future)
};

} // namespace NLR

#endif // __CONVOLUTION_MODE_H__