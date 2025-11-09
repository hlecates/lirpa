#ifndef __VnnLibInputParser_h__
#define __VnnLibInputParser_h__

#include "MString.h"
#include "Vector.h"
#include "BoundedTensor.h"

// Undefine macros to avoid conflicts with PyTorch
#ifdef Warning
#undef Warning
#endif
#ifdef LOG
#undef LOG
#endif

#include <torch/torch.h>

class VnnLibInputParser
{
public:
    /**
     * Parse input bounds from a VNN-LIB file.
     *
     * @param vnnlibFilePath Path to the .vnnlib file
     * @param expectedInputSize Expected number of input variables (for validation)
     * @return BoundedTensor containing lower and upper bounds for each input
     *
     * The parser extracts only input constraints (X_i variables) and ignores
     * output constraints (Y_i variables). It handles:
     * - (declare-const X_i Real)
     * - (assert (>= X_i value)) -> sets lower bound
     * - (assert (<= X_i value)) -> sets upper bound
     * - (assert (and ...)) -> multiple constraints
     *
     * Inputs without explicit bounds default to [-inf, +inf].
     */
    static BoundedTensor<torch::Tensor> parseInputBounds(const String &vnnlibFilePath,
                                                          unsigned expectedInputSize);

private:
    /**
     * Internal structure for tracking input bounds during parsing
     */
    struct InputBoundInfo {
        bool hasLowerBound;
        bool hasUpperBound;
        double lowerBound;  // Keep as double for accurate parsing
        double upperBound;  // Keep as double for accurate parsing

        InputBoundInfo()
            : hasLowerBound(false), hasUpperBound(false),
              lowerBound(-std::numeric_limits<double>::infinity()),
              upperBound(std::numeric_limits<double>::infinity()) {}
    };

    /**
     * Read and clean the VNN-LIB file contents
     */
    static String readVnnlibFile(const String &vnnlibFilePath);

    /**
     * Tokenize the VNN-LIB content
     */
    static Vector<String> tokenize(const String &vnnlibContent);

    /**
     * Parse the token stream and extract input bounds
     */
    static void parseTokens(const Vector<String> &tokens,
                           Vector<InputBoundInfo> &inputBounds);

    /**
     * Parse a single command (declare-const or assert)
     */
    static int parseCommand(int index,
                           const Vector<String> &tokens,
                           Vector<InputBoundInfo> &inputBounds);

    /**
     * Parse a declare-const command to identify input variables
     */
    static int parseDeclareConst(int index,
                                 const Vector<String> &tokens,
                                 Vector<InputBoundInfo> &inputBounds);

    /**
     * Parse an assert command to extract input bounds
     */
    static int parseAssert(int index,
                          const Vector<String> &tokens,
                          Vector<InputBoundInfo> &inputBounds);

    /**
     * Parse a condition (<=, >=, and)
     */
    static int parseCondition(int index,
                             const Vector<String> &tokens,
                             Vector<InputBoundInfo> &inputBounds);

    /**
     * Extract variable index from name (e.g., "X_0" -> 0)
     */
    static int extractVariableIndex(const String &varName);

    /**
     * Check if a variable name represents an input (X_i)
     */
    static bool isInputVariable(const String &varName);

    /**
     * Extract scalar value from string
     */
    static double extractScalar(const String &token);
};

#endif // __VnnLibInputParser_h__
