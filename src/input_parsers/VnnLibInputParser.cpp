#include "VnnLibInputParser.h"
#include "File.h"
#include "InputParserError.h"
#include "MStringf.h"
#include "FloatUtils.h"

#include <boost/regex.hpp>
#include <iostream>
#include <limits>

// Helper function to extract scalar values
double VnnLibInputParser::extractScalar(const String &token)
{
    std::string::size_type end;
    double value = std::stod(token.ascii(), &end);
    if (end != token.length())
    {
        throw InputParserError(InputParserError::UNEXPECTED_INPUT,
                              Stringf("'%s' is not a valid scalar", token.ascii()).ascii());
    }
    return value;
}

// Read VNN-LIB file and strip comments
String VnnLibInputParser::readVnnlibFile(const String &vnnlibFilePath)
{
//     std::cout << "[VnnLibInputParser] Reading file: " << vnnlibFilePath.ascii() << std::endl;

    if (!File::exists(vnnlibFilePath))
    {
        std::cerr << "[VnnLibInputParser] ERROR: File not found!" << std::endl;
        throw InputParserError(InputParserError::FILE_DOESNT_EXIST,
                              Stringf("VNN-LIB file not found: %s", vnnlibFilePath.ascii()).ascii());
    }

    File vnnlibFile(vnnlibFilePath);
    vnnlibFile.open(File::MODE_READ);

    String vnnlibContent;
    int lineCount = 0;

    try
    {
        while (true)
        {
            String line = vnnlibFile.readLine().trim();
            lineCount++;

            // Skip empty lines and comments
            if (line == "" || line.substring(0, 1) == ";")
            {
//                 std::cout << "[VnnLibInputParser] Skipping line " << lineCount
//                          << " (empty or comment)" << std::endl;
                continue;
            }

//             std::cout << "[VnnLibInputParser] Line " << lineCount << ": "
//                      << line.ascii() << std::endl;
            vnnlibContent += line;
        }
    }
    catch (const CommonError &e)
    {
        // READ_FAILED indicates end of file
        if (e.getCode() != CommonError::READ_FAILED)
            throw e;
    }

//     std::cout << "[VnnLibInputParser] Read " << lineCount << " lines total" << std::endl;
//     std::cout << "[VnnLibInputParser] Content length: " << vnnlibContent.length() << " characters" << std::endl;

    return vnnlibContent;
}

// Tokenize VNN-LIB content
Vector<String> VnnLibInputParser::tokenize(const String &vnnlibContent)
{
//     std::cout << "[VnnLibInputParser] Tokenizing content..." << std::endl;

    boost::regex re(R"(\(|\)|[\w\-\\.]+|<=|>=|\+|-|\*)");

    auto tokens_begin = boost::cregex_token_iterator(
        vnnlibContent.ascii(), vnnlibContent.ascii() + vnnlibContent.length(), re);
    auto tokens_end = boost::cregex_token_iterator();

    Vector<String> tokens;
    for (boost::cregex_token_iterator it = tokens_begin; it != tokens_end; ++it)
    {
        boost::csub_match match = *it;
        tokens.append(String(match.str().c_str()));
    }

//     std::cout << "[VnnLibInputParser] Generated " << tokens.size() << " tokens" << std::endl;
//     std::cout << "[VnnLibInputParser] First 20 tokens: ";
    // for (unsigned i = 0; i < tokens.size() && i < 20; ++i)
    // {
    //     std::cout << tokens[i].ascii() << " ";
    // }
    // std::cout << std::endl;

    return tokens;
}

// Check if variable name is an input variable (X_i format)
bool VnnLibInputParser::isInputVariable(const String &varName)
{
    List<String> parts = varName.tokenize("_");
    if (parts.size() != 2)
        return false;

    return parts.front() == "X";
}

// Extract variable index from name (e.g., "X_0" -> 0, "Y_3" -> 3)
int VnnLibInputParser::extractVariableIndex(const String &varName)
{
    List<String> parts = varName.tokenize("_");
    if (parts.size() != 2)
    {
        throw InputParserError(InputParserError::UNEXPECTED_INPUT,
                              Stringf("Invalid variable name format: %s", varName.ascii()).ascii());
    }

    const String &indexStr = parts.back();
    for (unsigned i = 0; i < indexStr.length(); ++i)
    {
        if (!std::isdigit(indexStr[i]))
        {
            throw InputParserError(InputParserError::UNEXPECTED_INPUT,
                                  Stringf("Invalid variable index in: %s", varName.ascii()).ascii());
        }
    }

    return atoi(indexStr.ascii());
}

// Parse tokens and extract input bounds
void VnnLibInputParser::parseTokens(const Vector<String> &tokens,
                                    Vector<InputBoundInfo> &inputBounds)
{
    int index = 0;
    while ((unsigned)index < tokens.size())
    {
        if (tokens[index] == "(")
        {
            index = parseCommand(index + 1, tokens, inputBounds);
            if ((unsigned)index < tokens.size() && tokens[index] == ")")
            {
                ++index;
            }
        }
        else
        {
            ++index;
        }
    }
}

// Parse a command (declare-const or assert)
int VnnLibInputParser::parseCommand(int index,
                                    const Vector<String> &tokens,
                                    Vector<InputBoundInfo> &inputBounds)
{
    if ((unsigned)index >= tokens.size())
        return index;

    const String &command = tokens[index];

    if (command == "declare-const")
    {
        return parseDeclareConst(index + 1, tokens, inputBounds);
    }
    else if (command == "assert")
    {
        return parseAssert(index + 1, tokens, inputBounds);
    }

    // Skip unknown commands - just find matching closing paren
    int depth = 1;
    ++index;
    while ((unsigned)index < tokens.size() && depth > 0)
    {
        if (tokens[index] == "(")
            ++depth;
        else if (tokens[index] == ")")
            --depth;
        ++index;
    }
    return index - 1;
}

// Parse declare-const to identify input variables
int VnnLibInputParser::parseDeclareConst(int index,
                                         const Vector<String> &tokens,
                                         Vector<InputBoundInfo> &inputBounds)
{
    if ((unsigned)index + 1 >= tokens.size())
        return index;

    const String &varName = tokens[index];
    const String &varType = tokens[index + 1];

//     std::cout << "[VnnLibInputParser] declare-const: " << varName.ascii()
//               << " " << varType.ascii() << std::endl;

    if (varType != "Real")
    {
        return index + 2;
    }

    // Only track input variables
    if (isInputVariable(varName))
    {
        int varIndex = extractVariableIndex(varName);
//         std::cout << "[VnnLibInputParser] Found input variable " << varName.ascii()
//                   << " (index " << varIndex << ")" << std::endl;

        // Ensure inputBounds vector is large enough
        while ((unsigned)varIndex >= inputBounds.size())
        {
            inputBounds.append(InputBoundInfo());
        }
    }
    else
    {
//         std::cout << "[VnnLibInputParser] Skipping non-input variable: " << varName.ascii() << std::endl;
    }

    return index + 2;
}

// Parse assert command to extract bounds
int VnnLibInputParser::parseAssert(int index,
                                   const Vector<String> &tokens,
                                   Vector<InputBoundInfo> &inputBounds)
{
    if ((unsigned)index >= tokens.size())
        return index;

    if (tokens[index] == "(")
    {
        return parseCondition(index + 1, tokens, inputBounds);
    }

    return index;
}

// Parse a condition (<= or >= or and)
int VnnLibInputParser::parseCondition(int index,
                                     const Vector<String> &tokens,
                                     Vector<InputBoundInfo> &inputBounds)
{
    if ((unsigned)index >= tokens.size())
        return index;

    const String &op = tokens[index];

    if (op == "and")
    {
        // Parse multiple conditions
        ++index;
        while ((unsigned)index < tokens.size() && tokens[index] != ")")
        {
            if (tokens[index] == "(")
            {
                index = parseCondition(index + 1, tokens, inputBounds);
            }
            else
            {
                ++index;
            }
        }
        return index + 1; // Skip closing )
    }
    else if (op == "<=" || op == ">=")
    {
        // Parse simple bound: (op var value) or (op value var)
        ++index;
        if ((unsigned)index + 1 >= tokens.size())
            return index;

        String token1 = tokens[index];
        String token2 = tokens[index + 1];

        String varName;
        String valueStr;
        bool varFirst = true;

        // Determine which token is variable and which is value
        if (isInputVariable(token1))
        {
            varName = token1;
            valueStr = token2;
            varFirst = true;
        }
        else if (isInputVariable(token2))
        {
            varName = token2;
            valueStr = token1;
            varFirst = false;
        }
        else
        {
            // Neither is an input variable, skip
            index += 2;
            // Find closing paren
            while ((unsigned)index < tokens.size() && tokens[index] != ")")
                ++index;
            return index + 1;
        }

        int varIndex = extractVariableIndex(varName);

        // Ensure inputBounds vector is large enough
        while ((unsigned)varIndex >= inputBounds.size())
        {
            inputBounds.append(InputBoundInfo());
        }

        double value = extractScalar(valueStr);

        // Apply bound based on operator and order
        if (op == "<=")
        {
            if (varFirst)
            {
                // X <= value -> upper bound
                inputBounds[varIndex].hasUpperBound = true;
                inputBounds[varIndex].upperBound = value;
//                 std::cout << "[VnnLibInputParser] Set upper bound: " << varName.ascii()
//                           << " <= " << value << std::endl;
            }
            else
            {
                // value <= X -> lower bound
                inputBounds[varIndex].hasLowerBound = true;
                inputBounds[varIndex].lowerBound = value;
//                 std::cout << "[VnnLibInputParser] Set lower bound: " << value
//                           << " <= " << varName.ascii() << std::endl;
            }
        }
        else // op == ">="
        {
            if (varFirst)
            {
                // X >= value -> lower bound
                inputBounds[varIndex].hasLowerBound = true;
                inputBounds[varIndex].lowerBound = value;
//                 std::cout << "[VnnLibInputParser] Set lower bound: " << varName.ascii()
//                           << " >= " << value << std::endl;
            }
            else
            {
                // value >= X -> upper bound
                inputBounds[varIndex].hasUpperBound = true;
                inputBounds[varIndex].upperBound = value;
//                 std::cout << "[VnnLibInputParser] Set upper bound: " << value
//                           << " >= " << varName.ascii() << std::endl;
            }
        }

        index += 2;
        // Find closing paren
        while ((unsigned)index < tokens.size() && tokens[index] != ")")
            ++index;
        return index + 1;
    }

    // Unknown operator, skip to closing paren
    while ((unsigned)index < tokens.size() && tokens[index] != ")")
        ++index;
    return index + 1;
}

// Main parsing entry point
BoundedTensor<torch::Tensor> VnnLibInputParser::parseInputBounds(const String &vnnlibFilePath,
                                                                  unsigned expectedInputSize)
{
//     std::cout << "\n[VnnLibInputParser] ========== Starting VNN-LIB Parsing ==========" << std::endl;
//     std::cout << "[VnnLibInputParser] Expected input size: " << expectedInputSize << std::endl;

    // Read and tokenize the file
    String content = readVnnlibFile(vnnlibFilePath);
    Vector<String> tokens = tokenize(content);

//     std::cout << "[VnnLibInputParser] Parsing tokens to extract bounds..." << std::endl;

    // Parse tokens to extract bounds
    Vector<InputBoundInfo> inputBounds;
    parseTokens(tokens, inputBounds);

//     std::cout << "[VnnLibInputParser] Found " << inputBounds.size() << " input variables" << std::endl;

    // Ensure we have at least expectedInputSize entries
    while (inputBounds.size() < expectedInputSize)
    {
        inputBounds.append(InputBoundInfo());
    }

    // Warn if we found more inputs than expected
    if (inputBounds.size() > expectedInputSize)
    {
//         std::cout << "[VnnLibInputParser] WARNING: VNN-LIB file specifies " << inputBounds.size()
//                   << " input variables, but model expects " << expectedInputSize << std::endl;
    }

//     std::cout << "\n[VnnLibInputParser] Summary of extracted bounds:" << std::endl;
    // Debug loop - commented out to reduce output
    // for (unsigned i = 0; i < expectedInputSize; ++i)
    // {
    //     if (i < inputBounds.size())
    //     {
    //         std::cout << "[VnnLibInputParser]   X_" << i << ": ["
    //                   << inputBounds[i].lowerBound << ", "
    //                   << inputBounds[i].upperBound << "]";
    //         if (!inputBounds[i].hasLowerBound)
    //             std::cout << " (lower=default)";
    //         if (!inputBounds[i].hasUpperBound)
    //             std::cout << " (upper=default)";
    //         std::cout << std::endl;
    //     }
    // }

    // Convert to torch tensors - use double precision first for accurate parsing
    torch::Tensor lowerBounds = torch::zeros({(long)expectedInputSize}, torch::kFloat64);
    torch::Tensor upperBounds = torch::zeros({(long)expectedInputSize}, torch::kFloat64);

    for (unsigned i = 0; i < expectedInputSize; ++i)
    {
        if (i < inputBounds.size())
        {
            lowerBounds[i] = inputBounds[i].lowerBound;
            upperBounds[i] = inputBounds[i].upperBound;
        }
        else
        {
            // No bounds specified, use infinity
            lowerBounds[i] = -std::numeric_limits<double>::infinity();
            upperBounds[i] = std::numeric_limits<double>::infinity();
        }
    }

    // Now convert to float32 to match Python's precision
    // This ensures the conversion happens in the same way as Python
    lowerBounds = lowerBounds.to(torch::kFloat32);
    upperBounds = upperBounds.to(torch::kFloat32);

//     std::cout << "[VnnLibInputParser] ========== VNN-LIB Parsing Complete ==========" << std::endl << std::endl;

    return BoundedTensor<torch::Tensor>(lowerBounds, upperBounds);
}
