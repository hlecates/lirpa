#include "LirpaMain.h"
#include "input_parsers/OnnxToTorch.h"
#include "TorchModel.h"
#include "LirpaError.h"
#include "configuration/LirpaConfiguration.h"
#include "input_parsers/OutputConstraint.h"
#include "input_parsers/VnnLibInputParser.h"
#include <torch/torch.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>

// Helper function to print bounds
static void printBounds(const torch::Tensor& lower, const torch::Tensor& upper) {
    torch::Tensor lb = lower;
    torch::Tensor ub = upper;
    if (lb.dim() == 0) lb = lb.unsqueeze(0);
    if (ub.dim() == 0) ub = ub.unsqueeze(0);

    std::cout << "Output Bounds:" << std::endl;
    std::cout << std::fixed << std::setprecision(6);

    for (int i = 0; i < lb.size(0); ++i) {
        if (i > 0) std::cout << " ";
        auto l = lb[i];
        auto u = ub[i];
        if (l.dim() > 0) l = l.flatten()[0];
        if (u.dim() > 0) u = u.flatten()[0];
        std::cout << "[" << l.item<double>() << ", " << u.item<double>() << "]";
    }
    std::cout << std::endl;
    std::cout << std::defaultfloat;
}

enum class PropertyStatus {
    Verified,
    Violated,
    Unknown
};

static PropertyStatus evaluatePropertyStatus(
    const NLR::OutputConstraintSet& constraints,
    const torch::Tensor& lowerBounds,
    const torch::Tensor& upperBounds,
    std::string& detail) {
    if (!constraints.hasConstraints()) {
        detail = "no output constraints found in VNN-LIB";
        return PropertyStatus::Unknown;
    }

    if (!lowerBounds.defined() || !upperBounds.defined()) {
        detail = "bounds are undefined";
        return PropertyStatus::Unknown;
    }

    torch::Tensor lb = lowerBounds.flatten();
    torch::Tensor ub = upperBounds.flatten();

    if (lb.numel() == 0 || ub.numel() == 0) {
        detail = "bounds are empty";
        return PropertyStatus::Unknown;
    }

    NLR::CMatrixResult cMatrix = constraints.toCMatrix();
    torch::Tensor thresholds = cMatrix.thresholds.to(ub.device());

    if (lb.numel() != thresholds.numel() || ub.numel() != thresholds.numel()) {
        detail = "bounds/threshold size mismatch";
        return PropertyStatus::Unknown;
    }

    if (cMatrix.hasORBranches) {
        Vector<NLR::BranchResult> branchResults =
            NLR::OutputConstraintSet::evaluateORBranches(lb, ub, thresholds,
                                                         cMatrix.branchMapping,
                                                         cMatrix.branchSizes);
        bool anyVerified = false;
        bool allRefuted = true;

        for (const auto& branch : branchResults) {
            if (branch.verified) {
                anyVerified = true;
            }
            if (!branch.refuted) {
                allRefuted = false;
            }
        }

        if (anyVerified) {
            detail = "at least one OR-branch verified";
            return PropertyStatus::Verified;
        }
        if (allRefuted) {
            detail = "all OR-branches refuted";
            return PropertyStatus::Violated;
        }
        detail = "no OR-branch verified or refuted";
        return PropertyStatus::Unknown;
    }

    torch::Tensor upperDiff = ub - thresholds;
    torch::Tensor lowerDiff = lb - thresholds;
    bool allVerified = (upperDiff <= 0).all().item<bool>();
    bool anyViolated = (lowerDiff > 0).any().item<bool>();

    if (allVerified) {
        detail = "all constraints satisfied by upper bounds";
        return PropertyStatus::Verified;
    }
    if (anyViolated) {
        detail = "some constraint violated by lower bounds";
        return PropertyStatus::Violated;
    }

    detail = "constraints inconclusive";
    return PropertyStatus::Unknown;
}

void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " <onnx_file> <vnnlib_file> [options]" << std::endl;
    std::cout << "   or: " << programName << " --input <onnx_file> --vnnlib <vnnlib_file> [options]" << std::endl;
    std::cout << std::endl;
    std::cout << "Positional arguments:" << std::endl;
    std::cout << "  onnx_file          Path to ONNX model file" << std::endl;
    std::cout << "  vnnlib_file        Path to VNN-LIB property file" << std::endl;
    std::cout << std::endl;
    std::cout << "Flag-based arguments:" << std::endl;
    std::cout << "  --input <path>                  Input ONNX model path" << std::endl;
    std::cout << "  --vnnlib <path>                 VNN-LIB property path (alias for --property)" << std::endl;
    std::cout << "  --property <path>               VNN-LIB property path" << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --method <crown|alpha-crown>    Analysis method (default: crown)" << std::endl;
    std::cout << "  --iterations <n>                Alpha-CROWN iterations (default: 20)" << std::endl;
    std::cout << "  --lr <float>                    Learning rate (default: 0.5)" << std::endl;
    std::cout << "  --lr-decay <float>              LR decay factor (default: 0.98)" << std::endl;
    std::cout << "  --timeout <seconds>             Global timeout (default: 0 = none)" << std::endl;
    std::cout << "  --seed <n>                      Random seed (default: 1)" << std::endl;
    std::cout << "  --verbose / --quiet             Verbosity control" << std::endl;
    std::cout << "  --optimize-lower / --no-optimize-lower" << std::endl;
    std::cout << "  --optimize-upper / --no-optimize-upper" << std::endl;
    std::cout << "  --enable-first-linear-ibp / --disable-first-linear-ibp" << std::endl;
    std::cout << "  --standard-crown / --crown-ibp  CROWN mode selection" << std::endl;
    std::cout << "  --help, -h                      Print this help message" << std::endl;
}

int lirpaMain(int argc, char* argv[]) {
    // Check for help flag
    if (argc > 1) {
        std::string firstArg = argv[1];
        if (firstArg == "--help" || firstArg == "-h") {
            printUsage(argv[0]);
            return 0;
        }
    }

    // Parse arguments - support both positional and flag-based
    std::string onnxFilePath;
    std::string vnnlibFilePath;
    bool useFlags = false;

    // Check if using flag-based arguments (--input, --property, --vnnlib)
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--input" || arg == "--property" || arg == "--vnnlib") {
            useFlags = true;
            break;
        }
    }

    if (useFlags) {
        // Parse flag-based arguments first
        LirpaConfiguration::parseArgs(argc, argv);
        
        // Get file paths from configuration
        if (LirpaConfiguration::INPUT_FILE_PATH.length() == 0) {
            std::cerr << "ERROR: --input flag is required" << std::endl;
            printUsage(argv[0]);
            return 1;
        }
        if (LirpaConfiguration::PROPERTY_FILE_PATH.length() == 0) {
            std::cerr << "ERROR: --property or --vnnlib flag is required" << std::endl;
            printUsage(argv[0]);
            return 1;
        }
        
        onnxFilePath = LirpaConfiguration::INPUT_FILE_PATH.ascii();
        vnnlibFilePath = LirpaConfiguration::PROPERTY_FILE_PATH.ascii();
    } else {
        // Parse positional arguments
        if (argc < 3) {
            printUsage(argv[0]);
            return 1;
        }

        onnxFilePath = argv[1];
        vnnlibFilePath = argv[2];

        // Prepare arguments for LirpaConfiguration::parseArgs
        // We need to skip the first two positional arguments
        std::vector<char*> configArgs;
        configArgs.push_back(argv[0]);  // program name
        for (int i = 3; i < argc; ++i) {
            configArgs.push_back(argv[i]);
        }

        // Parse configuration arguments
        LirpaConfiguration::parseArgs(static_cast<int>(configArgs.size()), configArgs.data());
    }

    try {
        // Force single-threaded execution for numerical determinism
        at::set_num_threads(1);
        at::set_num_interop_threads(1);
        
        if (LirpaConfiguration::VERBOSE) {
            std::cout << "Set threads to 1 for deterministic results" << std::endl;
        }

        if (LirpaConfiguration::VERBOSE) {
            if (torch::cuda::is_available()) {
                std::cout << "CUDA available with "
                          << torch::cuda::device_count() << " devices" << std::endl;
            }
            std::cout << "Using device: "
                      << LirpaConfiguration::getDevice().str() << std::endl;
        }

        std::cout << "Parsing ONNX file: " << onnxFilePath << std::endl;
        std::cout << "Parsing VNN-LIB file: " << vnnlibFilePath << std::endl;

        // Step 1: Create TorchModel from ONNX and VNN-LIB files
        std::shared_ptr<NLR::TorchModel> torchModel = std::make_shared<NLR::TorchModel>(
            String(onnxFilePath.c_str()),
            String(vnnlibFilePath.c_str())
        );

        if (LirpaConfiguration::VERBOSE) {
            std::cout << "TorchModel created successfully" << std::endl;
            std::cout << "Input size: " << torchModel->getInputSize() << std::endl;
            std::cout << "Output size: " << torchModel->getOutputSize() << std::endl;
            std::cout << "Number of nodes: " << torchModel->getNumNodes() << std::endl;
        }

        // Step 2: Verify input bounds were loaded from VNN-LIB file
        if (!torchModel->hasInputBounds()) {
            std::cerr << "ERROR: Input bounds not loaded from VNN-LIB file!" << std::endl;
            return 1;
        }

        BoundedTensor<torch::Tensor> inputBounds = torchModel->getInputBounds();

        if (LirpaConfiguration::VERBOSE) {
            std::cout << "\nInput bounds loaded from VNN-LIB file" << std::endl;
            torch::Tensor lowerBounds = inputBounds.lower();
            torch::Tensor upperBounds = inputBounds.upper();
            std::cout << "  Input bounds: " << lowerBounds.size(0) << " variables bounded in ["
                      << lowerBounds.min().item<double>() << ", "
                      << upperBounds.max().item<double>() << "]" << std::endl;
        }

        // Step 3: Check if specification matrix was loaded from VNN-LIB file
        torch::Tensor* specMatrix = nullptr;
        torch::Tensor C;  // Keep C in wider scope so pointer remains valid

        if (torchModel->hasSpecificationMatrix()) {
            // Get the specification matrix that was automatically parsed from VNN-LIB
            C = torchModel->getSpecificationMatrix();
            specMatrix = &C;

            if (LirpaConfiguration::VERBOSE) {
                std::cout << "\nUsing specification matrix from VNN-LIB file with "
                          << C.size(0) << " constraints" << std::endl;
            }
        } else {
            if (LirpaConfiguration::VERBOSE) {
                std::cout << "\nNo output constraints found in VNN-LIB file, computing raw output bounds" << std::endl;
            }
        }

        // Step 4: Run analysis based on configured method
        BoundedTensor<torch::Tensor> result;

        if (LirpaConfiguration::ANALYSIS_METHOD == LirpaConfiguration::AnalysisMethod::AlphaCROWN) {
            if (LirpaConfiguration::VERBOSE) {
                std::cout << "\nRunning Alpha-CROWN analysis..." << std::endl;
                std::cout << "Configuration:" << std::endl;
                std::cout << "  Method: AlphaCROWN" << std::endl;
                std::cout << "  Iterations: " << LirpaConfiguration::ALPHA_ITERATIONS << std::endl;
                std::cout << "  Optimize lower: " << (LirpaConfiguration::OPTIMIZE_LOWER ? "true" : "false") << std::endl;
                std::cout << "  Optimize upper: " << (LirpaConfiguration::OPTIMIZE_UPPER ? "true" : "false") << std::endl;
            }
            
            result = torchModel->compute_bounds(
                inputBounds,
                specMatrix,  // Specification matrix (or nullptr if no constraints)
                LirpaConfiguration::AnalysisMethod::AlphaCROWN,
                LirpaConfiguration::COMPUTE_LOWER,
                LirpaConfiguration::COMPUTE_UPPER
            );
        } else {
            if (LirpaConfiguration::VERBOSE) {
                std::cout << "\nRunning CROWN analysis..." << std::endl;
            }
            
            result = torchModel->compute_bounds(
                inputBounds,
                specMatrix,  // Specification matrix (or nullptr if no constraints)
                LirpaConfiguration::AnalysisMethod::CROWN,
                LirpaConfiguration::COMPUTE_LOWER,
                LirpaConfiguration::COMPUTE_UPPER
            );
        }

        // Step 5: Output the bounds
        if (result.lower().defined() && result.upper().defined()) {
            printBounds(result.lower(), result.upper());
        } else {
            std::cerr << "ERROR: Bounds are undefined" << std::endl;
            return 1;
        }

        // Step 6: Verify property if constraints exist in VNN-LIB
        PropertyStatus status = PropertyStatus::Unknown;
        std::string statusDetail;
        try {
            NLR::OutputConstraintSet outputConstraints =
                VnnLibInputParser::parseOutputConstraints(
                    String(vnnlibFilePath.c_str()),
                    torchModel->getOutputSize());
            status = evaluatePropertyStatus(outputConstraints,
                                            result.lower(),
                                            result.upper(),
                                            statusDetail);
        } catch (const std::exception& e) {
            statusDetail = std::string("failed to parse output constraints: ") + e.what();
            status = PropertyStatus::Unknown;
        }

        std::string statusLabel;
        switch (status) {
            case PropertyStatus::Verified:
                statusLabel = "VERIFIED";
                break;
            case PropertyStatus::Violated:
                statusLabel = "VIOLATED";
                break;
            case PropertyStatus::Unknown:
            default:
                statusLabel = "UNKNOWN";
                break;
        }

        std::cout << "\nProperty status: " << statusLabel << std::endl;
        if (!statusDetail.empty()) {
            std::cout << "Property details: " << statusDetail << std::endl;
        }

        return 0;

    } catch (const LirpaError& e) {
        std::cerr << "Error: " << e.getErrorClass() << " (code " << e.getCode() << ")" << std::endl;
        if (e.getUserMessage() && strlen(e.getUserMessage()) > 0) {
            std::cerr << "Message: " << e.getUserMessage() << std::endl;
        }
        return 1;
    } catch (const Error& e) {
        std::cerr << "Error: " << e.getErrorClass() << " (code " << e.getCode() << ")" << std::endl;
        if (e.getUserMessage() && strlen(e.getUserMessage()) > 0) {
            std::cerr << "Message: " << e.getUserMessage() << std::endl;
        }
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Error: Unknown exception occurred" << std::endl;
        return 1;
    }
}

