#include "src/input_parsers/OnnxToTorch.h"
#include "src/nlr/TorchModel.h"
#include "src/nlr/AlphaCROWNAnalysis.h"
#include <torch/torch.h>
#include <iostream>
#include <iomanip>
#include <string>

// Helper function to print bounds with optional rounding
static void printBounds(const torch::Tensor& lower, const torch::Tensor& upper, bool round = true) {
    torch::Tensor lb = lower;
    torch::Tensor ub = upper;
    if (lb.dim() == 0) lb = lb.unsqueeze(0);
    if (ub.dim() == 0) ub = ub.unsqueeze(0);

    std::cout << "Output Bounds:" << std::endl;

    if (round) {
        // Print with 6 decimal places when rounding
        std::cout << std::fixed << std::setprecision(6);
    } else {
        // Print with full precision (typically 15 significant digits for double)
        std::cout << std::scientific << std::setprecision(15);
    }

    for (int i = 0; i < lb.size(0); ++i) {
        if (i > 0) std::cout << " ";
        auto l = lb[i];
        auto u = ub[i];
        if (l.dim() > 0) l = l.flatten()[0];
        if (u.dim() > 0) u = u.flatten()[0];
        // Use double for consistency with Python's float32 precision handling
        std::cout << "[" << l.item<double>() << ", " << u.item<double>() << "]";
    }
    std::cout << std::endl;
    std::cout << std::defaultfloat;
}

int main(int argc, char* argv[]) {
    (void)argc;
    (void)argv;

    try {
        // Force single-threaded execution for numerical determinism
        at::set_num_threads(1);
        at::set_num_interop_threads(1);
        std::cout << "Set threads to 1 for deterministic results\n\n";
        // Configuration flags
        bool round = true;  // Set to false for full precision, true for 6 decimal places

        
        //std::string onnxFilePath = "../resources/onnx/deeper_network.onnx";
        std::string onnxFilePath = "../resources/onnx/ACASXU_run2a_1_1_batch_2000.onnx";
        //std::string onnxFilePath = "../resources/onnx/mnist-point.onnx";
        //std::string onnxFilePath = "../resources/onnx/sat_v6_c27.onnx";
        //std::string onnxFilePath = "../resources/onnx/layer-zoo/add.onnx";
        //std::string onnxFilePath = "../resources/onnx/toy_add_complex.onnx";
        //std::string onnxFilePath = "../resources/onnx/toy_matmul_add_chain.onnx";
        //std::string onnxFilePath = "../resources/onnx/toy_sub_chain.onnx";


        
        //std::string vnnlibFilePath = "../resources/properties/deeper_network_test.vnnlib";
        std::string vnnlibFilePath = "../resources/onnx/vnnlib/prop_1.vnnlib";
        //std::string vnnlibFilePath = "../resources/onnx/vnnlib/mnist-img10.vnnlib";
        //std::string vnnlibFilePath = "../resources/onnx/vnnlib/sat_v6_c27.vnnlib";
        //std::string vnnlibFilePath = "../resources/onnx/vnnlib/add_input.vnnlib";


        unsigned iterations = 20; 

        std::cout << "Parsing ONNX file: " << onnxFilePath << std::endl;
        std::cout << "Parsing VNN-LIB file: " << vnnlibFilePath << std::endl;

        // Step 1: Create TorchModel from ONNX and VNN-LIB files
        // Note: Variable mapping no longer needed for LIRPA pipeline
        std::shared_ptr<NLR::TorchModel> torchModel = std::make_shared<NLR::TorchModel>(String(onnxFilePath.c_str()),
                                             String(vnnlibFilePath.c_str()));

        std::cout << "TorchModel created successfully" << std::endl;
        std::cout << "Input size: " << torchModel->getInputSize() << std::endl;
        std::cout << "Output size: " << torchModel->getOutputSize() << std::endl;
        std::cout << "Number of nodes: " << torchModel->getNumNodes() << std::endl;

        // Print network structure
        std::cout << "\nNetwork Structure" << std::endl;
        for (unsigned i = 0; i < torchModel->getNumNodes(); ++i) {
            auto node = torchModel->getNode(i);
            if (!node) continue;

            std::cout << "Node " << i << ": ";
            switch (node->getNodeType()) {
                case NLR::NodeType::INPUT: std::cout << "INPUT"; break;
                case NLR::NodeType::LINEAR: std::cout << "LINEAR"; break;
                case NLR::NodeType::RELU: std::cout << "RELU"; break;
                case NLR::NodeType::CONSTANT: std::cout << "CONSTANT"; break;
                case NLR::NodeType::IDENTITY: std::cout << "IDENTITY"; break;
                case NLR::NodeType::RESHAPE: std::cout << "RESHAPE"; break;
                case NLR::NodeType::FLATTEN: std::cout << "FLATTEN"; break;
                case NLR::NodeType::SUB: std::cout << "SUB"; break;
                case NLR::NodeType::ADD: std::cout << "ADD"; break;
                default: std::cout << "UNKNOWN";
            }
            std::cout << " (in=" << node->getInputSize()
                      << ", out=" << node->getOutputSize() << ")";

            // Print dependencies
            auto deps = torchModel->getDependencies(i);
            if (!deps.empty()) {
                std::cout << " <- deps: [";
                for (size_t j = 0; j < deps.size(); ++j) {
                    std::cout << deps[j];
                    if (j < deps.size() - 1) std::cout << ", ";
                }
                std::cout << "]";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;

        // Step 2: Verify input bounds were loaded from VNN-LIB file
        if (!torchModel->hasInputBounds()) {
            std::cerr << "ERROR: Input bounds not loaded from VNN-LIB file!" << std::endl;
            return 1;
        }

        BoundedTensor<torch::Tensor> inputBounds = torchModel->getInputBounds();
        std::cout << "\nInput bounds loaded from VNN-LIB file:" << std::endl;
        torch::Tensor lowerBounds = inputBounds.lower();
        torch::Tensor upperBounds = inputBounds.upper();

        // Verify and convert to float32 to match auto_LiRPA
        std::cout << "Input bounds dtype: lower=" << lowerBounds.dtype()
                  << ", upper=" << upperBounds.dtype() << std::endl;

        if (lowerBounds.dtype() != torch::kFloat32) {
            lowerBounds = lowerBounds.to(torch::kFloat32);
            upperBounds = upperBounds.to(torch::kFloat32);
            inputBounds = BoundedTensor<torch::Tensor>(lowerBounds, upperBounds);
            std::cout << "Converted input bounds to float32" << std::endl;
        }

        // Print input bounds with high precision to verify exact values
        std::cout << std::fixed << std::setprecision(9);
        for (int i = 0; i < lowerBounds.size(0); ++i) {
            std::cout << "  X_" << i << ": [" << lowerBounds[i].item<double>()
                      << ", " << upperBounds[i].item<double>() << "]" << std::endl;
        }
        std::cout << std::defaultfloat;

        // Step 3: Run plain CROWN first
        std::cout << "\nRunning plain CROWN analysis..." << std::endl;
        BoundedTensor<torch::Tensor> crownResult = torchModel->compute_bounds(
            inputBounds,
            nullptr,  // No specification matrix
            NLR::AnalysisConfig::Method::CROWN,
            true,   // compute lower bounds
            true    // compute upper bounds
        );

        std::cout << "\n=== CROWN Results ===" << std::endl;
        if (crownResult.lower().defined() && crownResult.upper().defined()) {
            std::cout << "Output dtype: lower=" << crownResult.lower().dtype()
                      << ", upper=" << crownResult.upper().dtype() << std::endl;

            // First print with 6 decimal places to match auto_LiRPA
            printBounds(crownResult.lower(), crownResult.upper(), round);

            // Also print with full precision for debugging
            if (round) {
                std::cout << "Full precision CROWN bounds:" << std::endl;
                printBounds(crownResult.lower(), crownResult.upper(), false);
            }
        } else {
            std::cout << "Bounds are undefined" << std::endl;
        }

        // Step 4: Configure analysis for Alpha-CROWN
        NLR::AnalysisConfig config;
        config.method = NLR::AnalysisConfig::Method::AlphaCROWN;
        config.alpha_iterations = iterations;
        config.alpha_lr = 0.5;
        config.optimize_lower = true;
        config.optimize_upper = true;
        config.verbose = true;
        torchModel->setAnalysisConfig(config);

        std::cout << "\nRunning Alpha-CROWN analysis (via compute_bounds)..." << std::endl;
        std::cout << "Configuration:" << std::endl;
        std::cout << "  Method: AlphaCROWN" << std::endl;
        std::cout << "  Iterations: " << iterations << std::endl;
        std::cout << "  Optimizing both lower and upper bounds" << std::endl;

        // Step 5: Run analysis using unified compute_bounds() method
        BoundedTensor<torch::Tensor> result = torchModel->compute_bounds(
            inputBounds,
            nullptr,  // No specification matrix
            NLR::AnalysisConfig::Method::AlphaCROWN,
            true,   // compute lower bounds
            true    // compute upper bounds
        );

        // Step 6: Output the bounds
        std::cout << "\n=== Alpha-CROWN Results ===" << std::endl;
        if (result.lower().defined() && result.upper().defined()) {
            printBounds(result.lower(), result.upper(), round);
        } else {
            std::cout << "Bounds are undefined" << std::endl;
        }

        std::cout << "\nVerification completed successfully!" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
