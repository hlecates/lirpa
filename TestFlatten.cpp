#include "src/input_parsers/OnnxToTorch.h"
#include "src/nlr/TorchModel.h"
#include "src/nlr/CROWNAnalysis.h"
#include <torch/torch.h>
#include <iostream>
#include <iomanip>
#include <string>

int main(int argc, char* argv[]) {
    (void)argc;
    (void)argv;

    try {
        // Use ACAS network that contains flatten node
        std::string onnxFilePath = "../resources/onnx/ACASXU_run2a_1_1_batch_2000.onnx";
        std::string vnnlibFilePath = "../resources/onnx/vnnlib/prop_1.vnnlib";

        std::cout << "=== Testing Flatten Node in ACAS Network ===" << std::endl;
        std::cout << "Parsing ONNX file: " << onnxFilePath << std::endl;
        std::cout << "Parsing VNN-LIB file: " << vnnlibFilePath << std::endl;
        std::cout << std::endl;

        // Create TorchModel from ONNX and VNN-LIB files
        std::shared_ptr<NLR::TorchModel> torchModel = std::make_shared<NLR::TorchModel>(
            String(onnxFilePath.c_str()),
            String(vnnlibFilePath.c_str())
        );

        std::cout << "\nNetwork loaded successfully" << std::endl;
        std::cout << "Input size: " << torchModel->getInputSize() << std::endl;
        std::cout << "Output size: " << torchModel->getOutputSize() << std::endl;
        std::cout << "Number of nodes: " << torchModel->getNumNodes() << std::endl;

        // Print detailed node structure focusing on flatten area
        std::cout << "\n=== Node Structure Around Flatten ===" << std::endl;
        for (unsigned i = 30; i <= 35 && i < torchModel->getNumNodes(); ++i) {
            auto node = torchModel->getNode(i);
            if (!node) continue;

            std::cout << "Node " << i << ": ";
            switch (node->getNodeType()) {
                case NLR::NodeType::INPUT: std::cout << "INPUT"; break;
                case NLR::NodeType::LINEAR: std::cout << "LINEAR"; break;
                case NLR::NodeType::RELU: std::cout << "RELU"; break;
                case NLR::NodeType::FLATTEN:
                    std::cout << "FLATTEN (*** TARGET NODE ***)";
                    break;
                case NLR::NodeType::SUB: std::cout << "SUB"; break;
                case NLR::NodeType::ADD: std::cout << "ADD"; break;
                default: std::cout << "OTHER";
            }
            std::cout << " (in=" << node->getInputSize()
                      << ", out=" << node->getOutputSize() << ")";

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

        // Test forward pass through flatten node
        std::cout << "\n=== Testing Forward Pass Through Flatten ===" << std::endl;

        // Create a test input
        torch::Tensor test_input = torch::ones({1, 5}); // Batch size 1, 5 features
        test_input[0][0] = 0.64;
        test_input[0][1] = 0.0;
        test_input[0][2] = 0.0;
        test_input[0][3] = 0.475;
        test_input[0][4] = -0.475;

        std::cout << "Test input: " << test_input << std::endl;
        std::cout << "Test input shape: " << test_input.sizes() << std::endl;

        // Run forward pass on sub node (node 31)
        if (torchModel->getNumNodes() > 31) {
            auto subNode = torchModel->getNode(31);
            if (subNode) {
                std::cout << "\nRunning SUB node forward..." << std::endl;
                torch::Tensor sub_output = subNode->forward(test_input);
                std::cout << "SUB output shape: " << sub_output.sizes() << std::endl;
                std::cout << "SUB output: " << sub_output << std::endl;

                // Now test flatten node (node 32)
                if (torchModel->getNumNodes() > 32) {
                    auto flattenNode = torchModel->getNode(32);
                    if (flattenNode) {
                        std::cout << "\nRunning FLATTEN node forward..." << std::endl;
                        torch::Tensor flatten_output = flattenNode->forward(sub_output);
                        std::cout << "FLATTEN output shape: " << flatten_output.sizes() << std::endl;
                        std::cout << "FLATTEN output: " << flatten_output << std::endl;
                    }
                }
            }
        }

        // Test CROWN backward through flatten
        std::cout << "\n=== Testing CROWN Backward Through Flatten ===" << std::endl;
        if (torchModel->hasInputBounds()) {
            BoundedTensor<torch::Tensor> inputBounds = torchModel->getInputBounds();

            std::cout << "Running CROWN analysis (includes backward through flatten)..." << std::endl;
            BoundedTensor<torch::Tensor> crownResult = torchModel->compute_bounds(
                inputBounds,
                nullptr,
                NLR::AnalysisConfig::Method::CROWN,
                true,
                true
            );

            if (crownResult.lower().defined() && crownResult.upper().defined()) {
                std::cout << "\nCROWN bounds computed successfully!" << std::endl;
                std::cout << "Lower bounds: " << crownResult.lower() << std::endl;
                std::cout << "Upper bounds: " << crownResult.upper() << std::endl;
            }
        }

        std::cout << "\n=== Flatten Node Test Complete ===" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}