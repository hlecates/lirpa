#include "CROWNAnalysis.h"
#include "BoundedConstantNode.h"

#include "Debug.h"
#include "FloatUtils.h"
#include "MStringf.h"
#include "LirpaError.h"
#include "TimeUtils.h"

#include <fstream>
#include <vector>


namespace NLR {

// Helper function to write debug info to log file
static void writeToDebugLog(const std::string& message) {
    std::ofstream logFile("crown_debug_cpp.log", std::ios::app);
    if (logFile.is_open()) {
        logFile << message;
        logFile.close();
    }
}

// Helper function to get first N elements of a tensor as a string
static std::string tensorFirstN(const torch::Tensor& tensor, int n = 10) {
    if (!tensor.defined() || tensor.numel() == 0) {
        return "[]";
    }
    
    auto flat = tensor.flatten();
    int count = std::min(n, (int)flat.numel());
    
    std::ostringstream oss;
    oss << "[";
    for (int i = 0; i < count; ++i) {
        if (i > 0) oss << ", ";
        oss << std::fixed << std::setprecision(6) << flat[i].item<float>();
    }
    if (flat.numel() > count) {
        oss << ", ...";
    }
    oss << "]";
    return oss.str();
}


CROWNAnalysis::CROWNAnalysis( TorchModel *torchModel, bool useStandardCROWN )
    : _torchModel( torchModel )
{
    _useStandardCROWN = useStandardCROWN;
    _enableFirstLinearIBP = true; // Default to enabled 
    // Get all nodes from the torch model
    const Vector<std::shared_ptr<BoundedTorchNode>>& nodes = _torchModel->getNodes();
    
    // Initialize nodes map -> map network node indices to bounded nodes
    for ( unsigned i = 0; i < nodes.size(); ++i ) 
    {
        _nodes[i] = nodes[i];
    }
}


CROWNAnalysis::~CROWNAnalysis()
{

}


void CROWNAnalysis::run()
{
    log("run() - Starting");
    try {
        if (_useStandardCROWN) {
            // Standard CROWN: compute intermediate bounds via CROWN layer-by-layer
            Vector<unsigned> forwardOrder = _torchModel->topologicalSort();
            for (unsigned nodeIndex : forwardOrder) {
                if (_nodes[nodeIndex]->getNodeType() == NodeType::INPUT) continue;

                // Set current start context for alpha-CROWN
                auto& startNode = _nodes[nodeIndex];
                unsigned startSize = startNode->getOutputSize();
                std::string startKey = "/" + std::to_string(nodeIndex);
                _setCurrentStart(startKey, startSize);

                backwardFrom(nodeIndex);
                concretizeNode(nodeIndex);
            }
        } else {
            // CROWN-IBP: IBP for intermediates, CROWN for final
            computeIBPBounds();

            // Set current start context for the output node
            unsigned outputIndex = getOutputIndex();
            auto& outputNode = _nodes[outputIndex];
            unsigned outputSize = outputNode->getOutputSize();
            std::string startKey = "/" + std::to_string(outputIndex);
            _setCurrentStart(startKey, outputSize);

            backwardFrom(outputIndex);
            concretizeNode(outputIndex);
        }
    } catch (const std::exception& e) {
        log(Stringf("run() - Exception caught: %s", e.what()));
        throw;
    }
    log("run() - Completed");
}

void CROWNAnalysis::computeIBPBounds()
{
    resetProcessingState();
    Vector<unsigned> forwardOrder = _torchModel->topologicalSort();

    log(Stringf("computeIBPBounds() - Processing %u nodes", forwardOrder.size()));

    for (unsigned nodeIndex : forwardOrder) {

        if (isProcessed(nodeIndex)) continue;
        markProcessed(nodeIndex);

        auto& node = _nodes[nodeIndex];

        log(Stringf("computeIBPBounds() - Node %u (type=%s)", nodeIndex, nodeTypeToString(node->getNodeType()).c_str()));

        // Get input bounds for this node
        Vector<BoundedTensor<torch::Tensor>> inputBounds = getInputBoundsForNode(nodeIndex);

        log(Stringf("computeIBPBounds() - Node %u has %u input bounds", nodeIndex, inputBounds.size()));

        // For the input node, if explicit input bounds exist, store and continue
        if (node->getNodeType() == NodeType::INPUT) {
            if (_torchModel->hasInputBounds()) {
                torch::Tensor inputLower = _torchModel->getInputLowerBounds();
                torch::Tensor inputUpper = _torchModel->getInputUpperBounds();
                _ibpBounds[nodeIndex] = BoundedTensor<torch::Tensor>(inputLower, inputUpper);
                log(Stringf("computeIBPBounds() - Node %u (INPUT) stored with bounds shape [%lld]",
                    nodeIndex, (long long)inputLower.size(0)));
                continue;
            }
        }

        // Compute IBP bounds (same computation for all node types)
        BoundedTensor<torch::Tensor> ibpBounds = node->computeIntervalBoundPropagation(inputBounds);

        // Store IBP bounds
        _ibpBounds[nodeIndex] = ibpBounds;

        if (ibpBounds.lower().defined() && ibpBounds.upper().defined()) {
            log(Stringf("computeIBPBounds() - Node %u stored IBP bounds: lower shape [%lld], upper shape [%lld]",
                nodeIndex, (long long)ibpBounds.lower().size(0), (long long)ibpBounds.upper().size(0)));
        } else {
            log(Stringf("computeIBPBounds() - Node %u WARNING: IBP bounds are undefined!", nodeIndex));
        }
    }

    log(Stringf("computeIBPBounds() - Completed, stored bounds for %u nodes", _ibpBounds.size()));
}

void CROWNAnalysis::backwardFrom(unsigned startIndex)
{
    if ( !_nodes.exists(startIndex) ) {
        log(Stringf("backwardFrom() - Warning: start index %u not found in nodes.", startIndex));
        return;
    }

    // Fresh state for this run
    clearCrownState();

    // Initialize with identity matrices for the start node
    auto& startNode = _nodes[startIndex];
    unsigned startSize = startNode->getOutputSize();

    torch::Tensor identityMatrix = preprocessC(torch::Tensor(), startSize);
    _lA[startIndex] = BoundA(identityMatrix);
    _uA[startIndex] = BoundA(identityMatrix);

    _lowerBias[startIndex] = torch::zeros({startSize}, torch::kFloat32);
    _upperBias[startIndex] = torch::zeros({startSize}, torch::kFloat32);

    resetProcessingState();

    log(Stringf("backwardFrom() - Starting queue processing from node %u", startIndex));

    Queue<unsigned> queue;
    queue.push(startIndex);

    while (!queue.empty())
    {
        unsigned current = queue.peak();
        queue.pop();

        if ( isProcessed(current) ) continue;
        markProcessed(current);

        auto& node = _nodes[current];
        NodeType nodetype = node->getNodeType();

        if (!_lA.exists(current) && !_uA.exists(current)) {
            log(Stringf("No A matrices for node %u, skipping", current));
            continue;
        }

        // Bounds for current's inputs: IBP or CROWN-concrete depending on mode and availability
        Vector<BoundedTensor<torch::Tensor>> inputBounds = getInputBoundsForNode(current);

        BoundA currentLowerAlpha = _lA.exists(current) ? _lA[current] : BoundA();
        BoundA currentUpperAlpha = _uA.exists(current) ? _uA[current] : BoundA();

        Vector<Pair<BoundA, BoundA>> A_matrices;
        torch::Tensor lbias, ubias;
        
        // Check if we can skip full CROWN computation for first linear layer
        if (checkIBPFirstLinear(current) && !_useStandardCROWN) {
            // For first linear layers connected directly to inputs, IBP bounds are sufficient
            // and we can avoid the expensive CROWN backward computation and C matrix construction
            log(Stringf("backwardFrom() -  Skipping CROWN backward for first linear layer %u (using IBP)", current));
            continue;
        }
        
        // Skip CROWN backward propagation for non-perturbed nodes (constants, weights, biases)
        // Following auto_LiRPA's approach in backward_bound.py:261-265
        if (!node->isPerturbed()) {
            // For constant nodes, their contribution is added directly to the bias term
            // No need to propagate A matrices through them
            log(Stringf("backwardFrom() - Skipping non-perturbed node %u (%s)", 
                       current, node->getNodeName().ascii()));
            
            // TODO: Implement add_constant_node equivalent if needed
            // For now, just skip the backward propagation
            continue;
        }
        
        node->boundBackward(currentLowerAlpha, currentUpperAlpha, inputBounds, A_matrices, lbias, ubias);

        // CROWN DEBUG: Layer-wise bound information (to file)
        {
            std::ostringstream oss;
            oss << "\n============================================================\n";
            oss << "[CROWN LAYER] Node " << current << " (" << node->getNodeName().ascii() << ")\n";
            oss << "  Output size: " << node->getOutputSize() << "\n";
            
            if (_concreteBounds.exists(current)) {
                auto bounds = _concreteBounds[current];
                auto lower = bounds.lower();
                auto upper = bounds.upper();
                
                oss << "  Layer lower bounds (first 10): " << tensorFirstN(lower) << "\n";
                oss << "  Layer lower stats: min=" << std::fixed << std::setprecision(6)
                    << lower.min().item<float>() << ", max=" << lower.max().item<float>()
                    << ", mean=" << lower.mean().item<float>() << "\n";
                    
                oss << "  Layer upper bounds (first 10): " << tensorFirstN(upper) << "\n";
                oss << "  Layer upper stats: min=" << std::fixed << std::setprecision(6)
                    << upper.min().item<float>() << ", max=" << upper.max().item<float>()
                    << ", mean=" << upper.mean().item<float>() << "\n";
            }
            
            if (lbias.defined() && lbias.numel() > 0) {
                oss << "  Bias lower (first 10): " << tensorFirstN(lbias) << "\n";
                oss << "  Bias lower stats: mean=" << std::fixed << std::setprecision(6)
                    << lbias.mean().item<float>() << ", shape=[";
                auto shape = lbias.sizes();
                for (size_t i = 0; i < shape.size(); ++i) {
                    if (i > 0) oss << ", ";
                    oss << shape[i];
                }
                oss << "]\n";
            }
            
            if (ubias.defined() && ubias.numel() > 0) {
                oss << "  Bias upper (first 10): " << tensorFirstN(ubias) << "\n";
                oss << "  Bias upper stats: mean=" << std::fixed << std::setprecision(6)
                    << ubias.mean().item<float>() << ", shape=[";
                auto shape = ubias.sizes();
                for (size_t i = 0; i < shape.size(); ++i) {
                    if (i > 0) oss << ", ";
                    oss << shape[i];
                }
                oss << "]\n";
            }
            
            // A matrix logging omitted for simplicity with BoundA
            
            writeToDebugLog(oss.str());
        }

        if (_torchModel->getDependenciesMap().exists(current))
        {
            for (unsigned i = 0; i < _torchModel->getDependencies(current).size() && i < A_matrices.size(); ++i)
            {
                unsigned inputIndex = _torchModel->getDependencies(current)[i];

                BoundA new_lA = A_matrices[i].first();
                BoundA new_uA = A_matrices[i].second();

                addBound(inputIndex, new_lA, new_uA);

                torch::Tensor propagated_lbias = lbias.defined() ? lbias.clone() : torch::zeros_like(_lowerBias[current]);
                torch::Tensor propagated_ubias = ubias.defined() ? ubias.clone() : torch::zeros_like(_upperBias[current]);

                if (_lowerBias.exists(current)) propagated_lbias = propagated_lbias + _lowerBias[current];
                if (_upperBias.exists(current)) propagated_ubias = propagated_ubias + _upperBias[current];

                addBias(inputIndex, propagated_lbias, propagated_ubias);

                queue.push(inputIndex);
            }
        }

        if (nodetype != NodeType::INPUT) {
            _lA.erase(current);
            _uA.erase(current);
            _lowerBias.erase(current);
            _upperBias.erase(current);
        }
    }
    log("backwardFrom() - Completed.");
}

void CROWNAnalysis::clearCrownState()
{
    _lA.clear();
    _uA.clear();
    _lowerBias.clear();
    _upperBias.clear();
}

void CROWNAnalysis::concretizeNode(unsigned startIndex)
{
    log(Stringf("concretizeNode() - Starting concretization for node %u", startIndex));

    // Determine input node index
    int inputIndex = -1;
    for (const auto &p : _nodes) {
        if (p.second->getNodeType() == NodeType::INPUT) {
            inputIndex = static_cast<int>(p.first);
            break;
        }
    }

    log(Stringf("concretizeNode() - Input node index: %d", inputIndex));

    if (inputIndex < 0) {
        log(Stringf("concretizeNode() - No input node found, using IBP bounds for node %u", startIndex));
        if (_ibpBounds.exists(startIndex)) {
            _concreteBounds[startIndex] = _ibpBounds[startIndex];
            _torchModel->setConcreteBounds(startIndex, _ibpBounds[startIndex]);
            log(Stringf("concretizeNode() - Stored IBP bounds as concrete bounds for node %u", startIndex));
        } else {
            log(Stringf("concretizeNode() - WARNING: No IBP bounds available for node %u", startIndex));
        }
        return;
    }

    // Get input bounds
    torch::Tensor inputLower, inputUpper;
    if (_torchModel->hasInputBounds()) {
        inputLower = _torchModel->getInputLowerBounds();
        inputUpper = _torchModel->getInputUpperBounds();
    } else {
        unsigned inputSize = _torchModel->getInputSize();
        inputLower = torch::zeros({(long)inputSize}, torch::kFloat32);
        inputUpper = torch::ones({(long)inputSize}, torch::kFloat32);
    }
    inputLower = inputLower.to(torch::kFloat32);
    inputUpper = inputUpper.to(torch::kFloat32);

    log(Stringf("concretizeNode() - Checking for A matrices at input node %d", inputIndex));
    log(Stringf("concretizeNode() - _lA.exists(%d)=%d, _uA.exists(%d)=%d",
        inputIndex, _lA.exists(inputIndex), inputIndex, _uA.exists(inputIndex)));

    // Retrieve A and bias at input node (w.r.t. this start node)
    if (!_lA.exists(inputIndex) && !_uA.exists(inputIndex)) {
        log(Stringf("concretizeNode() - No A matrices at input node, using IBP bounds for node %u", startIndex));
        if (_ibpBounds.exists(startIndex)) {
            _concreteBounds[startIndex] = _ibpBounds[startIndex];
            _torchModel->setConcreteBounds(startIndex, _ibpBounds[startIndex]);
            log(Stringf("concretizeNode() - Stored IBP bounds as concrete bounds for node %u", startIndex));
        } else {
            log(Stringf("concretizeNode() - WARNING: No IBP bounds available for node %u", startIndex));
        }
        return;
    }
    
    BoundA lA_bound = _lA.exists(inputIndex) ? _lA[inputIndex] : BoundA();
    BoundA uA_bound = _uA.exists(inputIndex) ? _uA[inputIndex] : BoundA();
    
    torch::Tensor lA, uA;
    
    if (lA_bound.isPatches()) {
        auto p = lA_bound.asPatches();
        if (!p->input_shape.empty()) {
            lA = p->to_matrix(p->input_shape);
        } else {
            // Fallback to using known input shape?
            std::vector<int64_t> in_shape;
            // Assume 4D input [1, C, H, W] if we can infer?
            // Or assume already matrix?
            // Patches usually have input_shape.
            throw std::runtime_error("CROWNAnalysis::concretizeNode - Patches without input_shape, cannot convert to matrix");
        }
    } else {
        lA = lA_bound.asTensor();
    }
    
    if (uA_bound.isPatches()) {
        auto p = uA_bound.asPatches();
        if (!p->input_shape.empty()) {
            uA = p->to_matrix(p->input_shape);
        } else {
            throw std::runtime_error("CROWNAnalysis::concretizeNode - Patches without input_shape, cannot convert to matrix");
        }
    } else {
        uA = uA_bound.asTensor();
    }
    
    torch::Tensor lBias = _lowerBias.exists(inputIndex) ? _lowerBias[inputIndex] : torch::Tensor();
    torch::Tensor uBias = _upperBias.exists(inputIndex) ? _upperBias[inputIndex] : torch::Tensor();

    if (lA.defined() && lA.dim() >= 2) {
        int nodeDim = inputLower.size(0);
        int expectedNodeDim = lA.size(-1);
        if (nodeDim != expectedNodeDim) {
            log(Stringf("concretizeNode(%u) - Dim mismatch: input=%d, expected=%d; fallback to IBP", startIndex, nodeDim, expectedNodeDim));
            if (_ibpBounds.exists(startIndex)) {
                _concreteBounds[startIndex] = _ibpBounds[startIndex];
                _torchModel->setConcreteBounds(startIndex, _ibpBounds[startIndex]);
            }
            return;
        }
    }

    torch::Tensor concreteLower, concreteUpper;
    computeConcreteBounds(lA, uA, lBias, uBias, inputLower, inputUpper, concreteLower, concreteUpper);

    log(Stringf("concretizeNode() - Computed concrete bounds: lower.defined()=%d, upper.defined()=%d",
        concreteLower.defined(), concreteUpper.defined()));

    if (concreteLower.defined() && concreteUpper.defined()) {
        BoundedTensor<torch::Tensor> concreteBounds(concreteLower, concreteUpper);
        _concreteBounds[startIndex] = concreteBounds;
        _torchModel->setConcreteBounds(startIndex, concreteBounds);
        log(Stringf("concretizeNode() - Stored CROWN concrete bounds for node %u", startIndex));

        // CROWN DEBUG: Concretization complete (to file)
        {
            std::ostringstream oss;
            oss << "\n============================================================\n";
            oss << "[CROWN CONCRETIZATION COMPLETE]\n";
            oss << "  Final lower bounds (first 10): " << tensorFirstN(concreteLower) << "\n";
            oss << "  Final lower stats: min=" << std::fixed << std::setprecision(6)
                << concreteLower.min().item<float>() << ", max=" << concreteLower.max().item<float>()
                << ", mean=" << concreteLower.mean().item<float>() << ", shape=[";
            auto lower_shape = concreteLower.sizes();
            for (size_t i = 0; i < lower_shape.size(); ++i) {
                if (i > 0) oss << ", ";
                oss << lower_shape[i];
            }
            oss << "]\n";
            
            oss << "  Final upper bounds (first 10): " << tensorFirstN(concreteUpper) << "\n";
            oss << "  Final upper stats: min=" << std::fixed << std::setprecision(6)
                << concreteUpper.min().item<float>() << ", max=" << concreteUpper.max().item<float>()
                << ", mean=" << concreteUpper.mean().item<float>() << ", shape=[";
            auto upper_shape = concreteUpper.sizes();
            for (size_t i = 0; i < upper_shape.size(); ++i) {
                if (i > 0) oss << ", ";
                oss << upper_shape[i];
            }
            oss << "]\n";
            oss << "============================================================\n\n";
            
            writeToDebugLog(oss.str());
        }
    } else {
        log(Stringf("concretizeNode() - CROWN bounds undefined, falling back to IBP for node %u", startIndex));
        if (_ibpBounds.exists(startIndex)) {
            _concreteBounds[startIndex] = _ibpBounds[startIndex];
            _torchModel->setConcreteBounds(startIndex, _ibpBounds[startIndex]);
            log(Stringf("concretizeNode() - Stored IBP bounds as concrete bounds for node %u", startIndex));
        } else {
            log(Stringf("concretizeNode() - WARNING: No IBP bounds available for node %u", startIndex));
        }
    }

    log(Stringf("concretizeNode() - Finished, _concreteBounds.exists(%u)=%d",
        startIndex, _concreteBounds.exists(startIndex)));
}

// Helper function for establishing consistent tensor format (following auto-LiRPA's _preprocess_C)
torch::Tensor CROWNAnalysis::preprocessC(const torch::Tensor& C, unsigned outputSize) {
    // auto-LiRPA uses consistent (spec, batch, ...) format 
    // User provides (batch, spec) but internally converts to (spec, batch),
    // where batch is the number of constraints being verified, and spec is the number of outputs
    // For Marabou, we are assuming single constraint verification, so batch_size = 1
    
    // Ensure outputSize is valid
    if (outputSize == 0) {
        throw std::runtime_error("CROWNAnalysis: outputSize cannot be zero");
    }
    
    if (C.numel() == 0) {
        // Create identity matrix for single constraint verification
        // Shape should be [batch_size, output_size, output_size] for proper 3D operations
        // Following auto-LiRPA's approach: torch.eye(dim).unsqueeze(0).expand(batch_size, -1, -1)
        return torch::eye(outputSize, torch::kFloat32).unsqueeze(0);
    }
    
    // If C is provided, ensure it has the correct format
    if (C.dim() == 2) {
        // C has shape (batch, spec) -> keep as is for proper matrix multiplication
        return C; // Shape (batch, spec)
    } else if (C.dim() == 1) {
        // C has shape (spec) -> add batch dimension
        return C.unsqueeze(0); // Shape (1, spec)
    }
    
    // Default: return identity matrix with proper 3D shape
    // This creates [1, output_size, output_size]
    return torch::eye(outputSize, torch::kFloat32).unsqueeze(0); 
}


// Retained concretizeBounds for compatibility: concretize output node by default
void CROWNAnalysis::concretizeBounds()
{
    concretizeNode(getOutputIndex());
}

Vector<BoundedTensor<torch::Tensor>> CROWNAnalysis::getInputBoundsForNode(unsigned nodeIndex) {
    Vector<BoundedTensor<torch::Tensor>> inputBounds;
    
    if (_torchModel->getDependenciesMap().exists(nodeIndex) && !_torchModel->getDependencies(nodeIndex).empty()) {
        
        for (unsigned i = 0; i < _torchModel->getDependencies(nodeIndex).size(); ++i) {
            unsigned inputIndex = _torchModel->getDependencies(nodeIndex)[i];
            
            auto node = _nodes[inputIndex];
            unsigned outputSize = node->getOutputSize();
            torch::Tensor lower, upper;

            if (node->getNodeType() == NodeType::INPUT) {
                if (_torchModel->hasInputBounds()) {
                    lower = _torchModel->getInputLowerBounds();
                    upper = _torchModel->getInputUpperBounds();
                } else {
                    lower = torch::zeros({(long)outputSize}, torch::kFloat32);
                    upper = torch::ones({(long)outputSize}, torch::kFloat32);
                }
            } else {
                // First check for fixed intermediate bounds (alpha-CROWN best bounds tracking)
                auto fixedIt = _fixedConcreteBounds.find(inputIndex);
                if (fixedIt != _fixedConcreteBounds.end()) {
                    lower = fixedIt->second.first;
                    upper = fixedIt->second.second;
                }
                // Otherwise prefer CROWN concrete bounds in standard mode; otherwise IBP
                else if (_useStandardCROWN && _concreteBounds.exists(inputIndex)) {
                    lower = _concreteBounds[inputIndex].lower();
                    upper = _concreteBounds[inputIndex].upper();
                } else if (_ibpBounds.exists(inputIndex)) {
                    lower = _ibpBounds[inputIndex].lower();
                    upper = _ibpBounds[inputIndex].upper();
                } else {
                    lower = torch::zeros({(long)outputSize}, torch::kFloat32);
                    upper = torch::ones({(long)outputSize}, torch::kFloat32);
                }
            }
            inputBounds.append(BoundedTensor<torch::Tensor>(lower, upper));
        }
    } else {
        log(Stringf("getInputBoundsForNode() - Node %u has no dependencies", nodeIndex));
    }
    log(Stringf("getInputBoundsForNode() - Completed for node %u with %u input bounds", nodeIndex, inputBounds.size()));
    return inputBounds;
}



// Helpers to coerce shapes to (1, spec, n) and (1, n, 1)
static inline torch::Tensor ensure3A(const torch::Tensor& A) {
    if (!A.defined()) return A;
    if (A.dim() == 3) return A;
    if (A.dim() == 2) return A.unsqueeze(0);        // (1, spec, n)
    if (A.dim() == 1) return A.unsqueeze(0).unsqueeze(0);
    return A.unsqueeze(0); // best effort
}
static inline torch::Tensor ensure3x(const torch::Tensor& x) {
    // x is (n,) -> (1, n, 1); (b,n)->(b,n,1)
    if (!x.defined()) return x;
    if (x.dim() == 1) return x.unsqueeze(0).unsqueeze(-1);
    if (x.dim() == 2) return x.unsqueeze(-1);
    return x;
}
static inline torch::Tensor ensure3b(const torch::Tensor& b) {
    // b is (spec,) -> (1, spec, 1)
    if (!b.defined()) return b;
    if (b.dim() == 1) return b.unsqueeze(0).unsqueeze(-1);
    if (b.dim() == 2) return b.unsqueeze(-1);
    return b;
}


torch::Tensor CROWNAnalysis::computeConcreteLowerBound(
    const torch::Tensor& lA, const torch::Tensor& lBias,
    const torch::Tensor& xLower, const torch::Tensor& xUpper)
{
    if (!lA.defined()) return torch::Tensor();

    torch::Tensor AL = ensure3A(lA.to(torch::kFloat32));         // (1,spec,n)
    torch::Tensor xL = ensure3x(xLower.to(torch::kFloat32));     // (1,n,1)
    torch::Tensor xU = ensure3x(xUpper.to(torch::kFloat32));     // (1,n,1)
    torch::Tensor bL = ensure3b(lBias.to(torch::kFloat32));      // (1,spec,1)

    torch::Tensor Apos = torch::clamp_min(AL, 0);
    torch::Tensor Aneg = torch::clamp_max(AL, 0);

    // LB = βL + Apos * xL + Aneg * xU
    torch::Tensor term = Apos.bmm(xL) + Aneg.bmm(xU);            // (1,spec,1)
    torch::Tensor out  = term + bL;                              // (1,spec,1)
    return out.squeeze(-1).squeeze(0);                           // (spec,)
}


torch::Tensor CROWNAnalysis::computeConcreteUpperBound(
    const torch::Tensor& uA, const torch::Tensor& uBias,
    const torch::Tensor& xLower, const torch::Tensor& xUpper)
{
    if (!uA.defined()) return torch::Tensor();

    torch::Tensor AU = ensure3A(uA.to(torch::kFloat32));         // (1,spec,n)
    torch::Tensor xL = ensure3x(xLower.to(torch::kFloat32));     // (1,n,1)
    torch::Tensor xU = ensure3x(xUpper.to(torch::kFloat32));     // (1,n,1)
    torch::Tensor bU = ensure3b(uBias.to(torch::kFloat32));      // (1,spec,1)

    torch::Tensor Apos = torch::clamp_min(AU, 0);
    torch::Tensor Aneg = torch::clamp_max(AU, 0);

    // UB = βU + Apos * xU + Aneg * xL
    torch::Tensor term = Apos.bmm(xU) + Aneg.bmm(xL);            // (1,spec,1)
    torch::Tensor out  = term + bU;                              // (1,spec,1)
    return out.squeeze(-1).squeeze(0);                           // (spec,)
}


void CROWNAnalysis::computeConcreteBounds(
    const torch::Tensor& lA, const torch::Tensor& uA,
    const torch::Tensor& lBias, const torch::Tensor& uBias,
    const torch::Tensor& nodeLower, const torch::Tensor& nodeUpper,
    torch::Tensor& concreteLower, torch::Tensor& concreteUpper)
{
    concreteLower = computeConcreteLowerBound(lA, lBias, nodeLower, nodeUpper);
    concreteUpper = computeConcreteUpperBound(uA, uBias, nodeLower, nodeUpper);
}


unsigned CROWNAnalysis::getOutputIndex() const {
    // Find the node with the highest index (assuming it's the output)
    unsigned outputIndex = 0;
    for (const auto& pair : _nodes) {
        if (pair.first > outputIndex) {
            outputIndex = pair.first;
        }
    }
    return outputIndex;
}

// Add helper function for A matrix addition
BoundA CROWNAnalysis::addA(const BoundA& A1, const BoundA& A2) {
    if (!A1.defined()) return A2;
    if (!A2.defined()) return A1;

    if (A1.isTensor() && A2.isTensor()) {
        torch::Tensor t1 = A1.asTensor();
        torch::Tensor t2 = A2.asTensor();
        
        // Enforce shape consistency - this forces consistent (1, spec, n) everywhere
        // and avoids "randomly worse" steps caused by shape collapse
        if (t1.sizes() != t2.sizes()) {
            std::string shape1_str = "[";
            for (int i = 0; i < t1.dim(); ++i) {
                shape1_str += std::to_string(t1.size(i));
                if (i < t1.dim() - 1) shape1_str += ", ";
            }
            shape1_str += "]";

            std::string shape2_str = "[";
            for (int i = 0; i < t2.dim(); ++i) {
                shape2_str += std::to_string(t2.size(i));
                if (i < t2.dim() - 1) shape2_str += ", ";
            }
            shape2_str += "]";

            throw std::runtime_error(
                "CROWNAnalysis::addA - A shape mismatch: A1.shape=" + shape1_str +
                " vs A2.shape=" + shape2_str);
        }

        return BoundA(t1 + t2);
    } else if (A1.isPatches() && A2.isPatches()) {
        return BoundA(A1.asPatches()->add(A2.asPatches()));
    } else {
        throw std::runtime_error(
            "CROWNAnalysis::addA - Mixed types (Tensor/Patches) addition not implemented");
    }
}

// Add helper function for proper A matrix accumulation
void CROWNAnalysis::addBound(unsigned nodeIndex, const BoundA& lA, const BoundA& uA) {
    
    if (_lA.exists(nodeIndex)) {
        _lA[nodeIndex] = addA(_lA[nodeIndex], lA);
    } else {
        _lA[nodeIndex] = lA;
    }
    
    if (_uA.exists(nodeIndex)) {
        _uA[nodeIndex] = addA(_uA[nodeIndex], uA);
    } else {
        _uA[nodeIndex] = uA;
    }
}

void CROWNAnalysis::addBias(unsigned nodeIndex, const torch::Tensor& lBias, const torch::Tensor& uBias) 
{
    if (lBias.defined() && lBias.numel() > 0) {
        if (_lowerBias.exists(nodeIndex)) {
            _lowerBias[nodeIndex] = _lowerBias[nodeIndex] + lBias;
        } else {
            _lowerBias[nodeIndex] = lBias;
        }
    }
    if (uBias.defined() && uBias.numel() > 0) {
        if (_upperBias.exists(nodeIndex)) {
            _upperBias[nodeIndex] = _upperBias[nodeIndex] + uBias;
        } else {
            _upperBias[nodeIndex] = uBias;
        }
    }
    
    
}

bool CROWNAnalysis::isProcessed(unsigned nodeIndex) const
{
    return _torchModel->isProcessed(nodeIndex);
}

void CROWNAnalysis::resetProcessingState()
{
    _torchModel->resetProcessingState();
}

void CROWNAnalysis::clearConcreteBounds()
{
    _torchModel->clearConcreteBounds();
}

void CROWNAnalysis::markProcessed(unsigned nodeIndex)
{
    _torchModel->markProcessed(nodeIndex);
}

void CROWNAnalysis::log( const String &message )
{
    (void)message;
    if ( GlobalConfiguration::NETWORK_LEVEL_REASONER_LOGGING )
    {
        //printf( "CROWNAnalysis: %s\n", message.ascii() );
    }
}

// ------------------------------------------------------------
// Public Get mothods for the unit testing -> shouldn't be needed in the actual CROWN analysis (maybe for creating tightenings to update Marabou engine)
// ------------------------------------------------------------

torch::Tensor CROWNAnalysis::getIBPLowerBound(unsigned nodeIndex)
{
    if (_ibpBounds.exists(nodeIndex)) {
        return _ibpBounds[nodeIndex].lower();
    }
    return torch::Tensor();
}

torch::Tensor CROWNAnalysis::getIBPUpperBound(unsigned nodeIndex)
{
    if (_ibpBounds.exists(nodeIndex)) {
        return _ibpBounds[nodeIndex].upper();
    }
    return torch::Tensor();
}

torch::Tensor CROWNAnalysis::getCrownLowerBound(unsigned nodeIndex) const
{
    if (_lA.exists(nodeIndex)) {
        BoundA b = _lA[nodeIndex];
        if (b.isTensor()) return b.asTensor();
        // Return undefined or try convert?
        // For now undefined to avoid crash if input shape unknown
        return torch::Tensor();
    }
    return torch::Tensor();
}

torch::Tensor CROWNAnalysis::getCrownUpperBound(unsigned nodeIndex) const
{
    if (_uA.exists(nodeIndex)) {
        BoundA b = _uA[nodeIndex];
        if (b.isTensor()) return b.asTensor();
        return torch::Tensor();
    }
    return torch::Tensor();
}

bool CROWNAnalysis::hasIBPBounds(unsigned nodeIndex)
{
    return _ibpBounds.exists(nodeIndex);
}

bool CROWNAnalysis::hasCrownBounds(unsigned nodeIndex)
{
    return _lA.exists(nodeIndex) || _uA.exists(nodeIndex);
}

unsigned CROWNAnalysis::getNumNodes() const
{
    return _nodes.size();
}

std::shared_ptr<BoundedTorchNode> CROWNAnalysis::getNode(unsigned index) const
{
    if (_nodes.exists(index)) {
        return _nodes[index];
    }
    return nullptr;
}

unsigned CROWNAnalysis::getInputSize() const
{
    return _torchModel->getInputSize();
}

unsigned CROWNAnalysis::getOutputSize() const
{
    return _torchModel->getOutputSize();
}

// Concrete bound access methods
torch::Tensor CROWNAnalysis::getConcreteLowerBound(unsigned nodeIndex)
{
    if (_concreteBounds.exists(nodeIndex)) {
        return _concreteBounds[nodeIndex].lower();
    }
    return torch::Tensor();
}

torch::Tensor CROWNAnalysis::getConcreteUpperBound(unsigned nodeIndex)
{
    if (_concreteBounds.exists(nodeIndex)) {
        return _concreteBounds[nodeIndex].upper();
    }
    return torch::Tensor();
}

bool CROWNAnalysis::hasConcreteBounds(unsigned nodeIndex)
{
    return _concreteBounds.exists(nodeIndex);
}

void CROWNAnalysis::setFixedConcreteBounds(unsigned nodeIndex, const torch::Tensor& lower, const torch::Tensor& upper)
{
    _fixedConcreteBounds[nodeIndex] = std::make_pair(lower.detach().clone(), upper.detach().clone());
}

void CROWNAnalysis::clearFixedConcreteBounds()
{
    _fixedConcreteBounds.clear();
}

// Output bound access methods
BoundedTensor<torch::Tensor> CROWNAnalysis::getOutputBounds() const 
{
    unsigned outputIndex = getOutputIndex();
    if (_concreteBounds.exists(outputIndex)) {
        return _concreteBounds[outputIndex];
    }
    return BoundedTensor<torch::Tensor>(torch::Tensor(), torch::Tensor());
}

BoundedTensor<torch::Tensor> CROWNAnalysis::getOutputIBPBounds() const 
{
    unsigned outputIndex = getOutputIndex();
    if (_ibpBounds.exists(outputIndex)) {
        return _ibpBounds[outputIndex];
    }
    return BoundedTensor<torch::Tensor>(torch::Tensor(), torch::Tensor());
}

BoundedTensor<torch::Tensor> CROWNAnalysis::getNodeIBPBounds(unsigned nodeIndex) const {
    if (_ibpBounds.exists(nodeIndex)) {
        return _ibpBounds[nodeIndex];
    }
    return BoundedTensor<torch::Tensor>();
}

BoundedTensor<torch::Tensor> CROWNAnalysis::getNodeCrownBounds(unsigned nodeIndex) const {
    if (_lA.exists(nodeIndex) || _uA.exists(nodeIndex)) {
        torch::Tensor lA = getCrownLowerBound(nodeIndex);
        torch::Tensor uA = getCrownUpperBound(nodeIndex);
        return BoundedTensor<torch::Tensor>(lA, uA);
    }
    return BoundedTensor<torch::Tensor>();
}

BoundedTensor<torch::Tensor> CROWNAnalysis::getNodeConcreteBounds(unsigned nodeIndex) const {
    if (_concreteBounds.exists(nodeIndex)) {
        return _concreteBounds[nodeIndex];
    }
    return BoundedTensor<torch::Tensor>();
}

// First linear layer IBP fast path optimization methods
bool CROWNAnalysis::checkIBPFirstLinear(unsigned nodeIndex)
{
    if (!_enableFirstLinearIBP) {
        return false;
    }
    
    // Check if this is a first linear layer optimization candidate
    return isFirstLinearLayer(nodeIndex);
}

bool CROWNAnalysis::isFirstLinearLayer(unsigned nodeIndex)
{
    // Get the node
    if (!_nodes.exists(nodeIndex)) {
        return false;
    }
    
    auto& node = _nodes[nodeIndex];
    
    // Must be a linear layer
    if (node->getNodeType() != NodeType::LINEAR) {
        return false;
    }
    
    // Check if this linear layer is directly connected to input nodes
    if (!_torchModel->getDependenciesMap().exists(nodeIndex)) {
        return false;
    }
    
    const Vector<unsigned>& dependencies = _torchModel->getDependencies(nodeIndex);
    
    // For first linear layer optimization, we expect:
    // 1. Only one input dependency (the input node)
    // 2. That dependency should be an input node
    if (dependencies.size() != 1) {
        return false;
    }
    
    unsigned inputNodeIndex = dependencies[0];
    if (!_nodes.exists(inputNodeIndex)) {
        return false;
    }
    
    // The dependency must be an input node
    return _nodes[inputNodeIndex]->getNodeType() == NodeType::INPUT;
}

void CROWNAnalysis::setInputBounds(const BoundedTensor<torch::Tensor>& inputBounds) {
    log("[CROWNAnalysis] Setting input bounds");

    // Delegate to the torch model for input bounds management
    _torchModel->setInputBounds(inputBounds);

    log("[CROWNAnalysis] Input bounds set successfully");
}

} // namespace NLR
