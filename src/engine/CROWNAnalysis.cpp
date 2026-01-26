#include "CROWNAnalysis.h"
#include "nodes/BoundedConstantNode.h"
#include "nodes/BoundedBatchNormNode.h"

#include "Debug.h"
#include "MStringf.h"
#include "LirpaError.h"
#include "TimeUtils.h"

#include <vector>
#include <iomanip>
#include <sstream>


namespace NLR {


static std::string tensorStatsStr(const torch::Tensor& t) {
    if (!t.defined() || t.numel() == 0) return "undef";
    torch::Tensor f = t.to(torch::kFloat32);
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6)
        << "min=" << f.min().item<float>()
        << " max=" << f.max().item<float>()
        << " mean=" << f.mean().item<float>();
    return oss.str();
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

static std::string tensorShapeStr(const torch::Tensor& t) {
    if (!t.defined()) return "undef";
    std::ostringstream oss;
    oss << "[";
    auto shape = t.sizes();
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << shape[i];
    }
    oss << "]";
    return oss.str();
}

torch::Tensor CROWNAnalysis::buildCenterInputForForward() const {
    if (!_torchModel || !_torchModel->hasInputBounds()) return torch::Tensor();

    torch::Tensor lb = _torchModel->getInputLowerBounds();
    torch::Tensor ub = _torchModel->getInputUpperBounds();
    if (!lb.defined() || !ub.defined()) return torch::Tensor();

    lb = lb.to(torch::kFloat32);
    ub = ub.to(torch::kFloat32);
    torch::Tensor center = (lb + ub) / 2.0;

    // Match TorchModel::cacheForwardShapesFromCenter() input reshaping heuristic.
    // The ONNX models typically expect image tensors, while VNN-LIB bounds are flat.
    if (center.numel() == 9408) {
        center = center.view({1, 3, 56, 56});
    } else if (center.numel() == 12288) {
        center = center.view({1, 3, 64, 64});
    } else if (center.numel() == 3072) {
        center = center.view({1, 3, 32, 32});
    } else if (center.numel() == 784) {
        center = center.view({1, 1, 28, 28});
    } else {
        center = center.view({1, (long)center.numel()});
    }
    return center;
}

void CROWNAnalysis::ensureCenterActivations() {
    if (_hasCenterActivations) return;
    if (!_torchModel) return;

    torch::NoGradGuard no_grad;
    try {
        torch::Tensor center = buildCenterInputForForward();
        if (!center.defined()) return;
        _centerActivations = _torchModel->forwardAndStoreActivations(center);
        _hasCenterActivations = true;
    } catch (const std::exception& e) {
        (void)e;  // Suppress unused variable warning
        _hasCenterActivations = false;
        _centerActivations.clear();
    } catch (...) {
        _hasCenterActivations = false;
        _centerActivations.clear();
    }
}


CROWNAnalysis::CROWNAnalysis( TorchModel *torchModel )
    : _torchModel( torchModel )
{
    // Configuration is now accessed via LirpaConfiguration static members 
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


void CROWNAnalysis::run(bool enableGradients)
{
    log("run() - Starting");
    // Conditionally enable/disable gradient tracking
    // For Alpha-CROWN optimization, gradients must be enabled to flow through bounds computation
    std::unique_ptr<torch::NoGradGuard> no_grad;
    if (!enableGradients) {
        no_grad = std::make_unique<torch::NoGradGuard>(); // Disable gradient tracking for better performance
    }
    std::string stage = "start";
    try {
        // Reset per-run debug state.
        _hasCenterActivations = false;
        _centerActivations.clear();
        _foundFirstUnsound = false;
        _firstUnsoundNode = 0;

        // IMPORTANT: When gradients are enabled (Alpha-CROWN), use simple CROWN mode (single backward pass)
        // to avoid in-place modification conflicts during multiple backward passes.
        // The standard CROWN mode with multiple backward passes causes issues with PyTorch's autograd
        // when the same alpha tensors are accessed from different computation graphs.
        bool useStandardCrown = LirpaConfiguration::USE_STANDARD_CROWN && !enableGradients;
        
        if (useStandardCrown) {
            // Standard CROWN: Selective backward passes for ReLU layers

            // First, compute IBP bounds as reference (used for linear/conv layers)
            stage = "computeIBPBounds(standard)";
            computeIBPBounds();

            // Copy IBP bounds to concrete bounds for all nodes initially
            // Linear/Conv layers will keep these IBP bounds (they're tight enough)
            for (const auto& p : _ibpBounds) {
                _concreteBounds[p.first] = p.second;
                _torchModel->setConcreteBounds(p.first, p.second);
            }

            // Identify ReLU layers that need CROWN bounds (tighter than IBP)
            Vector<unsigned> nodesNeedingCrown;
            Vector<unsigned> forwardOrder = _torchModel->topologicalSort();

            // Find the input node index for checking first layer connections
            unsigned inputNodeIdx = 0;
            for (const auto& p : _nodes) {
                if (p.second->getNodeType() == NodeType::INPUT) {
                    inputNodeIdx = p.first;
                    break;
                }
            }

            for (unsigned nodeIdx : forwardOrder) {
                NodeType nodeType = _nodes[nodeIdx]->getNodeType();
                if (nodeType == NodeType::RELU || nodeType == NodeType::SIGMOID) {
                    // Check if this activation follows the first linear/conv layer
                    // If so, skip it as IBP bounds are already exact for first linear layer
                    bool isFirstLayerActivation = false;

                    if (_torchModel->getDependenciesMap().exists(nodeIdx)) {
                        const Vector<unsigned>& deps = _torchModel->getDependencies(nodeIdx);
                        if (deps.size() == 1) {
                            unsigned prevNode = deps[0];
                            // Check if previous node is linear/conv connected directly to input
                            if (_nodes[prevNode]->getNodeType() == NodeType::LINEAR ||
                                _nodes[prevNode]->getNodeType() == NodeType::CONV ||
                                _nodes[prevNode]->getNodeType() == NodeType::BATCHNORM) {

                                if (_torchModel->getDependenciesMap().exists(prevNode)) {
                                    const Vector<unsigned>& prevDeps = _torchModel->getDependencies(prevNode);
                                    // Check if directly connected to input (possibly through reshape)
                                    for (unsigned pd : prevDeps) {
                                        if (pd == inputNodeIdx) {
                                            isFirstLayerActivation = true;
                                            log(Stringf("run() - Skipping activation at node %u (follows first linear layer, IBP is exact)", nodeIdx));
                                            break;
                                        }
                                        // Also check if connected through reshape/flatten nodes
                                        if (_nodes[pd]->getNodeType() == NodeType::RESHAPE ||
                                            _nodes[pd]->getNodeType() == NodeType::FLATTEN) {
                                            if (_torchModel->getDependenciesMap().exists(pd)) {
                                                const Vector<unsigned>& reshapeDeps = _torchModel->getDependencies(pd);
                                                if (reshapeDeps.size() == 1 && reshapeDeps[0] == inputNodeIdx) {
                                                    isFirstLayerActivation = true;
                                                    log(Stringf("run() - Skipping activation at node %u (follows first linear after reshape)", nodeIdx));
                                                    break;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    if (!isFirstLayerActivation) {
                        // Instead of adding the activation node itself, we must add its INPUT nodes.
                        // This ensures that the pre-activation bounds (Linear/Conv) are tightened using CROWN,
                        // which are then used to calculate tighter relaxation slopes for this activation.

                        if (_torchModel->getDependenciesMap().exists(nodeIdx)) {
                            const Vector<unsigned>& deps = _torchModel->getDependencies(nodeIdx);
                            for (unsigned dep : deps) {
                                // Avoid adding duplicates or constant nodes if possible (though concretize handles it)

                                bool found = false;
                                for (unsigned existing : nodesNeedingCrown) {
                                    if (existing == dep) { found = true; break; }
                                }
                                if (!found) {
                                    nodesNeedingCrown.append(dep);
                                    log(Stringf("run() - Adding activation input node %u for CROWN bounds", dep));
                                }
                            }
                        }
                    }
                }
            }

            // Always compute CROWN bounds for the output node
            unsigned outputIndex = getOutputIndex();
            nodesNeedingCrown.append(outputIndex);
            

            log(Stringf("run() - Will perform %u backward passes (ReLUs%s)",
                        nodesNeedingCrown.size(), _torchModel->hasSpecificationMatrix() ? " + specification matrix output" : " + output"));

            // Perform selective backward passes from each node that needs CROWN bounds
            // This gives tight bounds where needed while keeping efficiency
            for (unsigned targetNode : nodesNeedingCrown) {
                auto& node = _nodes[targetNode];
                unsigned nodeSize = node->getOutputSize();
                std::string startKey = "/" + std::to_string(targetNode);
                
                // Identify unstable neurons for this node using IBP bounds
                Vector<unsigned> unstableIndices;
                bool sparseMode = false;
                
                if (_ibpBounds.exists(targetNode)) {
                    auto bounds = _ibpBounds[targetNode];
                    torch::Tensor lower = bounds.lower();
                    torch::Tensor upper = bounds.upper();
                    
                    if (lower.defined() && upper.defined()) {
                        // Identify neurons where lower < 0 < upper
                        // For ReLUs, stable means >= 0 or <= 0.
                        // For other nodes, "unstable" in this context means we want tighter bounds.
                        // Usually we want bounds for all neurons if we are at output.
                        // If we are at intermediate (pre-ReLU), we only care about neurons that are ambiguous for ReLU.
                        
                        // Heuristic: If node is not output node, assume it feeds into ReLU and check < 0 < u
                        if (targetNode != outputIndex) {
                            torch::Tensor unstableMask = (lower < 0) & (upper > 0);
                            int64_t unstableCount = unstableMask.sum().item<int64_t>();
                            
                            // Threshold for sparse mode (e.g. if < 50% neurons are unstable)
                            if (unstableCount < nodeSize && unstableCount > 0) {
                                // Collect indices
                                auto indices = unstableMask.nonzero().flatten(); // [N]
                                // Convert to Vector<unsigned>
                                auto indices_accessor = indices.accessor<int64_t, 1>();
                                for (int i = 0; i < indices.size(0); ++i) {
                                    unstableIndices.append(static_cast<unsigned>(indices_accessor[i]));
                                }
                                sparseMode = true;
                                log(Stringf("run() - Node %u: Sparse CROWN for %u/%u unstable neurons", 
                                    targetNode, unstableCount, nodeSize));
                            } else if (unstableCount == 0) {
                                // All stable! No need for CROWN at all.
                                log(Stringf("run() - Node %u: All neurons stable (IBP sufficient), skipping CROWN", targetNode));
                                // We can just use IBP bounds as concrete bounds
                                _concreteBounds[targetNode] = bounds;
                                _torchModel->setConcreteBounds(targetNode, bounds);
                                continue;
                            }
                        }
                    }
                }

                _setCurrentStart(startKey, sparseMode ? unstableIndices.size() : nodeSize);

                // Perform backward pass from this specific node
                stage = std::string("backwardFrom(targetNode=") + std::to_string(targetNode) + ")";
                backwardFrom(targetNode, unstableIndices);

                // Concretize CROWN bounds for this node
                stage = std::string("concretizeNode(targetNode=") + std::to_string(targetNode) + ")";
                concretizeNode(targetNode, unstableIndices);
            }
            
            // Output node bounds are computed above in the loop
            // If specification matrix is set in TorchModel, backwardFrom() will use it automatically

            // All nodes now have bounds: CROWN for ReLUs/output, IBP for linear/conv
        } else {
            // CROWN-IBP: IBP for intermediates, CROWN for final
            stage = "computeIBPBounds(crown_ibp)";
            computeIBPBounds();

            // Set current start context for the output node
            unsigned outputIndex = getOutputIndex();
            auto& outputNode = _nodes[outputIndex];
            unsigned outputSize = outputNode->getOutputSize();
            std::string startKey = "/" + std::to_string(outputIndex);
            _setCurrentStart(startKey, outputSize);

            stage = std::string("backwardFrom(outputIndex=") + std::to_string(outputIndex) + ")";
            backwardFrom(outputIndex);
            stage = std::string("concretizeNode(outputIndex=") + std::to_string(outputIndex) + ")";
            concretizeNode(outputIndex);
        }
    } catch (const CommonError &e) {
        std::ostringstream oss;
        oss << "CROWNAnalysis::run CommonError code=" << e.getCode() << " stage=" << stage;
        const char *msg = e.getUserMessage();
        if (msg && msg[0] != '\0') oss << " message=" << msg;
        throw std::runtime_error(oss.str());
    } catch (const Error &e) {
        std::ostringstream oss;
        oss << "CROWNAnalysis::run Error class=" << e.getErrorClass() << " code=" << e.getCode()
            << " stage=" << stage;
        const char *msg = e.getUserMessage();
        if (msg && msg[0] != '\0') oss << " message=" << msg;
        throw std::runtime_error(oss.str());
    } catch (const std::exception& e) {
        log(Stringf("run() - Exception caught: %s", e.what()));
        throw std::runtime_error(std::string("CROWNAnalysis::run exception stage=") + stage + ": " + e.what());
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

        // printf("computeIBPBounds() - Processing node %u\n", nodeIndex);

        // Get input bounds for this node
        Vector<BoundedTensor<torch::Tensor>> inputBounds = getInputBoundsForNode(nodeIndex);

        // log(Stringf("computeIBPBounds() - Node %u has %u input bounds", nodeIndex, inputBounds.size()));

        // For the input node, if explicit input bounds exist, store and continue
        if (node->getNodeType() == NodeType::INPUT) {
            if (_torchModel->hasInputBounds()) {
                torch::Tensor inputLower = _torchModel->getInputLowerBounds();
                torch::Tensor inputUpper = _torchModel->getInputUpperBounds();
                _ibpBounds[nodeIndex] = BoundedTensor<torch::Tensor>(inputLower, inputUpper);
                // log(Stringf("computeIBPBounds() - Node %u (INPUT) stored with bounds shape [%lld]",
                //     nodeIndex, (long long)inputLower.size(0)));
                continue;
            }
        }

        // Compute IBP bounds (same computation for all node types)
        BoundedTensor<torch::Tensor> ibpBounds = node->computeIntervalBoundPropagation(inputBounds);

        // Store IBP bounds
        _ibpBounds[nodeIndex] = ibpBounds;
    }

    log(Stringf("computeIBPBounds() - Completed, stored bounds for %u nodes", _ibpBounds.size()));
}

// Global counter for backward passes
// static int _backward_from_counter = 0;

void CROWNAnalysis::backwardFrom(unsigned startIndex, const Vector<unsigned>& unstableIndices, const torch::Tensor* C)
{
    unsigned current_dbg = startIndex;
    std::string stage = "start";
    try {
    // Increment and print counter
    // _backward_from_counter++;
    // printf("[C++ COUNTER] backwardFrom call #%d: startIndex=%u\n", _backward_from_counter, startIndex);

    stage = "checkStartIndexExists";
    if ( !_nodes.exists(startIndex) ) {
        log(Stringf("backwardFrom() - Warning: start index %u not found in nodes.", startIndex));
        return;
    }

    // Fresh state for this run
    stage = "clearCrownState";
    clearCrownState();
    _nodesNeedingBounds.clear(); // Clear the set of nodes needing bounds

    // Initialize with identity matrices or specification matrix for the start node
    auto& startNode = _nodes[startIndex];
    unsigned startSize = startNode->getOutputSize();
    unsigned outputIndex = getOutputIndex();
    
    torch::Tensor initMatrix;
    long numSpecs;
    
    // C matrix (specification matrix) should only be used when starting from the output node
    // For intermediate nodes, always use identity matrix
    bool isOutputNode = (startIndex == outputIndex);
    
    // Determine which matrix to use: explicit C, TorchModel specification, or identity
    // All C matrices must be preprocessed to match auto_LiRPA's internal format
    if (isOutputNode && C != nullptr) {
        // Explicit C matrix provided - preprocess it (using output node for correct sizing)
        initMatrix = preprocessC(*C, outputIndex);
        numSpecs = initMatrix.size(0);
        log(Stringf("backwardFrom() - Using explicitly provided C matrix (preprocessed), shape [%ld, %ld, %ld]",
                    initMatrix.size(0), initMatrix.size(1), initMatrix.size(2)));
    } else if (isOutputNode && _torchModel->hasSpecificationMatrix()) {
        // Query TorchModel for specification matrix - preprocess it (using output node for correct sizing)
        torch::Tensor specMatrix = _torchModel->getSpecificationMatrix();
        
        initMatrix = preprocessC(specMatrix, outputIndex);
        numSpecs = initMatrix.size(0);
        
        log(Stringf("backwardFrom() - Using specification matrix from TorchModel (preprocessed), shape [%ld, %ld, %ld]",
                    initMatrix.size(0), initMatrix.size(1), initMatrix.size(2)));
    } else {
        // For intermediate nodes or when no C matrix: use identity matrix
        // Identity matrix size matches the start node's output size
        initMatrix = preprocessC(torch::Tensor(), startIndex);
        numSpecs = initMatrix.size(0);
        if (isOutputNode) {
            log(Stringf("backwardFrom() - Using identity matrix (no specification matrix), shape [%ld, %ld, %ld]",
                        initMatrix.size(0), initMatrix.size(1), initMatrix.size(2)));
        } else {
            log(Stringf("backwardFrom() - Using identity matrix for intermediate node %u (output node is %u), shape [%ld, %ld, %ld]",
                        startIndex, outputIndex, initMatrix.size(0), initMatrix.size(1), initMatrix.size(2)));
        }
    }

    if (unstableIndices.empty()) {
        // Dense mode: use initMatrix (C matrix, specification matrix, or identity)
        _lA[startIndex] = BoundA(initMatrix);
        _uA[startIndex] = BoundA(initMatrix);
        
        // Initialize bias as [spec, batch] format: [numSpecs, 1]
        // This matches Python behavior where bias is always [spec, batch]
        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(_torchModel->getDevice());
        _lowerBias[startIndex] = torch::zeros({numSpecs, 1}, options);
        _upperBias[startIndex] = torch::zeros({numSpecs, 1}, options);
    } else {
        // Sparse/Gathered identity for selected outputs
        unsigned numUnstable = unstableIndices.size();
        // Shape: [numUnstable, 1, startSize] = [spec, batch, features]
        // C has 1s at (i, 0, unstableIndices[i])
        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(_torchModel->getDevice());
        torch::Tensor identityMatrix = torch::zeros({(long)numUnstable, 1, (long)startSize}, options);
        
        // Fill manually or use scatter
        // Since batch=1, we fill identityMatrix[i, 0, unstableIndices[i]] = 1
        auto accessor = identityMatrix.accessor<float, 3>();
        for (unsigned i = 0; i < numUnstable; ++i) {
            unsigned idx = unstableIndices[i];
            if (idx < startSize) {
                accessor[i][0][idx] = 1.0f;
            }
        }
        
        _lA[startIndex] = BoundA(identityMatrix);
        _uA[startIndex] = BoundA(identityMatrix);
        
        // Bias initialization: matches spec dimension (numUnstable)
        // Initialize as [spec, batch] format: [numUnstable, 1]
        // Bias accumulates terms from backward pass. Initial bias is 0.
        _lowerBias[startIndex] = torch::zeros({(long)numUnstable, 1}, options);
        _upperBias[startIndex] = torch::zeros({(long)numUnstable, 1}, options);
        
        log(Stringf("backwardFrom() - Initialized sparse C for node %u with %u unstable neurons", startIndex, numUnstable));
    }


    stage = "buildReachableSet";
    Set<unsigned> reachable;
    Queue<unsigned> work;
    work.push(startIndex);
    reachable.insert(startIndex);
    while (!work.empty()) {
        unsigned n = work.peak();
        work.pop();
        if (_torchModel->getDependenciesMap().exists(n)) {
            const auto &deps = _torchModel->getDependencies(n);
            for (unsigned d : deps) {
                if (!reachable.exists(d)) {
                    reachable.insert(d);
                    work.push(d);
                }
            }
        }
    }

    stage = "computePendingCounts";
    Map<unsigned, unsigned> pending;
    for (const auto &p : reachable) {
        unsigned v = p;
        unsigned cnt = 0;
        auto deps_of_v = _torchModel->getDependents(v);
        for (unsigned dep : deps_of_v) {
            if (reachable.exists(dep)) cnt++;
        }
        pending[v] = cnt;
    }

    // Start node is always ready (it is the source of identity A).
    Queue<unsigned> queue;
    queue.push(startIndex);

    // Local processed set (do not reuse TorchModel global processed flags).
    Set<unsigned> processed;

    log(Stringf("backwardFrom() - Starting scheduled processing from node %u (reachable=%u)", startIndex, reachable.size()));

    while (!queue.empty()) {
        unsigned current = queue.peak();
        current_dbg = current;
        stage = "loop";
        queue.pop();

        if (processed.exists(current)) continue;
        processed.insert(current);

        stage = "nodeLookup";
        if (!_nodes.exists(current)) {
            throw std::runtime_error("CROWNAnalysis::backwardFrom - missing node in _nodes map for index " + std::to_string(current));
        }
        auto& node = _nodes[current];
        NodeType nodetype = node->getNodeType();

        if (!_lA.exists(current) && !_uA.exists(current)) {
            log(Stringf("No A matrices for node %u, skipping", current));
            // Even if there is no A, we must still unblock dependencies for scheduling,
            // otherwise join points could deadlock.
            if (_torchModel->getDependenciesMap().exists(current)) {
                const Vector<unsigned> &deps = _torchModel->getDependencies(current);
                for (unsigned inputIndex : deps) {
                    if (!reachable.exists(inputIndex)) continue;
                    if (pending.exists(inputIndex) && pending[inputIndex] > 0) {
                        pending[inputIndex] = pending[inputIndex] - 1;
                        if (pending[inputIndex] == 0) queue.push(inputIndex);
                    }
                }
            }
            continue;
        }

        stage = "getInputBoundsForNode";
        // Bounds for current's inputs: IBP or CROWN-concrete depending on mode and availability
        Vector<BoundedTensor<torch::Tensor>> inputBounds = getInputBoundsForNode(current);

        BoundA currentLowerAlpha = _lA.exists(current) ? _lA[current] : BoundA();
        BoundA currentUpperAlpha = _uA.exists(current) ? _uA[current] : BoundA();

        Vector<Pair<BoundA, BoundA>> A_matrices;
        torch::Tensor lbias, ubias;
        
        // Check if we can skip full CROWN computation for first linear layer
        if (checkIBPFirstLinear(current) && !LirpaConfiguration::USE_STANDARD_CROWN) {
            // For first linear layers connected directly to inputs, IBP bounds are sufficient
            // and we can avoid the expensive CROWN backward computation and C matrix construction
            log(Stringf("backwardFrom() -  Skipping CROWN backward for first linear layer %u (using IBP)", current));
            continue;
        }
        
        // Skip CROWN backward propagation for non-perturbed nodes (constants, weights, biases)
        if (!node->isPerturbed()) {
            // For constant nodes, their contribution is added directly to the bias term
            // No need to propagate A matrices through them
            log(Stringf("backwardFrom() - Skipping non-perturbed node %u (%s)", 
                       current, node->getNodeName().ascii()));
            
            // TODO: Implement add_constant_node equivalent if needed
            // For now, just skip the backward propagation, but still advance scheduling.
            if (_torchModel->getDependenciesMap().exists(current)) {
                const Vector<unsigned> &deps = _torchModel->getDependencies(current);
                for (unsigned inputIndex : deps) {
                    if (!reachable.exists(inputIndex)) continue;
                    if (pending.exists(inputIndex) && pending[inputIndex] > 0) {
                        pending[inputIndex] = pending[inputIndex] - 1;
                        if (pending[inputIndex] == 0) queue.push(inputIndex);
                    }
                }
            }
            continue;
        }
        

        stage = "node.boundBackward";
        node->boundBackward(currentLowerAlpha, currentUpperAlpha, inputBounds, A_matrices, lbias, ubias);


        // No need to set intermediate bounds here - they're already set in run()
        // IBP bounds are used for linear/conv, CROWN for ReLUs

        stage = "propagateToDependencies";
        if (_torchModel->getDependenciesMap().exists(current))
        {
            for (unsigned i = 0; i < _torchModel->getDependencies(current).size() && i < A_matrices.size(); ++i)
            {
                unsigned inputIndex = _torchModel->getDependencies(current)[i];

                BoundA new_lA = A_matrices[i].first();
                BoundA new_uA = A_matrices[i].second();

                addBound(inputIndex, new_lA, new_uA);

                // IMPORTANT: Avoid bias duplication for multi-input nodes (e.g., Add(x, y)).
                // For an expression A*(x+y)+b, the bias term b should appear exactly once:
                //   A*x + A*y + b
                // If we propagate b to both inputs, it will be counted twice when paths merge.
                // We attach the accumulated bias to the first dependency only.
                torch::Tensor propagated_lbias;
                torch::Tensor propagated_ubias;
                if (i == 0) {
                    propagated_lbias = lbias.defined() ? lbias.clone() : torch::Tensor();
                    propagated_ubias = ubias.defined() ? ubias.clone() : torch::Tensor();

                    // Helper to normalize bias to [spec, batch] format
                    // Also tries to infer correct dimension order from A matrix if available
                    auto normalize_bias = [&](const torch::Tensor& bias, const BoundA& A_for_context) -> torch::Tensor {
                        if (!bias.defined() || bias.numel() == 0) return bias;
                        
                        // Get A matrix dimensions for context
                        int64_t A_spec = -1, A_batch = -1;
                        if (A_for_context.defined() && A_for_context.isTensor()) {
                            torch::Tensor A_tensor = A_for_context.asTensor();
                            if (A_tensor.dim() >= 3) {
                                // A format: [spec, batch, features]
                                A_spec = A_tensor.size(0);
                                A_batch = A_tensor.size(1);
                            } else if (A_tensor.dim() == 2) {
                                // A format: [spec, features], batch=1
                                A_spec = A_tensor.size(0);
                                A_batch = 1;
                            }
                        }
                        
                        // If 0D (scalar), expand to [spec, batch] format
                        if (bias.dim() == 0) {
                            if (A_spec > 0 && A_batch > 0) {
                                // Scalar -> expand to [spec, batch]
                                return bias.unsqueeze(0).unsqueeze(0).expand({A_spec, A_batch});
                            } else {
                                // No A context, expand to [1, 1]
                                return bias.unsqueeze(0).unsqueeze(0);
                            }
                        }
                        
                        // If 1D, need to determine if it's [spec] or [batch]
                        if (bias.dim() == 1) {
                            if (A_spec > 0 && A_batch > 0) {
                                // Use A matrix context to determine correct shape
                                if (bias.size(0) == A_spec) {
                                    // It's [spec], expand to [spec, batch]
                                    // [spec] -> unsqueeze(1) -> [spec, 1] -> expand -> [spec, batch]
                                    return bias.unsqueeze(1).expand({A_spec, A_batch});
                                } else if (bias.size(0) == A_batch) {
                                    // It's [batch], need to create [spec, batch]
                                    // [batch] -> unsqueeze(0) -> [1, batch] = [spec, batch] when spec=1
                                    // For A [1, 18, 50], bias [18] should become [1, 18]
                                    // unsqueeze(0) adds dimension at front: [18] -> [1, 18]
                                    torch::Tensor result = bias.unsqueeze(0); // [batch] -> [1, batch] = [1, 18]
                                    // If spec > 1, expand the spec dimension
                                    if (A_spec > 1) {
                                        result = result.expand({A_spec, A_batch}); // [1, 18] -> [spec, 18]
                                    }
                                    return result;
                                } else {
                                    // Doesn't match either - assume it's spec dimension, add batch=1
                                    // [spec] -> unsqueeze(1) -> [spec, 1]
                                    return bias.unsqueeze(1);
                                }
                            } else {
                                // No A context, just add batch dimension at the end
                                // [spec] -> unsqueeze(1) -> [spec, 1]
                                return bias.unsqueeze(1);
                            }
                        }
                        
                        // If 2D, check if dimensions match A matrix to infer correct order
                        if (bias.dim() == 2) {
                            if (A_spec > 0 && A_batch > 0) {
                                // Check if bias dimensions match A dimensions (in either order)
                                if (bias.size(0) == A_spec && bias.size(1) == A_batch) {
                                    // Already correct: [spec, batch]
                                    return bias;
                                } else if (bias.size(0) == A_batch && bias.size(1) == A_spec) {
                                    // Swapped: transpose to [spec, batch]
                                    return bias.transpose(0, 1);
                                } else {
                                    // Dimensions don't match A - this is an error case
                                    // Try to reshape if total elements match
                                    if (bias.numel() == A_spec * A_batch) {
                                        return bias.reshape({A_spec, A_batch});
                                    }
                                    // Otherwise return as-is and let error be caught later
                                    return bias;
                                }
                            } else {
                                // No A context, return as-is (assume already correct)
                                return bias;
                            }
                        }
                        
                        return bias;
                    };

                    // Get A matrix for context (use lA if available, otherwise uA)
                    BoundA A_for_context = currentLowerAlpha.defined() ? currentLowerAlpha : currentUpperAlpha;
                    
                    if (_lowerBias.exists(current)) {
                        auto cur = _lowerBias[current];
                        if (propagated_lbias.defined() && cur.defined()) {
                            // Normalize both to [spec, batch] format before comparison
                            // Use A matrix context to infer correct dimension order
                            torch::Tensor norm_propagated = normalize_bias(propagated_lbias, A_for_context);
                            torch::Tensor norm_cur = normalize_bias(cur, A_for_context);
                            
                            if (norm_propagated.sizes() != norm_cur.sizes()) {
                                std::ostringstream oss;
                                oss << "CROWNAnalysis::backwardFrom bias shape mismatch at node " << current
                                    << " (" << node->getNodeName().ascii() << "): "
                                    << "lbias=" << tensorShapeStr(propagated_lbias)
                                    << " (normalized: " << tensorShapeStr(norm_propagated) << ")"
                                    << " storedLowerBias=" << tensorShapeStr(cur)
                                    << " (normalized: " << tensorShapeStr(norm_cur) << ")";
                                throw std::runtime_error(oss.str());
                            }
                            propagated_lbias = norm_propagated + norm_cur;
                        } else {
                            propagated_lbias = cur.defined() ? normalize_bias(cur, A_for_context) : propagated_lbias;
                        }
                    } else {
                        propagated_lbias = normalize_bias(propagated_lbias, A_for_context);
                    }
                    
                    if (_upperBias.exists(current)) {
                        auto cur = _upperBias[current];
                        if (propagated_ubias.defined() && cur.defined()) {
                            // Normalize both to [spec, batch] format before comparison
                            // Use A matrix context to infer correct dimension order
                            torch::Tensor norm_propagated = normalize_bias(propagated_ubias, A_for_context);
                            torch::Tensor norm_cur = normalize_bias(cur, A_for_context);
                            
                            if (norm_propagated.sizes() != norm_cur.sizes()) {
                                std::ostringstream oss;
                                oss << "CROWNAnalysis::backwardFrom bias shape mismatch at node " << current
                                    << " (" << node->getNodeName().ascii() << "): "
                                    << "ubias=" << tensorShapeStr(propagated_ubias)
                                    << " (normalized: " << tensorShapeStr(norm_propagated) << ")"
                                    << " storedUpperBias=" << tensorShapeStr(cur)
                                    << " (normalized: " << tensorShapeStr(norm_cur) << ")";
                                throw std::runtime_error(oss.str());
                            }
                            propagated_ubias = norm_propagated + norm_cur;
                        } else {
                            propagated_ubias = cur.defined() ? normalize_bias(cur, A_for_context) : propagated_ubias;
                        }
                    } else {
                        propagated_ubias = normalize_bias(propagated_ubias, A_for_context);
                    }
                }

                addBias(inputIndex, propagated_lbias, propagated_ubias);
            }

            // Scheduling: mark that this current node has contributed to each dependency.
            const Vector<unsigned> &deps = _torchModel->getDependencies(current);
            for (unsigned inputIndex : deps) {
                if (!reachable.exists(inputIndex)) continue;
                if (pending.exists(inputIndex) && pending[inputIndex] > 0) {
                    pending[inputIndex] = pending[inputIndex] - 1;
                    if (pending[inputIndex] == 0) {
                        queue.push(inputIndex);
                    }
                }
            }
        }

        // Clean up A matrices and biases for nodes we're done with
        if (nodetype != NodeType::INPUT) {
            if (_lA.exists(current)) _lA.erase(current);
            if (_uA.exists(current)) _uA.erase(current);
            if (_lowerBias.exists(current)) _lowerBias.erase(current);
            if (_upperBias.exists(current)) _upperBias.erase(current);
        }
    }

    log("backwardFrom() - Completed.");
    } catch (const CommonError &e) {
        std::ostringstream oss;
        oss << "CROWNAnalysis::backwardFrom CommonError code=" << e.getCode()
            << " stage=" << stage
            << " startIndex=" << startIndex
            << " current=" << current_dbg;
        const char *msg = e.getUserMessage();
        if (msg && msg[0] != '\0') oss << " message=" << msg;
        throw std::runtime_error(oss.str());
    } catch (const Error &e) {
        std::ostringstream oss;
        oss << "CROWNAnalysis::backwardFrom Error class=" << e.getErrorClass()
            << " code=" << e.getCode()
            << " stage=" << stage
            << " startIndex=" << startIndex
            << " current=" << current_dbg;
        const char *msg = e.getUserMessage();
        if (msg && msg[0] != '\0') oss << " message=" << msg;
        throw std::runtime_error(oss.str());
    } catch (const std::exception &e) {
        throw std::runtime_error(std::string("CROWNAnalysis::backwardFrom exception stage=") + stage + ": " + e.what());
    }
}

void CROWNAnalysis::clearCrownState()
{
    _lA.clear();
    _uA.clear();
    _lowerBias.clear();
    _upperBias.clear();
}

void CROWNAnalysis::concretizeNode(unsigned startIndex, const Vector<unsigned>& unstableIndices)
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
        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(_torchModel->getDevice());
        inputLower = torch::zeros({(long)inputSize}, options);
        inputUpper = torch::ones({(long)inputSize}, options);
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
        
        if (!unstableIndices.empty()) {
            // If we computed sparse bounds, we need to scatter them into the full bounds tensor
            // Use IBP bounds as the base
            if (_ibpBounds.exists(startIndex)) {
                auto ibp = _ibpBounds[startIndex];
                if (ibp.lower().defined() && ibp.upper().defined()) {
                    // Detach to avoid breaking computation graph when modifying
                    torch::Tensor fullLower = ibp.lower().detach().clone();
                    torch::Tensor fullUpper = ibp.upper().detach().clone();
                 
                    // Create index tensor
                    auto indexOptions = torch::TensorOptions().dtype(torch::kLong).device(fullLower.device());
                    torch::Tensor indices = torch::tensor(
                        std::vector<int64_t>(unstableIndices.begin(), unstableIndices.end()),
                        indexOptions);
                    
                    // Flatten full bounds for scattering if needed, or use view
                    auto originalShape = fullLower.sizes();
                    torch::Tensor fullLowerFlat = fullLower.flatten();
                    torch::Tensor fullUpperFlat = fullUpper.flatten();
                    
                    // Scatter/Put - use non-in-place index_put to avoid breaking computation graph
                    // We want fullLowerFlat[indices] = concreteLower
                    // NOTE: Do NOT detach - we need to preserve gradients for Alpha-CROWN
                    torch::Tensor concreteLowerFlat = concreteLower.flatten();
                    torch::Tensor concreteUpperFlat = concreteUpper.flatten();
                    
                    // Use index_put (non-in-place) instead of index_put_ to avoid breaking gradients
                    // index_put returns a new tensor, so we need to assign it back
                    fullLowerFlat = fullLowerFlat.index_put({indices}, concreteLowerFlat);
                    fullUpperFlat = fullUpperFlat.index_put({indices}, concreteUpperFlat);
                    
                    // Reshape back
                    concreteLower = fullLowerFlat.reshape(originalShape);
                    concreteUpper = fullUpperFlat.reshape(originalShape);
                    
                    // log(Stringf("concretizeNode() - Scattered sparse CROWN bounds for %u neurons into IBP base", unstableIndices.size()));
                } else {
                    log(Stringf("concretizeNode() - WARNING: Sparse CROWN computed but no IBP base for node %u", startIndex));
                    // Cannot scatter without base size/values
                    return;
                }
            } else {
                log(Stringf("concretizeNode() - WARNING: Sparse CROWN computed but no IBP base for node %u", startIndex));
                return;
            }
        }

        // Intersecting with IBP bounds can make results UNSOUND if IBP bounds are incorrect
        // (e.g., due to shape/layout mismatches). For debugging/correctness, disable this.
        const bool kIntersectWithIBP = false;
        if (kIntersectWithIBP && _ibpBounds.exists(startIndex)) {
            auto ibp = _ibpBounds[startIndex];
            if (ibp.lower().defined() && ibp.upper().defined()) {
                concreteLower = torch::max(concreteLower, ibp.lower());
                concreteUpper = torch::min(concreteUpper, ibp.upper());
                log(Stringf("concretizeNode() - Intersected with IBP bounds for node %u", startIndex));
            }
        }

        BoundedTensor<torch::Tensor> concreteBounds(concreteLower, concreteUpper);
        _concreteBounds[startIndex] = concreteBounds;
        _torchModel->setConcreteBounds(startIndex, concreteBounds);
        log(Stringf("concretizeNode() - Stored CROWN concrete bounds for node %u", startIndex));

    } else {
        log(Stringf("concretizeNode() - CROWN bounds undefined, falling back to IBP for node %u", startIndex));
        if (_ibpBounds.exists(startIndex)) {
            _concreteBounds[startIndex] = _ibpBounds[startIndex];
            _torchModel->setConcreteBounds(startIndex, _ibpBounds[startIndex]);
        }
    }

    log(Stringf("concretizeNode() - Finished, _concreteBounds.exists(%u)=%d",
        startIndex, _concreteBounds.exists(startIndex)));
}

// Helper function for establishing consistent tensor format (following auto-LiRPA's _preprocess_C)
torch::Tensor CROWNAnalysis::preprocessC(const torch::Tensor& C, unsigned startIndex) {
    // Similar to auto_LiRPA's _preprocess_C function
    // Transforms C matrix to internal format: (spec, batch, *output_shape)
    // 
    // Input formats supported:
    //   - Empty tensor: creates identity matrix
    //   - 2D [batch, spec]: standard format, will be transposed and reshaped
    //   - 3D [num_specs, batch, output_dim]: already in spec-first format, just needs reshaping
    
    if (!_nodes.exists(startIndex)) {
        throw std::runtime_error("CROWNAnalysis::preprocessC: startIndex not found in nodes");
    }
    
    auto& startNode = _nodes[startIndex];
    unsigned outputSize = startNode->getOutputSize();
    
    // Ensure outputSize is valid
    if (outputSize == 0) {
        outputSize = 1;  // Use 1 as fallback to avoid division by zero
    }
    
    // If C is empty, create identity matrix
    if (C.numel() == 0) {
        // Create identity matrix in format (spec, batch, output)
        // For identity: spec_dim = outputSize, batch = 1
        // Shape: [outputSize, 1, outputSize]
        torch::Tensor identity = torch::eye(outputSize, torch::kFloat32); // [outputSize, outputSize]
        torch::Tensor result = identity.unsqueeze(1); // [outputSize, 1, outputSize]
        return result;
    }
    
    // Extract batch_size and output_dim from C shape
    // Following auto_LiRPA's _preprocess_C logic
    long batch_size, output_dim;
    
    if (C.dim() == 3) {
        
        bool isBatchSpecFormat = (C.size(0) == 1 && C.size(1) > 1);
        
        torch::Tensor C_processed;
        if (isBatchSpecFormat) {
            // C is in (batch, spec, output) format - transpose to (spec, batch, output)
            batch_size = C.size(0);
            output_dim = C.size(1);
            C_processed = C.transpose(0, 1); // (batch, spec, output) -> (spec, batch, output)
        } else {
            // C is already in (spec, batch, output) format - return as-is
            // Verify: first dim should be spec (numConstraints), second should be batch (usually 1)
            batch_size = C.size(1);
            output_dim = C.size(0);
            C_processed = C;
        }
        
        // For 3D C matrices, verify the shape is valid
        // The C matrix's output dimension (C.size(2)) should match the final output node's size
        if (C_processed.size(2) != (long)outputSize) {
            throw std::runtime_error(
                Stringf("CROWNAnalysis::preprocessC: C matrix output dimension (%ld) does not match node output size (%u)",
                        C_processed.size(2), outputSize).ascii());
        }
        
        // Return in (spec, batch, output) format for internal use
        return C_processed;
    } else if (C.dim() == 2) {
        // C has shape (batch, spec) - standard input format from user
        // Following auto_LiRPA: transpose to (spec, batch) and reshape
        batch_size = C.size(0);
        output_dim = C.size(1);
        
        torch::Tensor C_transformed = C.transpose(0, 1); // [spec, batch]
        // Reshape to (spec, batch, *output_shape)
        // For flat output, output_shape is [outputSize]
        std::vector<int64_t> new_shape = {output_dim, batch_size, (long)outputSize};
        return C_transformed.reshape(new_shape);
    } else if (C.dim() == 1) {
        // C has shape (spec) - add batch dimension and reshape
        batch_size = 1;
        output_dim = C.size(0);
        
        torch::Tensor C_transformed = C.unsqueeze(1); // [spec, 1]
        // Reshape to (spec, batch, *output_shape)
        std::vector<int64_t> new_shape = {output_dim, batch_size, (long)outputSize};
        return C_transformed.reshape(new_shape);
    } else {
        throw std::runtime_error("CROWNAnalysis::preprocessC: unsupported C matrix dimension");
    }
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
                    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(_torchModel->getDevice());
                    lower = torch::zeros({(long)outputSize}, options);
                    upper = torch::ones({(long)outputSize}, options);
                }
            } else {
                // First check for fixed intermediate bounds (alpha-CROWN best bounds tracking)
                auto fixedIt = _fixedConcreteBounds.find(inputIndex);
                if (fixedIt != _fixedConcreteBounds.end()) {
                    lower = fixedIt->second.first;
                    upper = fixedIt->second.second;
                }
                // Otherwise prefer CROWN concrete bounds in standard mode; otherwise IBP
                else if (LirpaConfiguration::USE_STANDARD_CROWN && _concreteBounds.exists(inputIndex)) {
                    lower = _concreteBounds[inputIndex].lower();
                    upper = _concreteBounds[inputIndex].upper();
                } else if (_ibpBounds.exists(inputIndex)) {
                    lower = _ibpBounds[inputIndex].lower();
                    upper = _ibpBounds[inputIndex].upper();
                } else {
                    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(_torchModel->getDevice());
                    lower = torch::zeros({(long)outputSize}, options);
                    upper = torch::ones({(long)outputSize}, options);
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
// A matrices in our codebase use [spec, batch, features] format
// But ensure3A expects [batch, spec, features] = [1, spec, n] format for bmm
static inline torch::Tensor ensure3A(const torch::Tensor& A) {
    if (!A.defined()) return A;
    if (A.dim() == 3) {
        // A matrices from backward propagation are in [spec, batch, features] format
        // We need to convert to [batch, spec, features] = [1, spec, features] for bmm
        
        // Detect format:
        // - If [spec, 1, features] where spec > 1: transpose to [1, spec, features]
        // - If [1, spec, features]: already correct, return as-is
        // - If [1, batch, features] where batch > 1: this is unusual, but keep as-is
        // - If [spec, batch, features] where batch > 1: transpose and take first batch
        
        torch::Tensor result;
        if (A.size(1) == 1 && A.size(0) > 1) {
            // [spec, 1, features] -> transpose to [1, spec, features]
            // This is the common case for specification matrices: [num_constraints, 1, output_dim]
            result = A.transpose(0, 1); // [spec, 1, features] -> [1, spec, features]
        } else if (A.size(0) == 1) {
            // Already [1, spec, features] or [1, batch, features] - assume correct format
            result = A;
        } else {
            // [spec, batch, features] where batch > 1
            // Transpose to [batch, spec, features], then take first batch
            torch::Tensor transposed = A.transpose(0, 1); // [spec, batch, features] -> [batch, spec, features]
            if (transposed.size(0) > 1) {
                // Take first batch: [batch, spec, features] -> [1, spec, features]
                result = transposed.narrow(0, 0, 1); // Take batch 0
            } else {
                result = transposed;
            }
        }
        return result;
    }
    if (A.dim() == 2) return A.unsqueeze(0);        // (1, spec, n)
    if (A.dim() == 1) return A.unsqueeze(0).unsqueeze(0);
    return A.unsqueeze(0); // best effort
}
static inline torch::Tensor ensure3x(const torch::Tensor& x) {
    // x is (n,) -> (1, n, 1); (b,n)->(b,n,1)
    // For bmm, we need [batch, n, 1] format
    if (!x.defined()) return x;
    if (x.dim() == 1) {
        // [n] -> [1, n, 1] for bmm
        return x.unsqueeze(0).unsqueeze(-1);
    }
    if (x.dim() == 2) {
        // [batch, n] -> [batch, n, 1] for bmm
        return x.unsqueeze(-1);
    }
    if (x.dim() == 3) {
        // Already [batch, n, 1] or similar
        return x;
    }
    // For other dimensions, try to reshape
    return x;
}
static inline torch::Tensor ensure3b(const torch::Tensor& b) {
    // b is (spec,) or (spec, batch) -> (1, spec, 1)
    // Our bias format is [spec, batch], need to convert to [1, spec, 1] for bmm
    if (!b.defined()) return b;
    if (b.dim() == 1) {
        // [spec] -> [1, spec, 1]
        return b.unsqueeze(0).unsqueeze(-1);
    }
    if (b.dim() == 2) {
        // [spec, batch] -> [1, spec, 1] (take first batch if batch > 1)
        if (b.size(1) > 1) {
            // Take first batch: [spec, batch] -> [spec] -> [1, spec, 1]
            return b.select(1, 0).unsqueeze(0).unsqueeze(-1);
        } else {
            // [spec, 1] -> [spec] -> [1, spec, 1]
            return b.squeeze(1).unsqueeze(0).unsqueeze(-1);
        }
    }
    if (b.dim() == 3) {
        // Already [1, spec, 1] or [batch, spec, 1]
        if (b.size(0) > 1) {
            // [batch, spec, 1] -> take first batch -> [1, spec, 1]
            return b.narrow(0, 0, 1);
        }
        return b;
    }
    return b;
}


torch::Tensor CROWNAnalysis::computeConcreteLowerBound(
    const torch::Tensor& lA, const torch::Tensor& lBias,
    const torch::Tensor& xLower, const torch::Tensor& xUpper)
{
    if (!lA.defined()) return torch::Tensor();


    torch::Tensor AL = ensure3A(lA.to(torch::kFloat32));         // (batch,spec,n) -> (1,spec,n) after ensure3A
    torch::Tensor xL = ensure3x(xLower.to(torch::kFloat32));     // (n,) -> (1,n,1)
    torch::Tensor xU = ensure3x(xUpper.to(torch::kFloat32));     // (n,) -> (1,n,1)
    torch::Tensor bL = ensure3b(lBias.to(torch::kFloat32));      // (spec,batch) -> (1,spec,1) after ensure3b


    if (AL.dim() != 3 || xL.dim() != 3 || xU.dim() != 3) {
        std::ostringstream oss;
        oss << "CROWNAnalysis::computeConcreteLowerBound - Shape error: "
            << "AL.dim()=" << AL.dim() << " xL.dim()=" << xL.dim() << " xU.dim()=" << xU.dim();
        throw std::runtime_error(oss.str());
    }

    torch::Tensor Apos = torch::clamp_min(AL, 0);
    torch::Tensor Aneg = torch::clamp_max(AL, 0);

    // LB = βL + Apos * xL + Aneg * xU
    // bmm: [batch, spec, n] @ [batch, n, 1] -> [batch, spec, 1]
    // Check that feature dimensions match
    if (AL.size(2) != xL.size(1)) {
        std::ostringstream oss;
        oss << "CROWNAnalysis::computeConcreteLowerBound - Feature dimension mismatch: "
            << "AL.size(2)=" << AL.size(2) << " (features) vs xL.size(1)=" << xL.size(1) << " (input size)";
        throw std::runtime_error(oss.str());
    }
    torch::Tensor term = Apos.bmm(xL) + Aneg.bmm(xU);            // (1,spec,1)
    torch::Tensor out  = term + bL;                              // (1,spec,1)
    
    torch::Tensor result = out.squeeze(-1).squeeze(0);           // (spec,)
    
    return result;
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
    
    torch::Tensor result = out.squeeze(-1).squeeze(0);           // (spec,)
    
    return result;
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
            // Try to normalize shapes by broadcasting or squeezing singleton dimensions
            torch::Tensor t1_norm = t1;
            torch::Tensor t2_norm = t2;
            
            // If one has more dimensions, try to broadcast the smaller one
            if (t1.dim() != t2.dim()) {
                if (t1.dim() > t2.dim()) {
                    // Try to expand t2 to match t1's dimensions
                    auto t1_sizes = t1.sizes().vec();
                    auto t2_sizes = t2.sizes().vec();
                    // Add singleton dimensions at the front
                    while (t2_sizes.size() < t1_sizes.size()) {
                        t2_sizes.insert(t2_sizes.begin(), 1);
                    }
                    t2_norm = t2.view(t2_sizes);
                } else {
                    // Try to expand t1 to match t2's dimensions
                    auto t1_sizes = t1.sizes().vec();
                    auto t2_sizes = t2.sizes().vec();
                    // Add singleton dimensions at the front
                    while (t1_sizes.size() < t2_sizes.size()) {
                        t1_sizes.insert(t1_sizes.begin(), 1);
                    }
                    t1_norm = t1.view(t1_sizes);
                }
            }
            
            // After normalization, check if they're compatible for broadcasting
            if (t1_norm.sizes() != t2_norm.sizes()) {
                // Try PyTorch's broadcasting
                try {
                    return BoundA(t1_norm + t2_norm);
                } catch (const std::exception& e) {
                    // If broadcasting fails, throw informative error
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
                        " vs A2.shape=" + shape2_str + " (after normalization, broadcasting failed: " +
                        std::string(e.what()) + ")");
                }
            }
            
            return BoundA(t1_norm + t2_norm);
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
    // Helper function to normalize bias shape to [spec, batch] format
    auto normalize_bias_shape = [](const torch::Tensor& bias) -> torch::Tensor {
        if (!bias.defined() || bias.numel() == 0) return bias;
        
        // If already 2D with shape [spec, batch], return as-is
        if (bias.dim() == 2) {
            return bias;
        }
        
        // If 1D [spec], expand to [spec, 1] (batch=1)
        if (bias.dim() == 1) {
            return bias.unsqueeze(1); // [spec] -> [spec, 1]
        }
        
        // If 0D (scalar), shouldn't happen but handle it
        if (bias.dim() == 0) {
            return bias.unsqueeze(0).unsqueeze(0); // scalar -> [1, 1]
        }
        
        // For higher dimensions, try to reshape to [spec, batch]
        // This shouldn't happen in normal flow, but handle edge cases
        if (bias.numel() > 0) {
            // Try to infer spec and batch dimensions
            // If last dimension is 1, it might be batch
            if (bias.size(-1) == 1 && bias.dim() > 1) {
                // Flatten all but last dimension as spec
                int64_t spec_size = bias.numel() / bias.size(-1);
                return bias.reshape({spec_size, 1});
            } else {
                // Assume all elements are spec, batch=1
                return bias.flatten().unsqueeze(1);
            }
        }
        
        return bias;
    };
    
    if (lBias.defined() && lBias.numel() > 0) {
        torch::Tensor normalized_lBias = normalize_bias_shape(lBias);
        
        if (_lowerBias.exists(nodeIndex)) {
            torch::Tensor existing = _lowerBias[nodeIndex];
            torch::Tensor normalized_existing = normalize_bias_shape(existing);
            
            // Check if shapes are compatible after normalization
            if (normalized_existing.sizes() != normalized_lBias.sizes()) {
                std::ostringstream oss;
                oss << "CROWNAnalysis::addBias: shape mismatch at node " << nodeIndex
                    << " after normalization: existing=" << tensorShapeStr(normalized_existing)
                    << " new=" << tensorShapeStr(normalized_lBias);
                throw std::runtime_error(oss.str());
            }
            
            _lowerBias[nodeIndex] = normalized_existing + normalized_lBias;
        } else {
            _lowerBias[nodeIndex] = normalized_lBias;
        }
    }
    if (uBias.defined() && uBias.numel() > 0) {
        torch::Tensor normalized_uBias = normalize_bias_shape(uBias);
        
        if (_upperBias.exists(nodeIndex)) {
            torch::Tensor existing = _upperBias[nodeIndex];
            torch::Tensor normalized_existing = normalize_bias_shape(existing);
            
            // Check if shapes are compatible after normalization
            if (normalized_existing.sizes() != normalized_uBias.sizes()) {
                std::ostringstream oss;
                oss << "CROWNAnalysis::addBias: shape mismatch at node " << nodeIndex
                    << " after normalization: existing=" << tensorShapeStr(normalized_existing)
                    << " new=" << tensorShapeStr(normalized_uBias);
                throw std::runtime_error(oss.str());
            }
            
            _upperBias[nodeIndex] = normalized_existing + normalized_uBias;
        } else {
            _upperBias[nodeIndex] = normalized_uBias;
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
}

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

// Determine which nodes need CROWN bounds (selective computation)
bool CROWNAnalysis::needsCROWNBounds(unsigned nodeIndex)
{
    if (!_nodes.exists(nodeIndex)) {
        return false;
    }

    auto& node = _nodes[nodeIndex];

    // Skip if already has valid concrete bounds
    if (_concreteBounds.exists(nodeIndex)) {
        return false;
    }

    // Skip constant/non-perturbed nodes (they don't need CROWN bounds)
    if (!node->isPerturbed()) {
        return false;
    }

    // For first linear layer connected to input, IBP provides equivalent tightness
    if (checkIBPFirstLinear(nodeIndex)) {
        // Use IBP bounds instead - they're equivalent for first linear layers
        if (_ibpBounds.exists(nodeIndex)) {
            _concreteBounds[nodeIndex] = _ibpBounds[nodeIndex];
            _torchModel->setConcreteBounds(nodeIndex, _ibpBounds[nodeIndex]);
        }
        return false;
    }

    // For ReLU with tight IBP bounds, IBP suffices
    if (node->getNodeType() == NodeType::RELU) {
        if (_ibpBounds.exists(nodeIndex)) {
            auto bounds = _ibpBounds[nodeIndex];
            // Check if bounds are reasonably tight (heuristic)
            // If many neurons are definitely active/inactive, IBP is good enough
            torch::Tensor lower = bounds.lower();
            torch::Tensor upper = bounds.upper();

            if (lower.defined() && upper.defined()) {
                // Count definitely active (lower > 0) and inactive (upper < 0) neurons
                float definitely_active = (lower > 0).sum().item<float>();
                float definitely_inactive = (upper < 0).sum().item<float>();
                float total = lower.numel();
                float stable_ratio = (definitely_active + definitely_inactive) / total;

                // If >80% neurons have stable status, IBP is sufficient
                if (stable_ratio > 0.8) {
                    _concreteBounds[nodeIndex] = _ibpBounds[nodeIndex];
                    _torchModel->setConcreteBounds(nodeIndex, _ibpBounds[nodeIndex]);
                    return false;
                }
            }
        }
    }

    // This node needs CROWN bounds for tightness
    return true;
}

// First linear layer IBP fast path optimization methods
bool CROWNAnalysis::checkIBPFirstLinear(unsigned nodeIndex)
{
    if (!LirpaConfiguration::ENABLE_FIRST_LINEAR_IBP) {
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
