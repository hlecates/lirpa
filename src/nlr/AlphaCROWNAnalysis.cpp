#include "AlphaCROWNAnalysis.h"
#include "BoundedReLUNode.h"

#include "Debug.h"
#include "FloatUtils.h"
#include "MStringf.h"
#include "LirpaError.h"
#include "TimeUtils.h"

#include <fstream>
#include <sstream>
#include <iomanip>

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

AlphaCROWNAnalysis::AlphaCROWNAnalysis(TorchModel* torchModel)
    : _torchModel(torchModel)
    , _alphaEnabled(true)
    , _initialized(false)
    , _iteration(1)
    , _learningRate(0.5f)
    , _optimizationStage("init")
{
    if (!_torchModel) {
        throw LirpaError(LirpaError::UNINITIALIZED_NODE, "AlphaCROWNAnalysis requires a valid TorchModel instance");
    }

    // Create CROWN analysis instance
    _crownAnalysis = std::make_unique<CROWNAnalysis>(_torchModel);
    //_crownAnalysis = std::make_unique<CROWNAnalysis>(_torchModel, false);
}

AlphaCROWNAnalysis::~AlphaCROWNAnalysis()
{
}

void AlphaCROWNAnalysis::initializeAlphaParameters()
{
    log("initializeAlphaParameters() - Starting alpha parameter initialization");
    
    if (_initialized) {
        log("initializeAlphaParameters() - Alpha parameters already initialized");
        return;
    }

    try {
        log("initializeAlphaParameters() - Forward pass");
        // network structure is already contained in the torch model, so this is just a pass through step
        performForwardPass();

        log("initializeAlphaParameters() - Prepare optimizable activations");
        // Collects the nodes that can be optimized --> for now this is only the relu layers
        prepareOptimizableActivations();

        log("initializeAlphaParameters() - Standard CROWN initialization pass");
        // Runs a CROWN pass to get the init CROWN slopes or alphas
        performCROWNInitializationPass();

        // NOTE: Alpha parameters are now created LAZILY via ensureAlphaFor() when needed
        // This allows per-(node,start) alpha with correct spec dimensions
        log("initializeAlphaParameters() - Alpha parameters will be created lazily per start");
        
        _initialized = true;
        log("initializeAlphaParameters() - Alpha parameter initialization completed successfully");
    } catch (const std::exception& e) {
        log(Stringf("initializeAlphaParameters() - Exception: %s", e.what()));
        throw;
    }
}

// TODO: check the auto grad functionality, what is it and where should we use it
void AlphaCROWNAnalysis::performForwardPass()
{
    // forward pass through the model to establish the network structure
    log("performForwardPass() - forward pass through model");
    
    // Enable autograd for the torch model to support gradient computations
    // This is crucial for the optimization phase where gradients flow through alpha parameters
    torch::GradMode::set_enabled(true);
    
    // The forward pass will be handled by the underlying CROWN analysis when we compute bounds so this just continues
    log("performForwardPass() - Forward pass structure established with autograd enabled");
}

void AlphaCROWNAnalysis::prepareOptimizableActivations()
{
    log("prepareOptimizableActivations() - Finding ReLU nodes for optimization");

    _optimizableNodes.clear();

    // Locate all optimizable activations in the network
    unsigned numNodes = _crownAnalysis->getNumNodes();
    for ( unsigned i = 0; i < numNodes; ++i )
    {
        auto node = _crownAnalysis->getNode(i);
        if ( node && node->getNodeType() == NodeType::RELU)
        {
            auto reluNode = std::dynamic_pointer_cast<BoundedReLUNode>(node);
            if ( reluNode )
            {
                _optimizableNodes.push_back(std::make_pair(i, reluNode));

                // Set optimization stage to 'init' for CROWN slope capture
                reluNode->setOptimizationStage("init");
                reluNode->setAlphaCrownAnalysis(this);

                log(Stringf("prepareOptimizableActivations() - Added ReLU node %u for optimization", i));
            }
        }
    }

    log(Stringf("prepareOptimizableActivations() - Found %u optimizable nodes", (unsigned)_optimizableNodes.size()));
}

void AlphaCROWNAnalysis::performCROWNInitializationPass()
{
    log("performCROWNInitializationPass() - Running standard CROWN to capture relaxation slopes");

    _crownAnalysis->setEnableFirstLinearIBP(false);
    
    _crownAnalysis->resetProcessingState();

    _crownAnalysis->run();

    log("performCROWNInitializationPass() - CROWN initialization pass completed");
}


void AlphaCROWNAnalysis::createAlphaParameters()
{
    log("createAlphaParameters() - Creating alpha parameters from CROWN slopes");

    for ( auto& nodePair : _optimizableNodes ) {
        unsigned nodeIndex = nodePair.first;
        auto node = nodePair.second;

        log(Stringf("createAlphaParameters() - Processing node %u", nodeIndex));

        unsigned outputSize = node->getOutputSize();
        if ( outputSize == 0 )
        {
            log(Stringf("createAlphaParameters() - Warning: Node %u has zero output size", nodeIndex));
            continue;
        }

        createAlphaForNode(nodeIndex, node, outputSize);
    }

    log("createAlphaParameters() - Alpha parameter creation completed");
}


void AlphaCROWNAnalysis::createAlphaForNode(unsigned nodeIndex, std::shared_ptr<BoundedAlphaOptimizeNode> node, unsigned outputSize)
{
    log(Stringf("createAlphaForNode() - Creating alpha parameters for node %u", nodeIndex));

    // Following auto_LiRPA's alpha structure:
    // alpha[start_node_name] has shape [2, spec_dim, batch_size, *neuron_shape]
    // Where:
    // - 2: upper/lower bound relaxations
    // - spec_dim: number of specifications (properties being verified)
    // - batch_size: number of verification queries (typically 1 for Marabou)
    // - neuron_shape: shape of the activation layer

    // Match auto_LiRPA default: do NOT share alpha across specs.
    // With identity C, specDim == model output size.
    unsigned specDim = _crownAnalysis->getOutputSize();
    unsigned batchSize = 1;

    // Create the alpha tensor without gradients first
    torch::Tensor alpha_init = torch::zeros({2, (long)specDim, (long)batchSize, (long)outputSize}, torch::dtype(torch::kFloat32));

    // Initialize alpha with CROWN slopes captured during init pass
    initializeAlphaWithCROWNSlopes(alpha_init, node, nodeIndex);

    // Now create the final alpha tensor with gradients enabled
    torch::Tensor alpha = alpha_init.clone().detach().requires_grad_(true);

    // Store alpha parameters
    AlphaParameters alphaParams;
    alphaParams.alpha = alpha;
    alphaParams.specDim = specDim;
    alphaParams.batchDim = batchSize;
    alphaParams.outDim = outputSize;
    alphaParams.requiresGrad = true;

    _alphaParameters[nodeIndex] = alphaParams;

    log(Stringf("createAlphaForNode() - Alpha parameters created for node %u with shape [%ld, %ld, %ld, %ld]", nodeIndex, alpha.size(0), alpha.size(1), alpha.size(2), alpha.size(3)));
}

void AlphaCROWNAnalysis::initializeAlphaWithCROWNSlopes(torch::Tensor& alpha, std::shared_ptr<BoundedAlphaOptimizeNode> node, unsigned nodeIndex)
{
    log(Stringf("initializeAlphaWithCROWNSlopes() - Initializing alpha for node %u", nodeIndex));
    log(Stringf("initializeAlphaWithCROWNSlopes() - Alpha tensor shape: [%ld, %ld, %ld, %ld]", alpha.size(0), alpha.size(1), alpha.size(2), alpha.size(3)));

    torch::Tensor lowerSlope = node->getCROWNSlope(true);   // dL (CROWN lower face choice)

    if (lowerSlope.defined()) {
        log(Stringf("initializeAlphaWithCROWNSlopes() - Lower slope dims: %ld", lowerSlope.dim()));

        // Expand to [1, 1, output]
        auto dL = lowerSlope.view({1, 1, -1});

        // Initialize α based on CROWN's lower-face selection:
        // - If dL = 0 (CROWN chose y ≥ 0), set α = 0
        // - If dL = 1 (CROWN chose y ≥ x), set α = 1
        // This preserves the CROWN baseline exactly at iter-0
        auto alpha0 = dL.clone();  // α directly equals the CROWN lower slope (0 or 1)

        // For the lower-bound α slice (index 0), use CROWN initialization
        alpha.index_put_({0}, alpha0);

        // For the upper-bound α slice (index 1), set to 0 (not used in lower-only runs)
        alpha.index_put_({1}, torch::zeros_like(alpha0));

        log(Stringf("initializeAlphaWithCROWNSlopes() - Initialized alpha from CROWN dual sign for node %u", nodeIndex));
    } else {
        // initialize with default values if CROWN slopes not available
        log(Stringf("initializeAlphaWithCROWNSlopes() - Warning: CROWN slopes not available for node %u, using default initialization", nodeIndex));

        // Default: choose y ≥ 0 face (α = 0)
        alpha.index_put_({0}, torch::zeros({1, 1, alpha.size(3)}));
        alpha.index_put_({1}, torch::zeros({1, 1, alpha.size(3)}));
    }
}

// Ensure alpha exists for (node, start) pair with correct shape and initialization
AlphaParameters& AlphaCROWNAnalysis::ensureAlphaFor(
    unsigned nodeIndex,
    const std::string& startKey,
    int specDim, int outDim,
    const torch::Tensor& input_lb,
    const torch::Tensor& input_ub)
{
    // Determine if this is for lower or upper bound based on current optimization stage
    // This is a simplified approach - in dual optimization, this will be called separately for each bound
    bool isLower = (_config.bound_side == BoundSide::Lower);

    // printf("[DEBUG] ensureAlphaFor node=%u, startKey=%s, bound_side=%s\n",
    //        nodeIndex, startKey.c_str(), isLower ? "LOWER" : "UPPER");

    // Choose the appropriate storage based on bound side
    auto& perStart = isLower ? _alphaByNodeStartLower[nodeIndex] : _alphaByNodeStartUpper[nodeIndex];
    auto it = perStart.find(startKey);

    bool need_new = (it == perStart.end());
    if (!need_new) {
        const auto& a = it->second.alpha;
        // Updated shape check: now [specDim, 1, outDim] instead of [2, specDim, 1, outDim]
        if (a.size(0) != specDim || a.size(2) != outDim) {
            need_new = true;
        }
    }

    if (need_new) {
        // Create alpha tensor WITHOUT gradients first
        // Shape: [specDim, 1, outDim] (no leading "2" dimension)
        torch::Tensor alpha = torch::zeros({specDim, 1, outDim}, torch::kFloat32);

        // Initialize alpha based on bound side
        torch::Tensor slope_init;
        {
            // Find the node by index and read its CROWN slope
            std::shared_ptr<BoundedAlphaOptimizeNode> nodePtr;
            for (auto &p : _optimizableNodes) {
                if (p.first == nodeIndex) {
                    nodePtr = p.second;
                    break;
                }
            }
            if (nodePtr) {
                // Both lower and upper bounds: use CROWN lower-face choice (0 or 1) as initialization
                // This matches auto-LiRPA's approach where both alpha slices start from CROWN initialization
                slope_init = nodePtr->getCROWNSlope(true);  // [out], entries 0 or 1
            } else {
                // Safe fallback
                auto always_active = input_lb.ge(0);
                auto always_inactive = input_ub.le(0);
                (void)always_inactive;
                slope_init = torch::where(always_active, torch::ones_like(input_lb), torch::zeros_like(input_lb));
            }
        }

        // Broadcast slope to [spec, 1, out]
        auto alpha_init = slope_init.view({1, 1, outDim}).expand({specDim, 1, outDim});
        alpha.copy_(alpha_init);

        // CROWN DEBUG: Log alpha initialization (to file)
        {
            std::ostringstream oss;
            auto node = _torchModel->getNode(nodeIndex);
            std::string nodeName = node ? node->getNodeName().ascii() : std::to_string(nodeIndex);
            
            oss << "\n[CROWN ALPHA INIT] " << nodeName << ", start=" << startKey
                << ", bound=" << (isLower ? "LOWER" : "UPPER") << "\n";
            oss << "  Alpha (first 10): " << tensorFirstN(alpha_init) << "\n";
            oss << "  Alpha: min=" << std::fixed << std::setprecision(6)
                << alpha_init.min().item<float>()
                << ", max=" << alpha_init.max().item<float>()
                << ", mean=" << alpha_init.mean().item<float>()
                << ", shape=[" << specDim << "," << 1L << "," << outDim << "]\n";
            
            writeToDebugLog(oss.str());
        }

        // NOW enable gradients after initialization
        alpha = alpha.detach().requires_grad_(true);

        AlphaParameters params;
        params.alpha = alpha;
        params.specDim = specDim;
        params.outDim = outDim;
        perStart[startKey] = std::move(params);
    }
    return perStart[startKey];
}

// NEW REFACTORED ENTRY METHOD - Returns optimized bounds for specified side
torch::Tensor AlphaCROWNAnalysis::computeOptimizedBounds(BoundSide side)
{
    log(Stringf("computeOptimizedBounds() - Starting optimization for %s bounds",
                side == BoundSide::Lower ? "LOWER" : "UPPER"));

    // Initialize alpha parameters if not already done
    if (!_initialized) {
        log("computeOptimizedBounds() - Initializing alpha parameters");
        initializeAlphaParameters();
    }

    // Check if alpha optimization should be performed
    if (!shouldPerformOptimization()) {
        log("computeOptimizedBounds() - Alpha optimization not applicable, falling back to standard CROWN");
        _crownAnalysis->run();
        return side == BoundSide::Lower ? extractLowerBoundsFromCROWN() : extractUpperBoundsFromCROWN();
    }

    // Set the bound side in config
    _config.bound_side = side;
    bool isLower = (side == BoundSide::Lower);

    // Reset alpha parameters from CROWN slopes for this bound side
    resetAlphasFromCROWNSlopes(isLower);

    // Set optimization stage
    setOptimizationStage("opt");

    // Disable first linear IBP for alpha optimization
    _crownAnalysis->setEnableFirstLinearIBP(false);

    // Run one CROWN pass to trigger lazy alpha parameter creation
    // This is necessary because alphas are created on-demand during backward pass
    _crownAnalysis->clearConcreteBounds();
    _crownAnalysis->resetProcessingState();
    _crownAnalysis->run();

    // Create optimizer for this bound side
    auto alphaParams = collectAlphaParameters(isLower);

    // DEBUG: Print initial alpha values and dtypes (disabled for clean output)
    // if (!alphaParams.empty()) {
    //     printf("[DEBUG Alpha Init] Alpha parameter diagnostics:\n");
    //     auto& storageMap = isLower ? _alphaByNodeStartLower : _alphaByNodeStartUpper;
    //     for (auto& [nodeIdx, perStart] : storageMap) {
    //         for (auto& [startKey, ap] : perStart) {
    //             if (ap.alpha.defined()) {
    //                 printf("  Node %u [%s]: shape=[%lld,%lld,%lld], dtype=%s, requires_grad=%d, min=%.6f, max=%.6f, mean=%.6f\n",
    //                        nodeIdx, startKey.c_str(),
    //                        (long long)ap.alpha.size(0), (long long)ap.alpha.size(1), (long long)ap.alpha.size(2),
    //                        torch::toString(ap.alpha.dtype()).c_str(),
    //                        ap.alpha.requires_grad(),
    //                        ap.alpha.min().item<float>(),
    //                        ap.alpha.max().item<float>(),
    //                        ap.alpha.mean().item<float>());
    //             }
    //         }
    //     }
    // }

    if (alphaParams.empty()) {
        printf("[ERROR] No alpha parameters found! Alpha-CROWN optimization cannot proceed.\n");
        printf("[ERROR] This usually means ReLU nodes were not properly initialized.\n");
        log("computeOptimizedBounds() - No alpha parameters found, returning CROWN bounds");
        _crownAnalysis->run();
        return side == BoundSide::Lower ? extractLowerBoundsFromCROWN() : extractUpperBoundsFromCROWN();
    }

    auto optimizer = std::make_shared<torch::optim::Adam>(
        alphaParams,
        torch::optim::AdamOptions(_learningRate)
            .betas(std::make_tuple(0.9, 0.999))
            .eps(1e-8)
    );

    // Track best bounds
    torch::Tensor bestBounds;
    float currentLR = _learningRate;

    printf("[Alpha-CROWN %s] Optimizing with %u iterations, %zu params, lr=%.3f\n",
           isLower ? "LOWER" : "UPPER", _iteration, alphaParams.size(), currentLR);

    // Optimization loop
    for (unsigned iter = 0; iter < _iteration; ++iter) {
        log(Stringf("computeOptimizedBounds() - %s iteration %u/%u",
                    isLower ? "LOWER" : "UPPER", iter + 1, _iteration));

        // Enable/disable gradients based on iteration
        std::unique_ptr<torch::NoGradGuard> no_grad;
        if (iter == _iteration - 1) {
            no_grad = std::make_unique<torch::NoGradGuard>();
        }

        // Clear previous bounds and compute with current alphas
        _crownAnalysis->clearConcreteBounds();
        _crownAnalysis->resetProcessingState();
        _crownAnalysis->run();

        // Extract current bounds
        torch::Tensor currentLower = extractLowerBoundsFromCROWN();
        torch::Tensor currentUpper = extractUpperBoundsFromCROWN();
        torch::Tensor currentBound = isLower ? currentLower : currentUpper;

        // Compute loss for all iterations to include in summary
        torch::Tensor loss;
        if (iter < _iteration - 1) {
            loss = computeLoss(currentLower, currentUpper, isLower);
        }

        // Update best bounds
        bool improved = false;
        if (isLower) {
            improved = !bestBounds.defined() || (currentBound > bestBounds).any().item<bool>();
            if (improved) {
                bestBounds = bestBounds.defined() ? torch::max(bestBounds, currentBound) : currentBound.clone();
                updateBestAlphas({0}, isLower);
            }
        } else {
            improved = !bestBounds.defined() || (currentBound < bestBounds).any().item<bool>();
            if (improved) {
                bestBounds = bestBounds.defined() ? torch::min(bestBounds, currentBound) : currentBound.clone();
                updateBestAlphas({0}, isLower);
            }
        }

        // Print one-line summary for each iteration
        printf("[%s iter %2u] L:[%.6f] U:[%.6f] loss:%.6f lr:%.6f%s\n",
               isLower ? "LOWER" : "UPPER", iter,
               currentLower[0].item<float>(), currentUpper[0].item<float>(),
               loss.defined() ? loss.item<float>() : 0.0f,
               currentLR,
               improved ? " *improved*" : "");

        // CROWN DEBUG: Print alpha values for this iteration (to file)
        {
            std::ostringstream oss;
            oss << "\n[CROWN ALPHA VALUES - Iteration " << iter << "]\n";
            
            // Get the appropriate storage based on bound side
            auto& storageMap = isLower ? _alphaByNodeStartLower : _alphaByNodeStartUpper;
            
            for (auto& [nodeIdx, perStart] : storageMap) {
                for (auto& [startKey, ap] : perStart) {
                    if (ap.alpha.defined()) {
                        auto node = _torchModel->getNode(nodeIdx);
                        std::string nodeName = node ? node->getNodeName().ascii() : std::to_string(nodeIdx);
                        
                        oss << "  Layer " << nodeName << ", Alpha '" << startKey << "':\n";
                        oss << "    Values (first 10): " << tensorFirstN(ap.alpha) << "\n";
                        oss << "    Stats: mean=" << std::fixed << std::setprecision(6)
                            << ap.alpha.mean().item<float>()
                            << ", min=" << ap.alpha.min().item<float>()
                            << ", max=" << ap.alpha.max().item<float>()
                            << ", shape=[";
                        auto shape = ap.alpha.sizes();
                        for (size_t i = 0; i < shape.size(); ++i) {
                            if (i > 0) oss << ", ";
                            oss << shape[i];
                        }
                        oss << "]\n";
                    }
                }
            }
            writeToDebugLog(oss.str());
        }

        // Gradient step (except on last iteration)
        if (iter < _iteration - 1 && loss.defined()) {
            // Gradient verification: Check if loss has gradient function attached
            if (!loss.grad_fn()) {
                printf("[WARNING] Alpha-CROWN iter %u: Loss tensor has no grad_fn! Gradients will not flow.\n", iter);
                printf("[WARNING] This usually means weight matrices don't have requires_grad=True.\n");
            }

            optimizer->zero_grad();
            loss.backward();

            // Diagnostic: Verify gradients are actually computed
            bool has_gradients = false;
            float total_grad_norm = 0.0f;
            int params_with_grad = 0;
            for (const auto& param : alphaParams) {
                if (param.grad().defined() && param.grad().numel() > 0) {
                    has_gradients = true;
                    total_grad_norm += param.grad().norm().item<float>();
                    params_with_grad++;
                }
            }

            if (!has_gradients && iter == 0) {
                printf("[ERROR] Alpha-CROWN iter %u: No gradients computed after backward()!\n", iter);
                printf("[ERROR] Alpha optimization will not work. Check weight matrix properties.\n");
            }

            optimizer->step();

            // Clip alphas to [0,1]
            clipAlphaParameters();

            // Learning rate decay
            currentLR *= _config.lr_decay;
            for (auto& group : optimizer->param_groups()) {
                if (auto adamGroup = dynamic_cast<torch::optim::AdamOptions*>(&group.options())) {
                    adamGroup->lr(currentLR);
                }
            }
        }
    }

    // Restore best alphas and compute final bounds
    restoreBestAlphas(isLower);
    _crownAnalysis->clearConcreteBounds();
    _crownAnalysis->resetProcessingState();
    _crownAnalysis->run();

    torch::Tensor finalBound = isLower ? extractLowerBoundsFromCROWN() : extractUpperBoundsFromCROWN();
    if (bestBounds.defined()) {
        finalBound = isLower ? torch::max(finalBound, bestBounds) : torch::min(finalBound, bestBounds);
    }

    log(Stringf("computeOptimizedBounds() - Optimization completed for %s bounds",
                isLower ? "LOWER" : "UPPER"));

    return finalBound;
}

// DEPRECATED - OLD IMPLEMENTATIONS
// std::pair<torch::Tensor, torch::Tensor> AlphaCROWNAnalysis::computeBoundsWithAlpha(BoundSide side)
// {
//     log(Stringf("computeBoundsWithAlpha() - Computing bounds for %s",
//                 side == BoundSide::Lower ? "LOWER" : "UPPER"));
// 
//     // Set the bound side in config (used by ensureAlphaFor to select correct alpha storage)
//     _config.bound_side = side;
// 
//     // Set optimization stage to enable alpha-based relaxations
//     setOptimizationStage("opt");
// 
//     // Disable first linear IBP for alpha optimization
//     _crownAnalysis->setEnableFirstLinearIBP(false);
// 
//     // Clear previous concrete bounds and reset processing state
//     _crownAnalysis->clearConcreteBounds();
//     _crownAnalysis->resetProcessingState();
// 
//     // Run CROWN analysis with current alpha parameters
//     _crownAnalysis->run();
// 
//     // Extract bounds from CROWN analysis
//     torch::Tensor lowerBounds = extractLowerBoundsFromCROWN();
//     torch::Tensor upperBounds = extractUpperBoundsFromCROWN();
// 
//     log(Stringf("computeBoundsWithAlpha() - Bounds computed for %s",
//                 side == BoundSide::Lower ? "LOWER" : "UPPER"));
// 
//     return std::make_pair(lowerBounds, upperBounds);
// }
// 
// // DEPRECATED - OLD DUAL LOOP IMPLEMENTATION
// // std::pair<torch::Tensor, torch::Tensor> AlphaCROWNAnalysis::run()
// // {
// //     log("run() - Starting simplified Alpha-CROWN analysis (Marabou integration mode)");
// //
// //     // Call the full run() method with both bounds computed but optimizing lower only
// //     // Following auto-LiRPA approach: always compute both, but optimize based on bound_side
// //     return run(nullptr, nullptr, true, true);
// // }
// //
// // std::pair<torch::Tensor, torch::Tensor> AlphaCROWNAnalysis::run(
// //     const torch::Tensor* input,
// //     const torch::Tensor* specificationMatrix,
// //     bool computeLowerBounds,
// //     bool computeUpperBounds)
// // {
// //     (void)input;
// //     (void)specificationMatrix;
// //
// //     log("run() - Starting Alpha-CROWN analysis");
// //
// //     try {
// //         // Initialize alpha parameters if not already done
// //         if (!_initialized) {
// //             log("run() - Initializing alpha parameters");
// //             initializeAlphaParameters();
// //         } else {
// //             log("run() - Alpha parameters already initialized, skipping initialization");
// //         }
// //
// //         // Check if alpha optimization should be performed
// //         if ( !shouldPerformOptimization() ) {
// //             log("run() - Alpha optimization not applicable, falling back to standard CROWN");
// //             _crownAnalysis->run();
// //
// //             // TODO: expose the exctraction of these bounds
// //             // Since we are making a standalone tool, whould probably centralize this in the torchModel
// //         }
// //
// //         // TODO: Expose some way of getting input bounds, Again mirror the torchmodel idea, should store them globally in the torch model
// //
// //         // Run PGD optimization loop
// //         log("run() - Starting PGD optimization loop");
// //
// //         // Store original first linear IBP setting to restore later
// //         bool originalFirstLinearIBP = _crownAnalysis->getEnableFirstLinearIBP();
// //
// //         // Reset for optimization phase
// //         resetForOptimization();
// //
// //         // TODO: Create/check default tensors
// //         torch::Tensor defaultInput, defaultSpec;
// //         const torch::Tensor* actualInput = input;
// //         const torch::Tensor* actualSpec = specificationMatrix;
// //         if (!actualInput) {
// //             // Create placeholder input tensor - actual input comes from Marabou's constraint system
// //             defaultInput = torch::zeros({1, 1}); // Minimal placeholder
// //             actualInput = &defaultInput;
// //         }
// //         if (!actualSpec) {
// //             // Create placeholder specification matrix - actual specs come from Marabou's query
// //             defaultSpec = torch::eye(1); // Identity matrix placeholder
// //             actualSpec = &defaultSpec;
// //         }
// //
// //         // Run the main optimization loop
// //         auto result = runOptimizationLoop(*actualInput, *actualSpec, computeLowerBounds, computeUpperBounds);
// //
// //         log("run() - Finalizing optimization");
// //         setOptimizationStage("reuse");
// //
// //         // Restore original first linear IBP setting
// //         _crownAnalysis->setEnableFirstLinearIBP(originalFirstLinearIBP);
// //
// //         log("run() - Alpha-CROWN analysis completed successfully");
// //         return result;
// //
// //     } catch (const std::exception& e) {
// //         log(Stringf("run() - Exception during Alpha-CROWN analysis: %s", e.what()));
// //         throw;
// //     }
// // }
// 
// // DEPRECATED - Optimization loop now in TorchModel
// std::pair<torch::Tensor, torch::Tensor> AlphaCROWNAnalysis::runOptimizationLoop(
//     const torch::Tensor& input,
//     const torch::Tensor& specificationMatrix,
//     bool computeLowerBounds,
//     bool computeUpperBounds)
// {
//     (void)input;
//     (void)specificationMatrix;
//     (void)computeLowerBounds;
//     (void)computeUpperBounds;
// 
//     log("runOptimizationLoop() - Starting DUAL-SIDED Alpha-CROWN PGD optimization");
// 
//     // Initialize best alpha tracking storage (alphas will be tracked during optimization)
//     _bestAlphaByNodeStartLower.clear();
//     _bestAlphaByNodeStartUpper.clear();
// 
//     log("runOptimizationLoop() - Pass 1: Optimizing upper bounds");
// 
//     // Set bound_side for upper bound optimization
//     _config.bound_side = BoundSide::Upper;
// 
//     // Reset upper bound alpha parameters from CROWN slopes
//     // This ensures a clean initialization for the upper bound pass
//     resetAlphasFromCROWNSlopes(false); // false = upper
// 
//     // Best bounds tracking for upper pass
//     torch::Tensor bestUpperBounds;
//     torch::Tensor currentLowerBounds, currentUpperBounds;
// 
//     // Reset learning rate for upper bound pass
//     float originalLR = _learningRate;
//     _learningRate = originalLR;
// 
//     // Upper bound optimization loop
//     for (unsigned iteration = 0; iteration < _iteration; ++iteration) {
//         log(Stringf("runOptimizationLoop() - UPPER iter %u/%u", iteration + 1, _iteration));
// 
//         // Enable alpha optimization
//         setOptimizationStage("opt");
//         for (auto& nodePair : _optimizableNodes) {
//             nodePair.second->setAlphaCrownAnalysis(this);
//         }
//         _crownAnalysis->setEnableFirstLinearIBP(false);
// 
//         // Enable NoGradGuard for the last iteration
//         std::unique_ptr<torch::NoGradGuard> no_grad_guard;
//         if (iteration == _iteration - 1) {
//             no_grad_guard = std::make_unique<torch::NoGradGuard>();
//         }
// 
//         // Compute bounds
//         _crownAnalysis->clearConcreteBounds();
//         _crownAnalysis->resetProcessingState();
//         _crownAnalysis->run();
// 
//         // Create optimizer after first iteration (alphas are now created lazily)
//         if (iteration == 0) {
//             _optimizerUpper = createOptimizerUpper();
//             _optimizerUpper->zero_grad();
//         }
// 
//         currentLowerBounds = extractLowerBoundsFromCROWN();
//         currentUpperBounds = extractUpperBoundsFromCROWN();
// 
//         // Check if upper bound improved
//         bool ub_improved = currentUpperBounds.defined() &&
//                            (!bestUpperBounds.defined() ||
//                             (currentUpperBounds < bestUpperBounds).any().item<bool>());
// 
//         if (ub_improved) {
//             bestUpperBounds = bestUpperBounds.defined()
//                 ? torch::min(bestUpperBounds, currentUpperBounds)
//                 : currentUpperBounds.clone();
//             // Update best alpha parameters for upper bound
//             updateBestAlphas({0}, false); // false = upper
//             // Snapshot best intermediate bounds (update from upper pass)
//             snapshotBestIntermediateBounds();
//         }
// 
//         // Compute loss and perform gradient step (except on last iteration)
//         torch::Tensor loss;
//         if (iteration < _iteration - 1) {
//             loss = computeLoss(currentLowerBounds, currentUpperBounds, false); // false = optimize upper
// 
//             if (loss.defined() && _optimizationStage == "opt") {
//                 performGradientStep(_optimizerUpper, loss);
//                 decayLearningRate();
//                 updateOptimizerLearningRate(_optimizerUpper);
//             }
//         }
// 
//         // Print iteration summary
//         printf("[UPPER %u] ub=[%.6f, %.6f]",
//                iteration,
//                currentUpperBounds.defined() ? currentUpperBounds.min().item<float>() : 0.0f,
//                currentUpperBounds.defined() ? currentUpperBounds.max().item<float>() : 0.0f);
//         if (loss.defined()) {
//             printf(", loss=%.6f", loss.item<float>());
//         }
//         if (_optimizationStage == "opt") {
//             printf(", lr=%.6f", _learningRate);
//         }
//         printf("\n");
//     }
// 
//     // Final Upper Bound Pass (with best alpha)
//     restoreBestAlphas(false); // false = upper
// 
//     _crownAnalysis->setEnableFirstLinearIBP(false);
//     _crownAnalysis->clearConcreteBounds();
//     _crownAnalysis->resetProcessingState();
//     _crownAnalysis->run();
// 
//     torch::Tensor finalUpper = extractUpperBoundsFromCROWN();
//     if (bestUpperBounds.defined()) {
//         finalUpper = torch::min(finalUpper, bestUpperBounds);
//     }
// 
//     printf("[UPPER FINAL] ub=[%.6f, %.6f]\n",
//            finalUpper.min().item<float>(), finalUpper.max().item<float>());
// 
//     log("runOptimizationLoop() - Pass 2: Optimizing lower bounds");
// 
//     // Set bound_side for lower bound optimization
//     _config.bound_side = BoundSide::Lower;
// 
//     // Reset lower bound alpha parameters from CROWN slopes
//     // This ensures a clean initialization for the lower bound pass, independent of the upper bound pass
//     resetAlphasFromCROWNSlopes(true); // true = lower
// 
//     // Best bounds tracking for lower pass
//     torch::Tensor bestLowerBounds;
// 
//     // Reset learning rate for lower bound pass
//     _learningRate = originalLR;
// 
//     // Lower bound optimization loop
//     for (unsigned iteration = 0; iteration < _iteration; ++iteration) {
//         log(Stringf("runOptimizationLoop() - LOWER iter %u/%u", iteration + 1, _iteration));
// 
//         // Enable alpha optimization
//         setOptimizationStage("opt");
//         for (auto& nodePair : _optimizableNodes) {
//             nodePair.second->setAlphaCrownAnalysis(this);
//         }
//         _crownAnalysis->setEnableFirstLinearIBP(false);
// 
//         // Enable NoGradGuard for the last iteration
//         std::unique_ptr<torch::NoGradGuard> no_grad_guard;
//         if (iteration == _iteration - 1) {
//             no_grad_guard = std::make_unique<torch::NoGradGuard>();
//         }
// 
//         // Compute bounds
//         _crownAnalysis->clearConcreteBounds();
//         _crownAnalysis->resetProcessingState();
//         _crownAnalysis->run();
// 
//         // Create optimizer after first iteration (alphas are now created lazily)
//         if (iteration == 0) {
//             _optimizerLower = createOptimizerLower();
//             _optimizerLower->zero_grad();
//         }
// 
//         currentLowerBounds = extractLowerBoundsFromCROWN();
//         currentUpperBounds = extractUpperBoundsFromCROWN();
// 
//         // Check if lower bound improved
//         bool lb_improved = currentLowerBounds.defined() &&
//                            (!bestLowerBounds.defined() ||
//                             (currentLowerBounds > bestLowerBounds).any().item<bool>());
// 
//         if (lb_improved) {
//             bestLowerBounds = bestLowerBounds.defined()
//                 ? torch::max(bestLowerBounds, currentLowerBounds)
//                 : currentLowerBounds.clone();
//             // Update best alpha parameters for lower bound
//             updateBestAlphas({0}, true); // true = lower
//             // Snapshot best intermediate bounds
//             snapshotBestIntermediateBounds();
//         }
// 
//         // Compute loss and perform gradient step (except on last iteration)
//         torch::Tensor loss;
//         if (iteration < _iteration - 1) {
//             loss = computeLoss(currentLowerBounds, currentUpperBounds, true); // true = optimize lower
// 
//             if (loss.defined() && _optimizationStage == "opt") {
//                 performGradientStep(_optimizerLower, loss);
//                 decayLearningRate();
//                 updateOptimizerLearningRate(_optimizerLower);
//             }
//         }
// 
//         // Print iteration summary
//         printf("[LOWER %u] lb=[%.6f, %.6f]",
//                iteration,
//                currentLowerBounds.defined() ? currentLowerBounds.min().item<float>() : 0.0f,
//                currentLowerBounds.defined() ? currentLowerBounds.max().item<float>() : 0.0f);
//         if (loss.defined()) {
//             printf(", loss=%.6f", loss.item<float>());
//         }
//         if (_optimizationStage == "opt") {
//             printf(", lr=%.6f", _learningRate);
//         }
//         printf("\n");
//     }
// 
//     // Final Lower Bound Pass (with best alpha)
//     restoreBestAlphas(true); // true = lower
// 
//     _crownAnalysis->setEnableFirstLinearIBP(false);
//     _crownAnalysis->clearConcreteBounds();
//     _crownAnalysis->resetProcessingState();
//     _crownAnalysis->run();
// 
//     torch::Tensor finalLower = extractLowerBoundsFromCROWN();
//     if (bestLowerBounds.defined()) {
//         finalLower = torch::max(finalLower, bestLowerBounds);
//     }
// 
//     printf("[LOWER FINAL] lb=[%.6f, %.6f]\n",
//            finalLower.min().item<float>(), finalLower.max().item<float>());
// 
//     log("runOptimizationLoop() - DUAL-SIDED optimization completed");
//     printf("\n========== DUAL-SIDED OPTIMIZATION COMPLETE ==========\n");
//     printf("[FINAL RESULT] lb=[%.6f, %.6f], ub=[%.6f, %.6f]\n\n",
//            finalLower.min().item<float>(), finalLower.max().item<float>(),
//            finalUpper.min().item<float>(), finalUpper.max().item<float>());
// 
//     return std::make_pair(finalLower, finalUpper);
// }
// //

// Extract lower bounds from CROWN analysis (following auto_LiRPA approach)
torch::Tensor AlphaCROWNAnalysis::extractLowerBoundsFromCROWN()
{
    // Get output layer concrete bounds
    unsigned outputIndex = _crownAnalysis->getOutputIndex();

    if (_crownAnalysis->hasConcreteBounds(outputIndex)) {
        torch::Tensor lowerBounds = _crownAnalysis->getConcreteLowerBound(outputIndex);

        // DEBUG: Log bound extraction (disabled for clean output)
        // static int extract_count = 0;
        // if (extract_count++ % 10 == 0) {  // Log every 10th extraction to reduce spam
        //     printf("[DEBUG extractLowerBounds] Output index %u: dtype=%s, shape=[%lld], min=%.6f, max=%.6f\n",
        //            outputIndex,
        //            torch::toString(lowerBounds.dtype()).c_str(),
        //            (long long)lowerBounds.size(0),
        //            lowerBounds.min().item<float>(),
        //            lowerBounds.max().item<float>());
        // }

        return lowerBounds;
    }

    // Fallback to IBP bounds if concrete bounds not available
    if (_crownAnalysis->hasIBPBounds(outputIndex)) {
        torch::Tensor lowerBounds = _crownAnalysis->getIBPLowerBound(outputIndex);
        printf("[DEBUG extractLowerBounds] Using IBP bounds (fallback)\n");
        return lowerBounds;
    }

    // Return placeholder if no bounds available
    printf("[DEBUG extractLowerBounds] No bounds available, returning zeros\n");
    return torch::zeros({1});
}

// Extract upper bounds from CROWN analysis (used for tracking and reference bounds)
// Following auto_LiRPA approach - these bounds are tracked even when not directly optimized
torch::Tensor AlphaCROWNAnalysis::extractUpperBoundsFromCROWN()
{
    // Get output layer concrete bounds
    unsigned outputIndex = _crownAnalysis->getOutputIndex();
    
    if (_crownAnalysis->hasConcreteBounds(outputIndex)) {
        torch::Tensor upperBounds = _crownAnalysis->getConcreteUpperBound(outputIndex);
        return upperBounds;
    }
    
    // Fallback to IBP bounds if concrete bounds not available
    if (_crownAnalysis->hasIBPBounds(outputIndex)) {
        torch::Tensor upperBounds = _crownAnalysis->getIBPUpperBound(outputIndex);
        return upperBounds;
    }
    
    // Return placeholder if no bounds available
    return torch::zeros({1});
}

// DEPRECATED - Optimizer creation now in TorchModel
// std::shared_ptr<torch::optim::Optimizer> AlphaCROWNAnalysis::createOptimizerLower()
// {
//     log("createOptimizerLower() - Creating Adam optimizer for lower bound alpha parameters");
// 
//     auto alphaParams = collectAlphaParameters(true); // true = lower
//     if (alphaParams.empty()) {
//         throw NLRError(NLRError::LAYER_TYPE_NOT_SUPPORTED,
//                       "No alpha parameters found for lower bound optimization");
//     }
// 
//     auto optimizer = std::make_shared<torch::optim::Adam>(
//         alphaParams,
//         torch::optim::AdamOptions(_learningRate)
//             .betas(std::make_tuple(0.9, 0.999))
//             .eps(1e-8)
//     );
// 
//     log(Stringf("createOptimizerLower() - Created Adam optimizer with LR=%.3f for %u parameter tensors",
//                _learningRate, (unsigned)alphaParams.size()));
// 
//     return optimizer;
// }
// 
// std::shared_ptr<torch::optim::Optimizer> AlphaCROWNAnalysis::createOptimizerUpper()
// {
//     log("createOptimizerUpper() - Creating Adam optimizer for upper bound alpha parameters");
// 
//     auto alphaParams = collectAlphaParameters(false); // false = upper
// 
//     // printf("[DEBUG] createOptimizerUpper: found %u alpha parameter tensors\n", (unsigned)alphaParams.size());
//     // printf("[DEBUG] _alphaByNodeStartUpper size: %u\n", (unsigned)_alphaByNodeStartUpper.size());
// 
//     if (alphaParams.empty()) {
//         throw NLRError(NLRError::LAYER_TYPE_NOT_SUPPORTED,
//                       "No alpha parameters found for upper bound optimization");
//     }
// 
//     auto optimizer = std::make_shared<torch::optim::Adam>(
//         alphaParams,
//         torch::optim::AdamOptions(_learningRate)
//             .betas(std::make_tuple(0.9, 0.999))
//             .eps(1e-8)
//     );
// 
//     log(Stringf("createOptimizerUpper() - Created Adam optimizer with LR=%.3f for %u parameter tensors",
//                _learningRate, (unsigned)alphaParams.size()));
// 
//     return optimizer;
// }
// //

std::vector<torch::Tensor> AlphaCROWNAnalysis::collectAlphaParameters(bool isLower)
{
    std::vector<torch::Tensor> params;

    // Choose the appropriate storage based on bound side
    auto& storageMap = isLower ? _alphaByNodeStartLower : _alphaByNodeStartUpper;

    // Walk the nested map
    for (auto& [nodeIdx, perStart] : storageMap) {
        for (auto& [startKey, ap] : perStart) {
            if (ap.alpha.requires_grad()) {
                params.push_back(ap.alpha);
            }
        }
    }

    log(Stringf("collectAlphaParameters(%s) - Collected %u alpha parameter tensors",
                isLower ? "LOWER" : "UPPER", (unsigned)params.size()));
    return params;
}

// DEPRECATED - Learning rate update now in TorchModel
// void AlphaCROWNAnalysis::updateOptimizerLearningRate(std::shared_ptr<torch::optim::Optimizer>& optimizer)
// {
//     // Update the learning rate in Adam optimizer (following auto_LiRPA's exponential decay)
//     auto adamOptimizer = std::dynamic_pointer_cast<torch::optim::Adam>(optimizer);
//     if (adamOptimizer) {
//         // Update the learning rate in the optimizer's options
//         for (auto& group : adamOptimizer->param_groups()) {
//             if (auto adamGroup = dynamic_cast<torch::optim::AdamOptions*>(&group.options())) {
//                 adamGroup->lr(_learningRate);
//             }
//         }
//     }
// }
// //

// DEPRECATED - Best alpha tracking simplified
// void AlphaCROWNAnalysis::setupBestAlphaTracking()
// {
//     log("setupBestAlphaTracking() - Setting up best alpha value tracking for both lower and upper bounds");
// 
//     _bestAlphaByNodeStartLower.clear();
//     _bestAlphaByNodeStartUpper.clear();
// 
//     // Clone current lower alpha values as initial "best" values
//     for (auto& [nodeIdx, perStart] : _alphaByNodeStartLower) {
//         for (auto& [startKey, ap] : perStart) {
//             AlphaParameters copy = ap;
//             copy.alpha = ap.alpha.detach().clone();
//             _bestAlphaByNodeStartLower[nodeIdx][startKey] = std::move(copy);
//         }
//     }
// 
//     // Clone current upper alpha values as initial "best" values
//     for (auto& [nodeIdx, perStart] : _alphaByNodeStartUpper) {
//         for (auto& [startKey, ap] : perStart) {
//             AlphaParameters copy = ap;
//             copy.alpha = ap.alpha.detach().clone();
//             _bestAlphaByNodeStartUpper[nodeIdx][startKey] = std::move(copy);
//         }
//     }
// 
//     log(Stringf("setupBestAlphaTracking() - Initialized best alpha tracking for both bounds"));
// }
// //


void AlphaCROWNAnalysis::updateBestAlphas(const std::vector<int>& improvedIndices, bool isLower)
{
    (void)improvedIndices;

    log(Stringf("updateBestAlphas(%s) - Updating best alphas for ALL neurons (total loss improved)",
                isLower ? "LOWER" : "UPPER"));

    // Choose the appropriate storage based on bound side
    auto& currentStorage = isLower ? _alphaByNodeStartLower : _alphaByNodeStartUpper;
    auto& bestStorage = isLower ? _bestAlphaByNodeStartLower : _bestAlphaByNodeStartUpper;

    // Copy all current α to best when total loss improves
    for (auto& [nodeIdx, perStart] : currentStorage) {
        for (auto& [startKey, ap] : perStart) {
            // Create AlphaParameters if it doesn't exist in bestStorage
            if (bestStorage[nodeIdx].find(startKey) == bestStorage[nodeIdx].end()) {
                AlphaParameters copy = ap;
                copy.alpha = ap.alpha.detach().clone();
                bestStorage[nodeIdx][startKey] = std::move(copy);
            } else {
                bestStorage[nodeIdx][startKey].alpha = ap.alpha.detach().clone();
            }
        }
    }
}

void AlphaCROWNAnalysis::restoreBestAlphas(bool isLower)
{
    log(Stringf("restoreBestAlphas(%s) - Restoring all alpha parameters to best found values",
                isLower ? "LOWER" : "UPPER"));

    // Choose the appropriate storage based on bound side
    auto& currentStorage = isLower ? _alphaByNodeStartLower : _alphaByNodeStartUpper;
    auto& bestStorage = isLower ? _bestAlphaByNodeStartLower : _bestAlphaByNodeStartUpper;

    for (auto& [nodeIdx, perStart] : currentStorage) {
        for (auto& [startKey, ap] : perStart) {
            ap.alpha.data().copy_(bestStorage[nodeIdx][startKey].alpha.data());
        }
    }
}

void AlphaCROWNAnalysis::snapshotBestIntermediateBounds()
{
    log("snapshotBestIntermediateBounds() - Capturing coherent point-in-time snapshot of PRE-ACTIVATION bounds");

    // Clear previous snapshot and store entire current state as a coherent snapshot
    // This ensures the restored bounds represent a real state that actually occurred,
    // not an element-wise merge that may never have existed
    _bestIntermediateBounds.clear();

    unsigned numNodesSnapshoted = 0;

    // Iterate through all nodes and capture PRE-ACTIVATION bounds (inputs to ReLU)
    // The secant for ReLU uses the pre-activation [l,u], not post-activation
    unsigned numNodes = _crownAnalysis->getNumNodes();
    for (unsigned nodeIdx = 0; nodeIdx < numNodes; ++nodeIdx) {
        auto node = _crownAnalysis->getNode(nodeIdx);
        if (!node) continue;

        // Only snapshot bounds for nodes that are INPUTS to ReLU layers
        // (i.e., the pre-activation bounds that determine the ReLU secant)
        bool isPreActivation = false;

        // Check if this node feeds into a ReLU
        for (const auto& [reluIdx, reluNode] : _optimizableNodes) {
            // Get dependencies of the ReLU node
            auto deps = _torchModel->getDependencies(reluIdx);
            for (unsigned depIdx : deps) {
                if (depIdx == nodeIdx) {
                    isPreActivation = true;
                    break;
                }
            }
            if (isPreActivation) break;
        }

        if (!isPreActivation) continue;

        if (_crownAnalysis->hasConcreteBounds(nodeIdx)) {
            torch::Tensor lower = _crownAnalysis->getConcreteLowerBound(nodeIdx);
            torch::Tensor upper = _crownAnalysis->getConcreteUpperBound(nodeIdx);

            _bestIntermediateBounds[nodeIdx] = std::make_pair(lower.detach().clone(), upper.detach().clone());
            numNodesSnapshoted++;
        }
    }

    log(Stringf("snapshotBestIntermediateBounds() - Captured coherent snapshot for %u pre-activation nodes",
                numNodesSnapshoted));
}

void AlphaCROWNAnalysis::restoreBestIntermediateBounds()
{
    log("restoreBestIntermediateBounds() - Seeding best PRE-ACTIVATION bounds into CROWN analysis");

    if (_bestIntermediateBounds.empty()) {
        return;
    }

    // Seed the best pre-activation bounds into CROWNAnalysis
    // These will be used by getInputBoundsForNode() to compute tighter ReLU secants
    for (const auto& [nodeIdx, bounds] : _bestIntermediateBounds) {
        _crownAnalysis->setFixedConcreteBounds(nodeIdx, bounds.first, bounds.second);
    }

    log(Stringf("restoreBestIntermediateBounds() - Seeded bounds for %u pre-activation nodes",
                (unsigned)_bestIntermediateBounds.size()));
}

torch::Tensor AlphaCROWNAnalysis::computeLoss(const torch::Tensor& lowerBounds,
                                              const torch::Tensor& upperBounds,
                                              bool optimizeLower,
                                              c10::optional<torch::Tensor> stop_mask_opt /* (batch,1) bool */) {

    (void)stop_mask_opt;
    // Expect lowerBounds/upperBounds shaped (batch, spec) or (spec) for 1D case
    // Stage 1: spec-level reduction (sum over spec dim=1, keepdim=true).
    auto options = lowerBounds.defined() ? lowerBounds.options()
                                         : (upperBounds.defined() ? upperBounds.options()
                                                                  : torch::TensorOptions().dtype(torch::kFloat32));
    torch::Tensor loss_per_elem;  // shape (batch, 1) or scalar

    if (optimizeLower) {
        if (!lowerBounds.defined() || lowerBounds.numel() == 0) {
            return torch::zeros({1}, options); // safe scalar 0
        }
        // Handle both 1D and 2D tensors
        if (lowerBounds.dim() == 1) {
            // 1D tensor: just sum all elements
            loss_per_elem = -lowerBounds.sum().unsqueeze(0);  // make it 1D with size [1]
        } else {
            // 2D tensor: sum over spec dimension
            auto l = lowerBounds.sum(/*dim=*/1, /*keepdim=*/true);  // (batch,1)
            loss_per_elem = -l;  // total_loss = -1 * l
        }
    } else { // Upper
        if (!upperBounds.defined() || upperBounds.numel() == 0) {
            return torch::zeros({1}, options);
        }
        // Handle both 1D and 2D tensors
        if (upperBounds.dim() == 1) {
            // 1D tensor: just sum all elements
            loss_per_elem = upperBounds.sum().unsqueeze(0);  // make it 1D with size [1]
        } else {
            // 2D tensor: sum over spec dimension
            auto u = upperBounds.sum(/*dim=*/1, /*keepdim=*/true);  // (batch,1)
            loss_per_elem = u;  // total_loss = -1 * (-u) = u
        }
    }

    auto loss = loss_per_elem.sum();  // scalar

    return loss;
}

// DEPRECATED - Gradient step now in TorchModel
// void AlphaCROWNAnalysis::performGradientStep(std::shared_ptr<torch::optim::Optimizer>& optimizer, const torch::Tensor& loss)
// {
//     // Clear gradients
//     optimizer->zero_grad();
// 
//     // Backward pass to compute gradients
//     loss.backward();
// 
//     // Update parameters
//     optimizer->step();
// 
//     // Immediately clamp alpha parameters to [0,1] after optimizer step
//     // Following auto_LiRPA pattern where clip_alpha() is called right after optimization
//     {
//         torch::NoGradGuard no_grad;
//         // Clamp lower bound alpha parameters
//         for (auto& [nodeIdx, perStart] : _alphaByNodeStartLower) {
//             for (auto& [startKey, ap] : perStart) {
//                 ap.alpha.data().clamp_(0.0f, 1.0f);
//             }
//         }
//         // Clamp upper bound alpha parameters
//         for (auto& [nodeIdx, perStart] : _alphaByNodeStartUpper) {
//             for (auto& [startKey, ap] : perStart) {
//                 ap.alpha.data().clamp_(0.0f, 1.0f);
//             }
//         }
//     }
// }
// //

std::vector<int> AlphaCROWNAnalysis::findImprovedIndices(const torch::Tensor& currentBounds, const torch::Tensor& bestBounds, bool isLowerBound){

    std::vector<int> improvedIndices;

    if ( !currentBounds.defined() || !bestBounds.defined() || currentBounds.sizes() != bestBounds.sizes() )
    {
        return improvedIndices;
    }

    torch::Tensor improved;
    if( isLowerBound )
    {
        improved = currentBounds > bestBounds;
    }
    else
    {
        improved = currentBounds < bestBounds;
    }

    auto improvedMask = improved.nonzero().squeeze();

    if ( improvedMask.numel() > 0 )
    {
        if (improvedMask.dim() == 0) {
            // Handle scalar case (single improved index)
            int index = improvedMask.item<int>();
            improvedIndices.push_back(index);
        } else {
            // Handle tensor case (multiple improved indices)
            for ( int64_t i = 0; i < improvedMask.size(0); ++i )
            {
                int index = improvedMask[i].item<int>();
                improvedIndices.push_back(index);
            }
        }
    }

    return improvedIndices;
}

bool AlphaCROWNAnalysis::shouldPerformOptimization() const
{
    if ( !_alphaEnabled ) 
    {
        return false;
    }
    
    if ( !_initialized && _optimizableNodes.empty() ) 
    {
        // If not initialized yet, check if we have a network with ReLU nodes
        unsigned numNodes = _crownAnalysis->getNumNodes();
        for ( unsigned i = 0; i < numNodes; ++i ) 
        {
            auto node = _crownAnalysis->getNode(i);
            if ( node && node->getNodeType() == NodeType::RELU ) 
            {
                return true; // Found at least one ReLU node
            }
        }
        return false; // No ReLU nodes found
    }
    
    // If initialized, check if we have optimizable nodes
    return !_optimizableNodes.empty();
}

//------ Helpers (getters/setters/"changers")------

torch::Tensor AlphaCROWNAnalysis::getAlphaForNode(unsigned nodeIndex, bool isLowerBound, 
                                                 unsigned specIndex, unsigned batchIndex) const
{
    auto it = _alphaParameters.find(nodeIndex);
    if (it == _alphaParameters.end()) {
        return torch::Tensor();
    }
    
    const AlphaParameters& params = it->second;
    
    // Select appropriate alpha: 0 for lower bound, 1 for upper bound
    int alphaIndex = isLowerBound ? 0 : 1;

    // Extract alpha for specific specification and batch
    // alpha has shape [2, spec_dim, batch_size, output_size]
    // Add bounds checking to prevent index out of range errors
    if (alphaIndex >= params.alpha.size(0)) {
        alphaIndex = 0;
    }
    if (specIndex >= (unsigned)params.alpha.size(1)) {
        specIndex = 0;
    }
    if (batchIndex >= (unsigned)params.alpha.size(2)) {
        batchIndex = 0;
    }

    torch::Tensor selectedAlpha = params.alpha[alphaIndex][specIndex][batchIndex];
    
    return selectedAlpha;
}

// Fetch alpha slice for ALL specs at once for a specific start.
// Returns [spec, out] tensor for use at multiply time
torch::Tensor AlphaCROWNAnalysis::getAlphaForNodeAllSpecs(
    unsigned nodeIndex,
    bool isLower,
    const std::string& startKey,
    int specDim,
    int outDim,
    const torch::Tensor& input_lb,
    const torch::Tensor& input_ub)
{
    (void)isLower; // Unused - we use _config.bound_side instead in dual-sided optimization

    // In dual-sided optimization, we use _config.bound_side to determine which alpha storage to use
    // The isLower parameter indicates which bound is being computed, but we always use alphas
    // from the current optimization side (stored in _config.bound_side)
    // This ensures that during upper bound optimization, we use upper alphas even when computing lower bounds

    // Ensure per-(node,start) α exists with correct shape & initialization
    // This will use _config.bound_side (set by the optimization loop) to choose the correct storage
    auto& ap = ensureAlphaFor(nodeIndex, startKey, specDim, outDim, input_lb, input_ub);

    // ap.alpha: [spec, 1, out] -> drop batch dimension
    return ap.alpha.select(/*dim=*/1, /*batch=*/0); // [spec, out]
}

torch::Tensor AlphaCROWNAnalysis::getAllAlphaParameters() const
{
    if (_alphaParameters.empty()) {
        return torch::Tensor();
    }
    
    // Collect all alpha tensors for optimization
    std::vector<torch::Tensor> alphas;
    for (const auto& alphaPair : _alphaParameters) {
        alphas.push_back(alphaPair.second.alpha);
    }
    
    // Concatenate all alpha parameters (this might need adjustment based on optimizer requirements)
    return torch::cat(alphas, 0);
}

void AlphaCROWNAnalysis::setOptimizationStage(const std::string& stage)
{
    _optimizationStage = stage;
    
    // Propagate stage to all optimizable nodes
    for (auto& nodePair : _optimizableNodes) {
        auto node = nodePair.second;
        node->setOptimizationStage(stage);
    }
    
    log(Stringf("setOptimizationStage() - Set optimization stage to '%s'", stage.c_str()));
}

bool AlphaCROWNAnalysis::hasAlphaParameters(unsigned nodeIndex) const
{
    return _alphaParameters.find(nodeIndex) != _alphaParameters.end();
}

unsigned AlphaCROWNAnalysis::getNumOptimizableNodes() const
{
    return static_cast<unsigned>(_optimizableNodes.size());
}

std::vector<unsigned> AlphaCROWNAnalysis::getOptimizableNodeIndices() const
{
    std::vector<unsigned> indices;
    indices.reserve(_optimizableNodes.size());
    
    for (const auto& nodePair : _optimizableNodes) {
        indices.push_back(nodePair.first);
    }
    
    return indices;
}

void AlphaCROWNAnalysis::clipAlphaParameters()
{
    log("clipAlphaParameters() - Clipping alpha parameters to valid ranges [0, 1]");

    // Following auto_LiRPA pattern: clamp ALL alpha values to [0,1] regardless of optimization side
    // This prevents numerical instability for all bounds

    // Clip lower bound alpha parameters
    for (auto& [nodeIdx, perStart] : _alphaByNodeStartLower) {
        for (auto& [startKey, ap] : perStart) {
            ap.alpha.data().clamp_(0.0f, 1.0f);
        }
    }

    // Clip upper bound alpha parameters
    for (auto& [nodeIdx, perStart] : _alphaByNodeStartUpper) {
        for (auto& [startKey, ap] : perStart) {
            ap.alpha.data().clamp_(0.0f, 1.0f);
        }
    }

    log("clipAlphaParameters() - Alpha parameter clipping completed for both bounds");
}

void AlphaCROWNAnalysis::resetForOptimization()
{
    log("resetForOptimization() - Resetting analysis state for optimization");

    // Set optimization stage to 'opt' to enable alpha-based relaxations
    setOptimizationStage("opt");

    // Enable gradients for all lower bound alpha parameters
    for (auto& [nodeIdx, perStart] : _alphaByNodeStartLower) {
        for (auto& [startKey, ap] : perStart) {
            ap.alpha.requires_grad_(true);
        }
    }

    // Enable gradients for all upper bound alpha parameters
    for (auto& [nodeIdx, perStart] : _alphaByNodeStartUpper) {
        for (auto& [startKey, ap] : perStart) {
            ap.alpha.requires_grad_(true);
        }
    }

    log("resetForOptimization() - Analysis state reset for optimization (both bounds)");
}

void AlphaCROWNAnalysis::resetAlphasFromCROWNSlopes(bool isLower)
{
    log(Stringf("resetAlphasFromCROWNSlopes(%s) - Re-initializing alpha parameters from CROWN slopes",
                isLower ? "LOWER" : "UPPER"));

    // Choose the appropriate storage based on bound side
    auto& alphaStorage = isLower ? _alphaByNodeStartLower : _alphaByNodeStartUpper;

    // Clear all existing alpha parameters for this bound side
    // This ensures we start with a clean slate, preventing contamination from previous passes
    alphaStorage.clear();

    log(Stringf("resetAlphasFromCROWNSlopes(%s) - Cleared %s alpha storage, alphas will be re-created lazily",
                isLower ? "LOWER" : "UPPER", isLower ? "lower" : "upper"));

    // Alphas will be re-created lazily during the optimization pass via ensureAlphaFor()
    // with fresh CROWN slope initialization
}

CROWNAnalysis* AlphaCROWNAnalysis::getCROWNAnalysis() const
{
    return _crownAnalysis.get();
}

TorchModel* AlphaCROWNAnalysis::getTorchModel() const
{
    return _torchModel;
}

// Configuration synchronization (following auto_LiRPA approach)
void AlphaCROWNAnalysis::updateFromConfig()
{
    _alphaEnabled = _config.enable_alpha_crown;
    _iteration = _config.iteration;
    _learningRate = _config.lr_alpha;
    
    log(Stringf("updateFromConfig() - Updated from config: enable=%s, iterations=%u, lr=%.3f", 
               _alphaEnabled ? "true" : "false", _iteration, _learningRate));
}

void AlphaCROWNAnalysis::log(const String& message)
{
    (void)message;
    if (GlobalConfiguration::NETWORK_LEVEL_REASONER_LOGGING) {
        //printf("AlphaCROWNAnalysis: %s\n", message.ascii());
    }
}

} // namespace NLR
  