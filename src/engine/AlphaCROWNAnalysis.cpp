#include "AlphaCROWNAnalysis.h"
#include "nodes/BoundedAlphaOptimizedNode.h"
#include "nodes/BoundedSigmoidNode.h"

#include "Debug.h"
#include "MStringf.h"
#include "LunaError.h"
#include "TimeUtils.h"

#include <sstream>
#include <iomanip>
#include <cmath>
#include <limits>
#include <unordered_map>

namespace NLR {

AlphaCROWNAnalysis::AlphaCROWNAnalysis(TorchModel* torchModel)
    : _torchModel(torchModel)
    , _alphaEnabled(true)
    , _initialized(false)
    , _iteration(LunaConfiguration::ALPHA_ITERATIONS)
    , _learningRate(LunaConfiguration::ALPHA_LR)
    , _optimizationStage("init")
{
    if (!_torchModel) {
        throw LunaError(LunaError::UNINITIALIZED_NODE, "AlphaCROWNAnalysis requires a valid TorchModel instance");
    }

    // Create CROWN analysis instance
    _crownAnalysis = std::make_unique<CROWNAnalysis>(_torchModel);
    
    // Initialize from LunaConfiguration
    updateFromConfig();
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
    log("prepareOptimizableActivations() - Finding alpha-optimizable activation nodes");

    _optimizableNodes.clear();

    // Locate all optimizable activations in the network
    unsigned numNodes = _crownAnalysis->getNumNodes();
    for ( unsigned i = 0; i < numNodes; ++i )
    {
        auto node = _crownAnalysis->getNode(i);
        if ( !node )
        {
            continue;
        }

        auto alphaNode = std::dynamic_pointer_cast<BoundedAlphaOptimizeNode>(node);
        if ( alphaNode )
        {
            _optimizableNodes.push_back(std::make_pair(i, alphaNode));

            // Set optimization stage to 'init' for CROWN slope capture
            alphaNode->setOptimizationStage("init");
            alphaNode->setAlphaCrownAnalysis(this);

            log(Stringf("prepareOptimizableActivations() - Added optimizable node %u (type=%d)", i, (int)node->getNodeType()));
        }
    }

    log(Stringf("prepareOptimizableActivations() - Found %u optimizable nodes", (unsigned)_optimizableNodes.size()));
}

void AlphaCROWNAnalysis::performCROWNInitializationPass()
{
    log("performCROWNInitializationPass() - Running standard CROWN to capture relaxation slopes");

    LunaConfiguration::ENABLE_FIRST_LINEAR_IBP = false;
    
    _crownAnalysis->resetProcessingState();

    _crownAnalysis->run(false); // No gradients needed for initialization pass

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
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(_torchModel->getDevice());
    torch::Tensor alpha_init = torch::zeros({2, (long)specDim, (long)batchSize, (long)outputSize}, options);

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
        auto zeroOptions = alpha.options();
        alpha.index_put_({0}, torch::zeros({1, 1, alpha.size(3)}, zeroOptions));
        alpha.index_put_({1}, torch::zeros({1, 1, alpha.size(3)}, zeroOptions));
    }
}

// Ensure alpha exists for (node, start) pair with correct shape and initialization
// FIXED: Only create alpha for unstable neurons (matching auto-LiRPA approach)
AlphaParameters& AlphaCROWNAnalysis::ensureAlphaFor(
    unsigned nodeIndex,
    const std::string& startKey,
    int specDim, int outDim,
    const torch::Tensor& input_lb,
    const torch::Tensor& input_ub)
{
    // Determine if this is for lower or upper bound based on current optimization stage
    bool isLower = (LunaConfiguration::BOUND_SIDE == LunaConfiguration::BoundSide::Lower);

    // Choose the appropriate storage based on bound side
    auto& perStart = isLower ? _alphaByNodeStartLower[nodeIndex] : _alphaByNodeStartUpper[nodeIndex];
    auto it = perStart.find(startKey);

    // Compute unstable mask.
    // ReLU-like activations use cross-zero instability.
    // Sigmoid/tanh-like activations benefit from alpha optimization on all
    // perturbed neurons (non-trivial interval width), not only cross-zero ones.
    // IMPORTANT: Detach input bounds to prevent computation graph from being retained
    // across iterations. The input bounds come from CROWN computations which may have
    // gradient tracking. If we don't detach, the unstableMask and unstableIndices
    // would be part of the computation graph, causing "backward through graph a second time"
    // errors when these indices are reused in subsequent iterations.
    auto input_lb_flat = input_lb.detach().flatten();
    auto input_ub_flat = input_ub.detach().flatten();
    torch::Tensor unstableMask;
    auto nodePtr = _crownAnalysis->getNode(nodeIndex);
    if (nodePtr && nodePtr->getNodeType() == NodeType::SIGMOID) {
        unstableMask = (input_ub_flat - input_lb_flat) > 1e-6;
    } else {
        unstableMask = (input_lb_flat < 0) & (input_ub_flat > 0);
    }
    int numUnstable = unstableMask.sum().item<int>();
    bool need_new = (it == perStart.end());
    if (!need_new) {
        const auto& ap = it->second;
        bool spec_mismatch = false;
        if (ap.alpha.defined()) {
            const auto storedSpecDim = (int)ap.alpha.size(0);
            // Treat storedSpecDim == specDim + 1 as compatible when default slot is used.
            spec_mismatch = (storedSpecDim != specDim)
                && !(ap.hasSpecDefaultSlot && storedSpecDim == specDim + 1);
        }
        if (spec_mismatch) {
            // Spec dimension mismatch requires re-creation.
            perStart.erase(it);
            need_new = true;
        } else {
            // Keep existing alpha even if numUnstable changes to avoid shape churn
            // during optimization (closer to auto_LiRPA's fixed alpha indices).
            return it->second;
        }
    }

    if (need_new) {
        // Use options without gradient tracking for creating intermediate tensors
        auto options = input_lb.options().dtype(torch::kFloat32).requires_grad(false);

        AlphaParameters params;
        params.specDim = specDim;
        params.outDim = outDim;
        params.numUnstable = numUnstable;
        // Detach the mask to ensure no gradient tracking (should already be detached since
        // it's computed from detached input bounds, but be explicit)
        params.unstableMask = unstableMask.detach().clone();

        if (numUnstable == 0) {
            // No unstable neurons - create empty alpha tensor
            if (nodePtr && nodePtr->getNodeType() == NodeType::SIGMOID) {
                params.alpha = torch::empty({specDim, 1, 0, 8}, options);
            } else {
                params.alpha = torch::empty({specDim, 1, 0}, options);
            }
            params.unstableIndices = torch::empty({0}, torch::kLong);
            perStart[startKey] = std::move(params);
            return perStart[startKey];
        }

        // Get indices of unstable neurons
        // Use detach() to ensure no computation graph references are retained
        params.unstableIndices = torch::nonzero(unstableMask).flatten().to(torch::kLong).detach();

        // Initialize alpha based on CROWN slopes (only for unstable neurons)
        torch::Tensor slope_init;
        torch::Tensor full_slope_debug;
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
                // Get CROWN lower-face choice (0 or 1) as initialization
                torch::Tensor full_slope = nodePtr->getCROWNSlope(true);  // [outDim], entries 0 or 1
                full_slope_debug = full_slope;
                // Extract only unstable neuron slopes using index_select
                slope_init = full_slope.flatten().index_select(0, params.unstableIndices);  // [numUnstable]
            } else {
                // Safe fallback: use 0.5 for unstable neurons (midpoint of valid range)
                slope_init = torch::full({numUnstable}, 0.5f, options);
            }
        }

        // Determine whether to allocate a default spec slot (sparse spec alpha)
        bool useSparseSpec = false;
        Vector<unsigned> cachedUnstableSpecs;
        bool cachedSparseMode = false;
        unsigned cachedNodeSize = 0;
        if (_crownAnalysis->getAlphaStartCacheInfo(startKey, cachedUnstableSpecs, cachedSparseMode, cachedNodeSize)) {
            useSparseSpec = cachedSparseMode && (int)cachedUnstableSpecs.size() == specDim;
        }

        // Create alpha tensor only for unstable neurons.
        // For sigmoid, keep 8 slots per neuron (auto_LiRPA s-shaped style):
        // [tp_pos_l, tp_pos_u, tp_neg_l, tp_neg_u, tp_both_lower_l,
        //  tp_both_lower_u, tp_both_upper_l, tp_both_upper_u]
        int alphaSlots = 1;
        if (nodePtr && nodePtr->getNodeType() == NodeType::SIGMOID) {
            alphaSlots = 8;
        }

        // Shape:
        // - default: [specDim(+1 if sparse), 1, numUnstable]
        // - sigmoid: [specDim(+1 if sparse), 1, numUnstable, alphaSlots]
        int specDimAlpha = useSparseSpec ? (specDim + 1) : specDim;
        torch::Tensor alpha;
        torch::Tensor expanded;
        if (alphaSlots == 1) {
            alpha = torch::zeros({specDimAlpha, 1, numUnstable}, options.dtype());
            expanded = slope_init.view({1, 1, numUnstable})
                        .expand({specDim, 1, numUnstable})
                        .contiguous()
                        .clone()
                        .to(options.dtype());
        } else {
            auto mid_unstable = ((input_lb_flat + input_ub_flat) * 0.5f)
                .index_select(0, params.unstableIndices)
                .to(options.dtype());
            torch::Tensor d_lower_unstable = mid_unstable.clone();
            torch::Tensor d_upper_unstable = mid_unstable.clone();
            if (nodePtr) {
                auto sigmoidNode = std::dynamic_pointer_cast<BoundedSigmoidNode>(nodePtr);
                if (sigmoidNode) {
                    auto d_pair = sigmoidNode->getDefaultTangentPoints(input_lb, input_ub);
                    d_lower_unstable = d_pair.first.flatten().index_select(0, params.unstableIndices).to(options.dtype());
                    d_upper_unstable = d_pair.second.flatten().index_select(0, params.unstableIndices).to(options.dtype());
                }
            }

            alpha = torch::zeros({specDimAlpha, 1, numUnstable, alphaSlots}, options.dtype());
            auto expanded_mid = mid_unstable.view({1, 1, numUnstable})
                .expand({specDim, 1, numUnstable})
                .contiguous()
                .clone()
                .to(options.dtype());
            auto expanded_d_lower = d_lower_unstable.view({1, 1, numUnstable})
                .expand({specDim, 1, numUnstable})
                .contiguous()
                .clone()
                .to(options.dtype());
            auto expanded_d_upper = d_upper_unstable.view({1, 1, numUnstable})
                .expand({specDim, 1, numUnstable})
                .contiguous()
                .clone()
                .to(options.dtype());

            // tp_pos/tp_neg init with midpoint.
            alpha.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(), 0}, expanded_mid);
            alpha.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(), 1}, expanded_mid);
            alpha.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(), 2}, expanded_mid);
            alpha.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(), 3}, expanded_mid);
            // tp_both_lower / tp_both_upper init from precomputed defaults (auto_LiRPA style).
            alpha.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(), 4}, expanded_d_lower);
            alpha.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(), 5}, expanded_d_lower);
            alpha.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(), 6}, expanded_d_upper);
            alpha.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(), 7}, expanded_d_upper);
            expanded = alpha.narrow(0, 0, specDim);
        }
        if (useSparseSpec) {
            alpha.narrow(0, 1, specDim).copy_(expanded);
            params.hasSpecDefaultSlot = true;
        } else {
            alpha.copy_(expanded);
        }

        // IMPORTANT: Use detach() to make alpha a proper leaf tensor.
        // ReLU-like alpha is clipped to [0,1]. Sigmoid alpha stores tangent points
        // and is clamped against dynamic interval-dependent limits in node logic.
        // Without detach(), the clamp operation creates a non-leaf tensor with a computation graph,
        // which can cause "modified by inplace operation" errors when the tensor is updated by optimizer
        if (alphaSlots == 1) {
            alpha = torch::clamp(alpha, 0.0f, 1.0f).detach();
        } else {
            alpha = alpha.detach();
        }
        alpha.set_requires_grad(true);

        params.alpha = alpha;
        perStart[startKey] = std::move(params);

    }
    return perStart[startKey];
}

// NEW REFACTORED ENTRY METHOD - Returns optimized bounds for specified side
torch::Tensor AlphaCROWNAnalysis::computeOptimizedBounds(LunaConfiguration::BoundSide side)
{
    log(Stringf("computeOptimizedBounds() - Starting optimization for %s bounds",
                side == LunaConfiguration::BoundSide::Lower ? "LOWER" : "UPPER"));
    if (!_initialized) {
        log("computeOptimizedBounds() - Initializing alpha parameters");
        initializeAlphaParameters();
    }

    if (!shouldPerformOptimization()) {
        log("computeOptimizedBounds() - Alpha optimization not applicable, falling back to standard CROWN");
        _crownAnalysis->run(false);
        return side == LunaConfiguration::BoundSide::Lower ? extractLowerBoundsFromCROWN() : extractUpperBoundsFromCROWN();
    }

    LunaConfiguration::BOUND_SIDE = side;
    bool isLower = (side == LunaConfiguration::BoundSide::Lower);
    resetAlphasFromCROWNSlopes(isLower);

    _bestIntermediateBounds.clear();
    _crownAnalysis->clearAllNodeBounds();
    setOptimizationStage("opt");
    LunaConfiguration::ENABLE_FIRST_LINEAR_IBP = false;

    _crownAnalysis->setAlphaStartCacheEnabled(true);
    _crownAnalysis->clearConcreteBounds();
    _crownAnalysis->clearCrownState();
    _crownAnalysis->resetProcessingState();
    _crownAnalysis->run(true);

    snapshotBestIntermediateBounds();

    torch::Tensor initialLower = extractLowerBoundsFromCROWN();
    torch::Tensor initialUpper = extractUpperBoundsFromCROWN();
    torch::Tensor initialBound = isLower ? initialLower : initialUpper;

    float initial_width = 0.0f;
    if (initialLower.defined() && initialUpper.defined() && initialLower.numel() > 0) {
        initial_width = (initialUpper - initialLower).mean().item<float>();
    }

    auto alphaParams = collectAlphaParameters(isLower);
    if (alphaParams.empty()) {
        _crownAnalysis->run(false);
        return side == LunaConfiguration::BoundSide::Lower ? extractLowerBoundsFromCROWN() : extractUpperBoundsFromCROWN();
    }

    auto optimizer = std::make_shared<torch::optim::Adam>(
        alphaParams,
        torch::optim::AdamOptions(_learningRate).betas(std::make_tuple(0.9, 0.999)).eps(1e-8));

    torch::Tensor bestBounds;
    float best_width = initial_width;
    float currentLR = _learningRate;
    const unsigned early_stop_patience = 5;
    unsigned iterations_without_improvement = 0;
    float best_loss = std::numeric_limits<float>::infinity();

    for (unsigned iter = 0; iter < _iteration; ++iter) {
        std::unique_ptr<torch::NoGradGuard> no_grad;
        if (iter == _iteration - 1) {
            no_grad = std::make_unique<torch::NoGradGuard>();
        }

        _crownAnalysis->clearConcreteBounds();
        _crownAnalysis->clearAllNodeBounds();
        _crownAnalysis->resetProcessingState();
        _crownAnalysis->run(true);

        torch::Tensor currentLower = extractLowerBoundsFromCROWN();
        torch::Tensor currentUpper = extractUpperBoundsFromCROWN();
        torch::Tensor currentBound = isLower ? currentLower : currentUpper;
        torch::Tensor currentBoundDetached = currentBound.detach();

        torch::Tensor loss;
        if (iter < _iteration - 1) {
            loss = computeLoss(currentLower, currentUpper, isLower);
        }

        bool improved = false;
        if (!bestBounds.defined()) {
            bestBounds = currentBoundDetached.clone();
            improved = true;
            int64_t specDim = currentBoundDetached.dim() == 0
                ? 1
                : (currentBoundDetached.dim() == 1 ? currentBoundDetached.size(0)
                                                   : currentBoundDetached.size(currentBoundDetached.dim() - 1));
            std::vector<int> allIndices;
            allIndices.reserve(static_cast<size_t>(specDim));
            for (int64_t i = 0; i < specDim; ++i) {
                allIndices.push_back(static_cast<int>(i));
            }
            updateBestAlphas(allIndices, isLower);
        } else {
            auto improvedIndices = findImprovedIndices(currentBoundDetached, bestBounds, isLower);
            improved = !improvedIndices.empty();
            if (improved) {
                bestBounds = isLower
                    ? torch::max(bestBounds, currentBoundDetached).detach()
                    : torch::min(bestBounds, currentBoundDetached).detach();
                updateBestAlphas(improvedIndices, isLower);
            }
        }

        if (improved) {
            if (currentLower.defined() && currentUpper.defined() && currentLower.numel() > 0) {
                best_width = (currentUpper - currentLower).mean().item<float>();
            }
            snapshotBestIntermediateBounds();
        }

        float current_loss = loss.defined() ? std::abs(loss.item<float>()) : 0.0f;
        if (improved || (loss.defined() && current_loss < best_loss)) {
            best_loss = current_loss;
            iterations_without_improvement = 0;
        } else if (loss.defined()) {
            iterations_without_improvement++;
        }
        if (iterations_without_improvement >= early_stop_patience) {
            break;
        }

        if (iter < _iteration - 1 && loss.defined()) {
            if (!loss.requires_grad()) {
                break;
            }
            optimizer->zero_grad();
            loss.backward();
            _crownAnalysis->clearConcreteBounds();
            _crownAnalysis->resetProcessingState();
            optimizer->step();
            clipAlphaParameters();

            currentLR *= LunaConfiguration::ALPHA_LR_DECAY;
            for (auto& group : optimizer->param_groups()) {
                if (auto adamGroup = dynamic_cast<torch::optim::AdamOptions*>(&group.options())) {
                    adamGroup->lr(currentLR);
                }
            }
        }
    }

    restoreBestAlphas(isLower);
    _crownAnalysis->clearConcreteBounds();
    restoreBestIntermediateBounds();
    _crownAnalysis->resetProcessingState();
    _crownAnalysis->run(false);

    torch::Tensor finalLower = extractLowerBoundsFromCROWN();
    torch::Tensor finalUpper = extractUpperBoundsFromCROWN();
    torch::Tensor finalBound = isLower ? finalLower : finalUpper;
    if (bestBounds.defined()) {
        finalBound = isLower ? torch::max(finalBound, bestBounds) : torch::min(finalBound, bestBounds);
    }

    float final_width = 0.0f;
    if (finalLower.defined() && finalUpper.defined() && finalLower.numel() > 0) {
        final_width = (finalUpper - finalLower).mean().item<float>();
    }
    float total_improvement_pct = 0.0f;
    if (initial_width > 0) {
        total_improvement_pct = ((initial_width - final_width) / initial_width) * 100.0f;
    }
    log(Stringf("computeOptimizedBounds() - Final bound width: %.6f (initial: %.6f, improvement: %.2f%%)",
                final_width, initial_width, total_improvement_pct));

    return finalBound;
}

// Extract lower bounds from CROWN analysis (following auto_LiRPA approach)
torch::Tensor AlphaCROWNAnalysis::extractLowerBoundsFromCROWN()
{
    // Get output layer concrete bounds
    unsigned outputIndex = _crownAnalysis->getOutputIndex();

    if (_crownAnalysis->hasConcreteBounds(outputIndex)) {
        torch::Tensor lowerBounds = _crownAnalysis->getConcreteLowerBound(outputIndex);

        return lowerBounds;
    }

    // Fallback to IBP bounds if concrete bounds not available
    if (_crownAnalysis->hasIBPBounds(outputIndex)) {
        torch::Tensor lowerBounds = _crownAnalysis->getIBPLowerBound(outputIndex);
        return lowerBounds;
    }

    // Return placeholder if no bounds available
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(_torchModel->getDevice());
    return torch::zeros({1}, options);
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
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(_torchModel->getDevice());
    return torch::zeros({1}, options);
}

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

void AlphaCROWNAnalysis::updateBestAlphas(const std::vector<int>& improvedIndices, bool isLower)
{
    if (improvedIndices.empty()) {
        return;
    }

    log(Stringf("updateBestAlphas(%s) - Updating best alphas for %zu improved specs",
                isLower ? "LOWER" : "UPPER", improvedIndices.size()));

    // Choose the appropriate storage based on bound side
    auto& currentStorage = isLower ? _alphaByNodeStartLower : _alphaByNodeStartUpper;
    auto& bestStorage = isLower ? _bestAlphaByNodeStartLower : _bestAlphaByNodeStartUpper;

    torch::NoGradGuard no_grad;
    using torch::indexing::Slice;

    for (auto& [nodeIdx, perStart] : currentStorage) {
        for (auto& [startKey, ap] : perStart) {
            if (!ap.alpha.defined() || ap.alpha.numel() == 0) {
                continue;
            }

            // Create AlphaParameters if it doesn't exist in bestStorage
            auto& bestPerStart = bestStorage[nodeIdx];
            auto itBest = bestPerStart.find(startKey);
            if (itBest == bestPerStart.end()) {
                AlphaParameters copy = ap;
                copy.alpha = ap.alpha.detach().clone();
                bestPerStart[startKey] = std::move(copy);
                continue;
            }

            auto& bestAp = itBest->second;
            auto currentAlpha = ap.alpha.detach();

            // If shape mismatch, replace entire alpha to avoid invalid indexing
            if (!bestAp.alpha.defined() || bestAp.alpha.sizes() != currentAlpha.sizes()) {
                bestAp.alpha = currentAlpha.clone();
                continue;
            }

            auto idxTensor = torch::tensor(
                improvedIndices,
                torch::TensorOptions().dtype(torch::kLong).device(currentAlpha.device()));

            // If alpha has a default spec slot, shift indices by 1 to skip the default
            if (ap.hasSpecDefaultSlot) {
                idxTensor = idxTensor + 1;
            }

            // Guard against out-of-range indices
            auto validMask = idxTensor < currentAlpha.size(0);
            idxTensor = idxTensor.index({validMask});
            if (idxTensor.numel() == 0) {
                continue;
            }

            // Update only the improved spec indices on dim 0 and keep all trailing dims.
            bestAp.alpha.index_put_({idxTensor}, currentAlpha.index({idxTensor}));
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
            auto& bestAp = bestStorage[nodeIdx][startKey];
            // Avoid size mismatch: only copy if shapes match
            if (ap.alpha.defined() && bestAp.alpha.defined()
                && ap.alpha.sizes() == bestAp.alpha.sizes()) {
                ap.alpha.data().copy_(bestAp.alpha.data());
            } else {
                // Align best storage to current shape for future runs
                if (ap.alpha.defined()) {
                    bestAp.alpha = ap.alpha.detach().clone();
                }
            }
        }
    }
}

void AlphaCROWNAnalysis::snapshotBestIntermediateBounds()
{
    // Element-wise best tracking: keep the tightest bounds ever seen for each layer
    // - Lower bounds: take max (tighter lower = higher value)
    // - Upper bounds: take min (tighter upper = lower value)
    // This may produce a combined state that never actually occurred together,
    // but tighter bounds are always sound for verification.

    unsigned numNodesUpdated = 0;
    unsigned numNodesNew = 0;

    // Iterate through all nodes and capture PRE-ACTIVATION bounds
    // (inputs to any alpha-optimizable activation node).
    unsigned numNodes = _crownAnalysis->getNumNodes();
    for (unsigned nodeIdx = 0; nodeIdx < numNodes; ++nodeIdx) {
        auto node = _crownAnalysis->getNode(nodeIdx);
        if (!node) continue;

        // Only track bounds for nodes that are INPUTS to optimizable activations.
        bool isPreActivation = false;

        // Check if this node feeds into an optimizable activation node.
        for (const auto& [actIdx, actNode] : _optimizableNodes) {
            (void)actNode;
            auto deps = _torchModel->getDependencies(actIdx);
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
            torch::Tensor currentLower = _crownAnalysis->getConcreteLowerBound(nodeIdx).detach();
            torch::Tensor currentUpper = _crownAnalysis->getConcreteUpperBound(nodeIdx).detach();

            auto it = _bestIntermediateBounds.find(nodeIdx);
            if (it != _bestIntermediateBounds.end()) {
                // Element-wise merge: keep tightest bounds
                // Detach to ensure no gradient history is preserved
                torch::Tensor& storedLower = it->second.first;
                torch::Tensor& storedUpper = it->second.second;

                storedLower = torch::max(storedLower, currentLower).detach().clone();
                storedUpper = torch::min(storedUpper, currentUpper).detach().clone();
                numNodesUpdated++;
            } else {
                // First time seeing this node - clone and detach to break gradient history
                _bestIntermediateBounds[nodeIdx] = std::make_pair(
                    currentLower.detach().clone(),
                    currentUpper.detach().clone());
                numNodesNew++;
            }
        }
    }

    if (numNodesNew > 0 || numNodesUpdated > 0) {
        log(Stringf("snapshotBestIntermediateBounds() - Element-wise best: %u nodes updated, %u nodes new",
                    numNodesUpdated, numNodesNew));
    }
}

void AlphaCROWNAnalysis::restoreBestIntermediateBounds()
{
    // Restore best intermediate bounds into CROWNAnalysis using node-based storage
    // This matches auto_LiRPA's fix_interm_bounds=True behavior where intermediate
    // bounds are fixed from the first computation and reused in subsequent iterations
    if (_bestIntermediateBounds.empty()) {
        return;
    }

    log("restoreBestIntermediateBounds() - Seeding best bounds into nodes");

    unsigned numRestored = 0;
    for (const auto& [nodeIdx, bounds] : _bestIntermediateBounds) {
        // Store bounds in node (auto_LiRPA style: node.lower/node.upper)
        // Detach to ensure no gradient history leaks into the new computation graph
        auto node = _crownAnalysis->getNode(nodeIdx);
        if (node) {
            node->setBounds(bounds.first.detach(), bounds.second.detach());
            numRestored++;
        }
    }

    log(Stringf("restoreBestIntermediateBounds() - Restored bounds for %u nodes", numRestored));
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

std::vector<int> AlphaCROWNAnalysis::findImprovedIndices(const torch::Tensor& currentBounds, const torch::Tensor& bestBounds, bool isLowerBound){

    std::vector<int> improvedIndices;

    if ( !currentBounds.defined() || !bestBounds.defined() || currentBounds.sizes() != bestBounds.sizes() )
    {
        return improvedIndices;
    }

    torch::Tensor improved = isLowerBound ? (currentBounds > bestBounds)
                                          : (currentBounds < bestBounds);

    if (!improved.defined() || improved.numel() == 0) {
        return improvedIndices;
    }

    // Reduce to per-spec mask: bounds are [spec] or [batch, spec]
    torch::Tensor perSpec;
    if (improved.dim() == 0) {
        if (improved.item<bool>()) {
            improvedIndices.push_back(0);
        }
        return improvedIndices;
    } else if (improved.dim() == 1) {
        perSpec = improved;
    } else {
        // Reduce all leading dims (e.g., batch) into a per-spec mask
        perSpec = improved;
        for (int d = 0; d < improved.dim() - 1; ++d) {
            perSpec = perSpec.any(0);
        }
    }

    auto improvedMask = perSpec.nonzero().flatten();
    if (improvedMask.numel() > 0) {
        for (int64_t i = 0; i < improvedMask.size(0); ++i) {
            improvedIndices.push_back(improvedMask[i].item<int>());
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
        // If not initialized yet, check whether the network has at least one
        // alpha-optimizable activation node.
        unsigned numNodes = _crownAnalysis->getNumNodes();
        for ( unsigned i = 0; i < numNodes; ++i ) 
        {
            auto node = _crownAnalysis->getNode(i);
            if ( node && std::dynamic_pointer_cast<BoundedAlphaOptimizeNode>(node) ) 
            {
                return true;
            }
        }
        return false;
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
// Returns AlphaResult with alpha [spec, numUnstable] and mapping info
AlphaCROWNAnalysis::AlphaResult AlphaCROWNAnalysis::getAlphaForNodeAllSpecs(
    unsigned nodeIndex,
    bool isLower,
    const std::string& startKey,
    int specDim,
    int outDim,
    const torch::Tensor& input_lb,
    const torch::Tensor& input_ub)
{
    (void)isLower; // Unused - we use LunaConfiguration::BOUND_SIDE instead in dual-sided optimization

    // Ensure per-(node,start) α exists with correct shape & initialization
    // This will use LunaConfiguration::BOUND_SIDE (set by the optimization loop) to choose the correct storage
    auto& ap = ensureAlphaFor(nodeIndex, startKey, specDim, outDim, input_lb, input_ub);

    AlphaResult result;
    result.numUnstable = ap.numUnstable;
    result.outDim = ap.outDim;
    result.unstableMask = ap.unstableMask;
    result.unstableIndices = ap.unstableIndices;
    result.hasSpecDefaultSlot = ap.hasSpecDefaultSlot;

    // ap.alpha: [spec, 1, numUnstable] -> drop batch dimension
    // Use squeeze(1) to maintain gradient connection
    result.alpha = ap.alpha.squeeze(1); // [spec or spec+1, numUnstable]

    return result;
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

    // IMPORTANT: Use NoGradGuard and .data().clamp_() to modify in-place without breaking autograd
    // Using set_data() or creating new tensors can cause version mismatch errors when views
    // of these tensors are used in the computation graph (e.g., via squeeze() in getAlphaForNodeAllSpecs)
    torch::NoGradGuard no_grad;

    auto shouldClip01 = [&](unsigned nodeIdx) -> bool {
        auto node = _crownAnalysis->getNode(nodeIdx);
        // Sigmoid uses tangent-point alpha parameters, not [0,1] slopes.
        return !(node && node->getNodeType() == NodeType::SIGMOID);
    };

    // Clip lower bound alpha parameters
    for (auto& [nodeIdx, perStart] : _alphaByNodeStartLower) {
        for (auto& [startKey, ap] : perStart) {
            (void)startKey;
            if (!shouldClip01(nodeIdx)) continue;
            // Use .data().clamp_() to directly modify underlying data without autograd tracking
            // This avoids version mismatch issues with views created during forward pass
            ap.alpha.data().clamp_(0.0f, 1.0f);
        }
    }

    // Clip upper bound alpha parameters
    for (auto& [nodeIdx, perStart] : _alphaByNodeStartUpper) {
        for (auto& [startKey, ap] : perStart) {
            (void)startKey;
            if (!shouldClip01(nodeIdx)) continue;
            // Use .data().clamp_() to directly modify underlying data without autograd tracking
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
    _crownAnalysis->clearAlphaStartCache();

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

// Configuration synchronization - read from LunaConfiguration
void AlphaCROWNAnalysis::updateFromConfig()
{
    _alphaEnabled = (LunaConfiguration::ANALYSIS_METHOD == LunaConfiguration::AnalysisMethod::AlphaCROWN);
    _iteration = LunaConfiguration::ALPHA_ITERATIONS;
    _learningRate = LunaConfiguration::ALPHA_LR;
    
    log(Stringf("updateFromConfig() - Updated from LunaConfiguration: enable=%s, iterations=%u, lr=%.3f", 
               _alphaEnabled ? "true" : "false", _iteration, _learningRate));
}

void AlphaCROWNAnalysis::log(const String& message)
{
    if (LunaConfiguration::NETWORK_LEVEL_REASONER_LOGGING || LunaConfiguration::VERBOSE) {
        printf("AlphaCROWNAnalysis: %s\n", message.ascii());
    }
}

} // namespace NLR
  