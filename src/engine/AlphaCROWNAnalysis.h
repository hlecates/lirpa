#ifndef __AlphaCROWNAnalysis_h__
#define __AlphaCROWNAnalysis_h__

#include "CROWNAnalysis.h"
#include "BoundedTensor.h"
#include "Map.h"
#include "Vector.h"
#include "MString.h"
#include "configuration/LunaConfiguration.h"

#include <torch/torch.h>
#include <memory>
#include <vector>
#include <unordered_map>

namespace NLR {

// Forward declarations
class BoundedAlphaOptimizeNode;
class TorchModel;

// BoundSide enum and AlphaCROWNConfig struct moved to LunaConfiguration
// Use LunaConfiguration::BoundSide and LunaConfiguration static members

// Structure to hold alpha parameters for a single layer and start
struct AlphaParameters {
    torch::Tensor alpha;        // Shape: [spec_dim, 1, num_unstable] - only unstable neurons
    torch::Tensor unstableMask; // Shape: [outDim] - bool mask indicating unstable neurons
    torch::Tensor unstableIndices; // Shape: [num_unstable] - indices of unstable neurons
    int specDim{0};             // Number of specifications being verified
    int batchDim{1};            // Batch dimension (typically 1)
    int outDim{0};              // Total number of neurons in the layer
    int numUnstable{0};         // Number of unstable neurons (alpha.size(-1))
    bool requiresGrad{true};    // Whether gradients are enabled
    bool hasSpecDefaultSlot{false}; // Sparse-spec alpha includes default slot
};

class AlphaCROWNAnalysis
{
public:
    AlphaCROWNAnalysis(TorchModel* torchModel);
    
    ~AlphaCROWNAnalysis();
    
    void initializeAlphaParameters();

    torch::Tensor getAlphaForNode(unsigned nodeIndex, bool isLowerBound, unsigned specIndex = 0, unsigned batchIndex = 0) const;

    // Result structure for getAlphaForNodeAllSpecs
    struct AlphaResult {
        torch::Tensor alpha;          // [spec, numUnstable] - alpha values for unstable neurons only
        torch::Tensor unstableMask;   // [outDim] - bool mask of unstable neurons
        torch::Tensor unstableIndices;// [numUnstable] - indices of unstable neurons
        int numUnstable{0};           // Number of unstable neurons
        int outDim{0};                // Total number of neurons
        bool hasSpecDefaultSlot{false}; // Alpha includes default spec slot
    };

    // Fetch alpha slice for ALL specs at once for a specific start.
    // Returns AlphaResult with alpha [spec, numUnstable] and mapping info
    AlphaResult getAlphaForNodeAllSpecs(
        unsigned nodeIndex,
        bool isLower,
        const std::string& startKey,
        int specDim,
        int outDim,
        const torch::Tensor& input_lb,
        const torch::Tensor& input_ub);
    
    void setOptimizationStage(const std::string& stage);

    std::string getOptimizationStage() const { return _optimizationStage; }
    
    bool hasAlphaParameters(unsigned nodeIndex) const;
    
    unsigned getNumOptimizableNodes() const;
    
    std::vector<unsigned> getOptimizableNodeIndices() const;
    
    void resetForOptimization();

    // Reset alpha parameters from CROWN slopes for fresh optimization pass
    void resetAlphasFromCROWNSlopes(bool isLower);

    CROWNAnalysis* getCROWNAnalysis() const;
    TorchModel* getTorchModel() const;

    torch::Tensor getAllAlphaParameters() const;

    // Bounds access methods
    std::pair<torch::Tensor, torch::Tensor> getAllCROWNBounds() const;
    std::pair<torch::Tensor, torch::Tensor> getCROWNBoundsForLayer(unsigned nodeIndex) const;
    std::pair<torch::Tensor, torch::Tensor> getAllConcreteBounds() const;
    std::pair<torch::Tensor, torch::Tensor> getConcreteBoundsForLayer(unsigned nodeIndex) const;
    
    // Alpha parameter access methods
    std::unordered_map<unsigned, AlphaParameters> getAllAlphaParametersByLayer() const;
    AlphaParameters getAlphaParametersForLayer(unsigned nodeIndex) const;
    std::vector<torch::Tensor> getAllAlphaTensors() const;

    // NEW REFACTORED ENTRY METHOD - Returns optimized bounds for specified side
    // This method handles the complete optimization loop internally
    torch::Tensor computeOptimizedBounds(LunaConfiguration::BoundSide side);

    // Preserve intermediate bounds for reuse in next optimization call
    // Call this after first optimization (e.g., upper) to reuse state for second (e.g., lower)
    void preserveIntermediateBoundsForNextOptimization();

    // Check if intermediate bounds are available for reuse
    bool hasIntermediateBoundsForReuse() const { return _reuseIntermediateBounds; }

    /* DEPRECATED - OLD IMPLEMENTATIONS
    std::pair<torch::Tensor, torch::Tensor> computeBoundsWithAlpha(BoundSide side);

    std::pair<torch::Tensor, torch::Tensor> run();

    std::pair<torch::Tensor, torch::Tensor> run(
        const torch::Tensor* input,
        const torch::Tensor* specificationMatrix,
        bool computeLowerBounds = true,
        bool computeUpperBounds = true
    );
    */


    void clipAlphaParameters();

    // Alpha parameter collection (needed by TorchModel for optimizer creation)
    std::vector<torch::Tensor> collectAlphaParameters(bool isLower);

    // Configuration parameters
    bool isAlphaEnabled() const { return _alphaEnabled; }
    void setAlphaEnabled(bool enabled) { _alphaEnabled = enabled; }
    
    bool isInitialized() const { return _initialized; }
    
    unsigned getIterations() const { return _iteration; }
    void setIterations(unsigned iterations) { _iteration = iterations; }
    
    float getLearningRate() const { return _learningRate; }
    void setLearningRate(float lr) { _learningRate = lr; }
    void decayLearningRate() { _learningRate *= LunaConfiguration::ALPHA_LR_DECAY; }
    
    // Configuration is now accessed via LunaConfiguration static members
    // Removed getConfig()/setConfig() methods - use LunaConfiguration directly
    
    // Set individual config options (update LunaConfiguration directly)
    void setIteration(unsigned iteration) { 
        LunaConfiguration::ALPHA_ITERATIONS = iteration; 
        _iteration = iteration; 
    }
    void setLrAlpha(float lr) { 
        LunaConfiguration::ALPHA_LR = lr; 
        _learningRate = lr; 
    }
    void setKeepBest(bool keep) { LunaConfiguration::KEEP_BEST = keep; }
    void setOptimizer(const std::string& opt) { LunaConfiguration::OPTIMIZER = String(opt.c_str()); }
    void setBoundSide(LunaConfiguration::BoundSide side) { LunaConfiguration::BOUND_SIDE = side; }

    // Optimization side queries (read from LunaConfiguration)
    LunaConfiguration::BoundSide getBoundSide() const { return LunaConfiguration::BOUND_SIDE; }
    bool isOptimizingLower() const { return LunaConfiguration::BOUND_SIDE == LunaConfiguration::BoundSide::Lower; }
    bool isOptimizingUpper() const { return LunaConfiguration::BOUND_SIDE == LunaConfiguration::BoundSide::Upper; }

private:

    TorchModel* _torchModel;
    std::unique_ptr<CROWNAnalysis> _crownAnalysis;

    // Alpha parameter storage: separate for lower and upper bounds
    // nodeIndex -> startKey -> AlphaParameters
    std::unordered_map<unsigned,
        std::unordered_map<std::string, AlphaParameters>> _alphaByNodeStartLower;
    std::unordered_map<unsigned,
        std::unordered_map<std::string, AlphaParameters>> _alphaByNodeStartUpper;

    // old storage (kept during migration for compatibility)
    std::unordered_map<unsigned, AlphaParameters> _alphaParameters;

    // Optimizable activation nodes
    std::vector<std::pair<unsigned, std::shared_ptr<BoundedAlphaOptimizeNode>>> _optimizableNodes;
    
    // Configuration is now accessed via LunaConfiguration static members
    // Removed _config member variable
    bool _alphaEnabled;         // Whether alpha optimization is enabled
    bool _initialized;          // Whether alpha parameters have been initialized
    bool _reuseIntermediateBounds{false}; // Reuse intermediate bounds from previous optimization
    unsigned _iteration;        // Number of optimization iterations (default: 20)
    float _learningRate;        // Learning rate for alpha optimization (default: 0.5)
    std::string _optimizationStage;  // Current optimization stage

    // Best alpha tracking for optimization: separate for lower and upper bounds
    std::unordered_map<unsigned,
        std::unordered_map<std::string, AlphaParameters>> _bestAlphaByNodeStartLower;
    std::unordered_map<unsigned,
        std::unordered_map<std::string, AlphaParameters>> _bestAlphaByNodeStartUpper;

    // Legacy best alpha storage
    std::unordered_map<unsigned, AlphaParameters> _bestAlphaParameters;

    /* DEPRECATED - Optimizers now managed by TorchModel
    std::shared_ptr<torch::optim::Optimizer> _optimizerLower;
    std::shared_ptr<torch::optim::Optimizer> _optimizerUpper;
    */

    // Best bounds tracking for dual optimization
    torch::Tensor _bestLowerBounds;
    torch::Tensor _bestUpperBounds;

    // Best intermediate bounds tracking (per-node concrete bounds)
    // Maps nodeIndex -> (best_lower, best_upper) for each layer
    std::unordered_map<unsigned, std::pair<torch::Tensor, torch::Tensor>> _bestIntermediateBounds;


    
    // Initialization phase methods
    
    void performForwardPass();
    void prepareOptimizableActivations();
    void performCROWNInitializationPass();
    void createAlphaParameters();
    
    void createAlphaForNode(unsigned nodeIndex,
                           std::shared_ptr<BoundedAlphaOptimizeNode> optimizableNode,
                           unsigned outputSize);

    void initializeAlphaWithCROWNSlopes(torch::Tensor& alpha,
                                       std::shared_ptr<BoundedAlphaOptimizeNode> optimizableNode,
                                       unsigned nodeIndex);

    // Ensure alpha exists for (node, start) pair with correct shape and initialization
    AlphaParameters& ensureAlphaFor(
        unsigned nodeIndex,
        const std::string& startKey,
        int specDim, int outDim,
        const torch::Tensor& input_lb,
        const torch::Tensor& input_ub);
    
    // Update best alphas for improvements (for specific bound side)
    void updateBestAlphas(const std::vector<int>& improvedIndices, bool isLower);

    // Reset alpha to best known values (for specific bound side)
    void restoreBestAlphas(bool isLower);

    /* DEPRECATED - Best alpha tracking simplified
    void setupBestAlphaTracking();
    */

    // Best intermediate bounds tracking (following auto-LiRPA)
    void snapshotBestIntermediateBounds();
    void restoreBestIntermediateBounds();

    torch::Tensor computeLoss(const torch::Tensor& lowerBounds,
                             const torch::Tensor& upperBounds,
                             bool optimizeLower,
                             c10::optional<torch::Tensor> stop_mask_opt = c10::nullopt);

    /* DEPRECATED - Gradient step now in TorchModel
    void performGradientStep(std::shared_ptr<torch::optim::Optimizer>& optimizer,
                           const torch::Tensor& loss);
    */
  
    std::vector<int> findImprovedIndices(const torch::Tensor& currentBounds,
                                        const torch::Tensor& bestBounds,
                                        bool isLowerBound);

    // Helper method
    bool shouldPerformOptimization() const;
    
    // Bound extraction methods (following auto_LiRPA approach)
    torch::Tensor extractLowerBoundsFromCROWN();
    torch::Tensor extractUpperBoundsFromCROWN();
    
    // Configuration synchronization
    void updateFromConfig();

    // Utility 
    void log(const String& message);
};

} // namespace NLR

#endif // __AlphaCROWNAnalysis_h__