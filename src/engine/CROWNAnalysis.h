#ifndef __CROWNAnalysis_h__
#define __CROWNAnalysis_h__

#include "TorchModel.h"
#include "nodes/BoundedTorchNode.h"
#include "BoundedTensor.h"
#include "BoundResult.h"
#include "Map.h"
#include "Vector.h"
#include "Set.h"
#include "Queue.h"
#include "input_parsers/OutputConstraint.h"
#include "configuration/LunaConfiguration.h"

#include <torch/torch.h>
#include <memory>
#include <unordered_map>

namespace NLR {

// Context structure for tracking current backward start
struct CrownStartContext {
    std::string start_key; // e.g. "/input-3", "/input-7", "/15"
    int spec_dim{1};
};

// Cached start-node spec info for alpha-CROWN
struct AlphaStartCache {
    Vector<unsigned> unstableIndices;
    bool sparseMode{false};
    unsigned nodeSize{0};
    bool initialized{false};
};

class CROWNAnalysis
{
public:
    CROWNAnalysis( TorchModel *torchModel );
    ~CROWNAnalysis();

    // Analysis execution
    // enableGradients: if true, enables gradient tracking (needed for Alpha-CROWN optimization)
    void run(bool enableGradients = false);
    
    // Node access
    std::shared_ptr<BoundedTorchNode> getNode(unsigned index) const;
    unsigned getInputSize() const;
    unsigned getOutputSize() const;
    unsigned getOutputIndex() const;

    // Public access methods for testing
    torch::Tensor getIBPLowerBound(unsigned nodeIndex);
    torch::Tensor getIBPUpperBound(unsigned nodeIndex);
    torch::Tensor getCrownLowerBound(unsigned nodeIndex) const;
    torch::Tensor getCrownUpperBound(unsigned nodeIndex) const;
    bool hasIBPBounds(unsigned nodeIndex);
    bool hasCrownBounds(unsigned nodeIndex);
    unsigned getNumNodes() const;

    // Concrete bound access methods
    torch::Tensor getConcreteLowerBound(unsigned nodeIndex);
    torch::Tensor getConcreteUpperBound(unsigned nodeIndex);
    bool hasConcreteBounds(unsigned nodeIndex);

    // Output bound access methods
    BoundedTensor<torch::Tensor> getOutputBounds() const;
    BoundedTensor<torch::Tensor> getOutputIBPBounds() const;

    // Model access for testing 
    TorchModel* getModel() const { return _torchModel; }

    // Additional public methods for testing
    Vector<BoundedTensor<torch::Tensor>> getInputBoundsForNode(unsigned nodeIndex);

    // Processing state
    void resetProcessingState();
    void clearConcreteBounds();
    void clearAllNodeBounds();  // Clear bounds stored in all nodes (node._lower/._upper)
    void markProcessed(unsigned nodeIndex);
    bool isProcessed(unsigned nodeIndex) const;

    // First linear layer IBP fast path optimization
    bool checkIBPFirstLinear(unsigned nodeIndex);
    bool isFirstLinearLayer(unsigned nodeIndex);
    // Configuration gating: use LunaConfiguration::ENABLE_FIRST_LINEAR_IBP directly
    
    // Check if IBP can be used for intermediate bounds (walks backward checking ibpIntermediate flags)
    bool checkIBPIntermediate(unsigned nodeIndex);

    // Compute intermediate bounds lazily - called when a node needs bounds on its inputs
    void computeIntermediateBoundsLazy(unsigned nodeIndex);

    // Check prior bounds - triggers lazy computation for nodes with requiresInputBounds
    void checkPriorBounds(unsigned nodeIndex);

    // Compute IBP bounds for a single node (used by lazy computation)
    void computeIBPForNode(unsigned nodeIndex);


    void computeIBPBounds();
    void computeCrownBackwardPropagation();
    void concretizeBounds();

    // Compute the forward pass vlaues via the torch model for concretizing the bounds
    void computeForwardPassValues();

    // Updated concrete bound method signatures
    torch::Tensor computeConcreteLowerBound(const torch::Tensor& lA, const torch::Tensor& lBias,
                                           const torch::Tensor& xLower, const torch::Tensor& xUpper);
    torch::Tensor computeConcreteUpperBound(const torch::Tensor& uA, const torch::Tensor& uBias,
                                           const torch::Tensor& xLower, const torch::Tensor& xUpper);


    void setInputBounds(const BoundedTensor<torch::Tensor>& inputBounds);
    BoundedTensor<torch::Tensor> getNodeIBPBounds(unsigned nodeIndex) const;
    BoundedTensor<torch::Tensor> getNodeCrownBounds(unsigned nodeIndex) const;
    BoundedTensor<torch::Tensor> getNodeConcreteBounds(unsigned nodeIndex) const;

    // Helper functions for A matrix accumulation (following auto-LiRPA's approach)
    BoundA addA(const BoundA& A1, const BoundA& A2);
    void addBound(unsigned nodeIndex, const BoundA& lA, const BoundA& uA);
    void addBias(unsigned nodeIndex, const torch::Tensor& lBias, const torch::Tensor& uBias);

    // Backward propagation starting from an arbitrary node (for standard CROWN intermediates)
    // If C is nullptr, queries TorchModel for specification matrix, otherwise uses identity
    void backwardFrom(unsigned startIndex, const Vector<unsigned>& unstableIndices = {}, const torch::Tensor* C = nullptr);

    // Per-run cleanup of temporary CROWN state
    void clearCrownState();

    // Current start context for alpha-CROWN
    const CrownStartContext& currentStart() const { return _cur; }
    const std::string& currentStartKey() const { return _currentStartKey; }
    int currentStartSpecDim() const { return _currentStartSpecDim; }
    const Vector<unsigned>& currentStartSpecIndices() const { return _currentStartSpecIndices; }

    void _setCurrentStart(const std::string& key, int specDim) {
        _cur.start_key = key;
        _cur.spec_dim = specDim;
        _currentStartKey = key;
        _currentStartSpecDim = specDim;
    }

    // Alpha-CROWN: cache start-node unstable sets from init/reference bounds
    void setAlphaStartCacheEnabled(bool enabled) { _alphaStartCacheEnabled = enabled; }
    void clearAlphaStartCache() { _alphaStartCache.clear(); }
    bool getAlphaStartCacheInfo(const std::string& key,
                                Vector<unsigned>& unstableIndices,
                                bool& sparseMode,
                                unsigned& nodeSize) const;

    // Concretize bounds for a specific node index
    void concretizeNode(unsigned startIndex, const Vector<unsigned>& unstableIndices = {});

    // Determine which nodes need CROWN bounds (selective computation)
    bool needsCROWNBounds(unsigned nodeIndex);

private:
    TorchModel *_torchModel;
    // Configuration is now accessed via LunaConfiguration static members
    // Removed _useStandardCROWN and _enableFirstLinearIBP member variables

    // Node-centric graph structure (delegated to TorchModel)
    // ie all graph management is done by torch model
    Map<unsigned, std::shared_ptr<BoundedTorchNode>> _nodes;
    
    // A matrix storage following auto-LiRPA's approach
    Map<unsigned, BoundA> _lA;  // lower bound A matrices
    Map<unsigned, BoundA> _uA;  // upper bound A matrices

    Map<unsigned, torch::Tensor> _lowerBias;
    Map<unsigned, torch::Tensor> _upperBias;

    Map<unsigned, BoundedTensor<torch::Tensor>> _ibpBounds;

    // Concrete Bounds
    Map<unsigned, BoundedTensor<torch::Tensor>> _concreteBounds;

    // Forward value for concretizing bounds
    Map<unsigned, torch::Tensor> _forwardPassValues;

    // Center-point activations (for soundness checks / debugging).
    // Computed lazily from (inputLower+inputUpper)/2 and cached for the lifetime of this analysis run.
    bool _hasCenterActivations{false};
    Map<unsigned, torch::Tensor> _centerActivations;

    // Track first unsound node (center not contained in concretized bounds).
    bool _foundFirstUnsound{false};
    unsigned _firstUnsoundNode{0};

    // Alpha-CROWN: cache unstable indices per start node (fixed at init)
    std::unordered_map<std::string, AlphaStartCache> _alphaStartCache;
    bool _alphaStartCacheEnabled{false};
    Vector<unsigned> _currentStartSpecIndices;

    // Concretize Bounds
    void computeConcreteBounds(const torch::Tensor& lA, const torch::Tensor& uA,
                              const torch::Tensor& lBias, const torch::Tensor& uBias,
                              const torch::Tensor& nodeLower, const torch::Tensor& nodeUpper,
                              torch::Tensor& concreteLower, torch::Tensor& concreteUpper);

    // Center-activation helpers
    void ensureCenterActivations();
    torch::Tensor buildCenterInputForForward() const;

    // Utility methods
    void log( const String &message );
    std::string nodeTypeToString(NodeType type) {
        switch (type) {
            case NodeType::INPUT: return "INPUT";
            case NodeType::CONSTANT: return "CONSTANT";
            case NodeType::LINEAR: return "LINEAR";
            case NodeType::RELU: return "RELU";
            case NodeType::RESHAPE: return "RESHAPE";
            case NodeType::FLATTEN: return "FLATTEN";
            case NodeType::IDENTITY: return "IDENTITY";
            case NodeType::ADD: return "ADD";
            case NodeType::SUB: return "SUB";
            case NodeType::CONV: return "CONV";
            case NodeType::BATCHNORM: return "BATCHNORM";
            case NodeType::SIGMOID: return "SIGMOID";
            default: return "UNKNOWN";
        }
    }

    // Helper function for establishing consistent tensor format
    // Similar to auto_LiRPA's _preprocess_C: transforms C from (batch, spec) to (spec, batch, *output_shape)
    torch::Tensor preprocessC(const torch::Tensor& C, unsigned startIndex);

    // Current start context
    CrownStartContext _cur;
    std::string _currentStartKey;
    int _currentStartSpecDim{1};

    // Track nodes that need bounds extracted during backward pass
    Set<unsigned> _nodesNeedingBounds;

    // Store intermediate A matrices and biases for computing bounds
    // These are snapshots during the backward pass
    Map<unsigned, Pair<BoundA, torch::Tensor>> _intermediateA;       // Lower A and bias
    Map<unsigned, Pair<BoundA, torch::Tensor>> _intermediateAUpper;  // Upper A and bias

};

} // namespace NLR

#endif // __CROWNAnalysis_h__
