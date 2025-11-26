#ifndef __CROWNAnalysis_h__
#define __CROWNAnalysis_h__

#include "TorchModel.h"
#include "BoundedTorchNode.h"
#include "BoundedTensor.h"
#include "BoundResult.h"
#include "Map.h"
#include "Vector.h"
#include "Set.h"
#include "Queue.h"

#include <torch/torch.h>
#include <memory>

namespace NLR {

// Context structure for tracking current backward start
struct CrownStartContext {
    std::string start_key; // e.g. "/input-3", "/input-7", "/15"
    int spec_dim{1};
};

class CROWNAnalysis
{
public:
    CROWNAnalysis( TorchModel *torchModel, bool useStandardCROWN = true );
    ~CROWNAnalysis();

    // Analysis execution
    void run();
    
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

    // Fixed intermediate bounds (for alpha-CROWN best bounds tracking)
    void setFixedConcreteBounds(unsigned nodeIndex, const torch::Tensor& lower, const torch::Tensor& upper);
    void clearFixedConcreteBounds();

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
    void markProcessed(unsigned nodeIndex);
    bool isProcessed(unsigned nodeIndex) const;

    // First linear layer IBP fast path optimization
    bool checkIBPFirstLinear(unsigned nodeIndex);
    bool isFirstLinearLayer(unsigned nodeIndex);
    void setEnableFirstLinearIBP(bool enable) { _enableFirstLinearIBP = enable; }
    bool getEnableFirstLinearIBP() const { return _enableFirstLinearIBP; }


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
    void backwardFrom(unsigned startIndex);

    // Per-run cleanup of temporary CROWN state
    void clearCrownState();

    // Current start context for alpha-CROWN
    const CrownStartContext& currentStart() const { return _cur; }
    const std::string& currentStartKey() const { return _currentStartKey; }
    int currentStartSpecDim() const { return _currentStartSpecDim; }

    void _setCurrentStart(const std::string& key, int specDim) {
        _cur.start_key = key;
        _cur.spec_dim = specDim;
        _currentStartKey = key;
        _currentStartSpecDim = specDim;
    }

    // Concretize bounds for a specific node index
    void concretizeNode(unsigned startIndex);

private:
    TorchModel *_torchModel;
    bool _useStandardCROWN;
    bool _enableFirstLinearIBP;

    // Node-centric graph structure (delegated to TorchModel)
    // ie all graph management is done by torch model
    Map<unsigned, std::shared_ptr<BoundedTorchNode>> _nodes;
    
    // A matrix storage following auto-LiRPA's approach
    Map<unsigned, BoundA> _lA;  // lower bound A matrices
    Map<unsigned, BoundA> _uA;  // upper bound A matrices

    // Bias accumulation following auto-LiRPA's approach
    // The following were global bias accumulation, since we need bounds on every nueron, need the bias for each nodes individual scope
    // torch::Tensor _lowerBias;
    // torch::Tensor _upperBias;
    Map<unsigned, torch::Tensor> _lowerBias;
    Map<unsigned, torch::Tensor> _upperBias;

    Map<unsigned, BoundedTensor<torch::Tensor>> _ibpBounds;

    // Concrete Bounds
    Map<unsigned, BoundedTensor<torch::Tensor>> _concreteBounds;

    // Fixed intermediate bounds (for alpha-CROWN best bounds tracking)
    std::unordered_map<unsigned, std::pair<torch::Tensor, torch::Tensor>> _fixedConcreteBounds;

    // Forward value for concretizing bounds
    Map<unsigned, torch::Tensor> _forwardPassValues;

    // Concretize Bounds
    void computeConcreteBounds(const torch::Tensor& lA, const torch::Tensor& uA,
                              const torch::Tensor& lBias, const torch::Tensor& uBias,
                              const torch::Tensor& nodeLower, const torch::Tensor& nodeUpper,
                              torch::Tensor& concreteLower, torch::Tensor& concreteUpper);

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
            default: return "UNKNOWN";
        }
    }

    // Helper function for establishing consistent tensor format
    torch::Tensor preprocessC(const torch::Tensor& C, unsigned outputSize);

    // Current start context
    CrownStartContext _cur;
    std::string _currentStartKey;
    int _currentStartSpecDim{1};

};

} // namespace NLR

#endif // __CROWNAnalysis_h__
