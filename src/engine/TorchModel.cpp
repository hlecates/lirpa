#include "TorchModel.h"
#include "CROWNAnalysis.h"
#include "AlphaCROWNAnalysis.h"
#include "LirpaError.h"
#include "LirpaConfiguration.h"
#include "OnnxToTorch.h"
#include "VnnLibInputParser.h"
#include "OutputConstraint.h"
#include <iostream>

namespace NLR {


// CONSTRUCTOR AND INITIALIZATION


TorchModel::TorchModel(const Vector<std::shared_ptr<BoundedTorchNode>>& nodes,
                       const Vector<unsigned>& inputIndices,
                       unsigned outputIndex,
                       const Map<unsigned, Vector<unsigned>>& dependencies)
    : _nodes(nodes), _inputIndices(inputIndices),
      _outputIndex(outputIndex), _dependencies(dependencies), _input_size(0), _output_size(0),
      _hasSpecificationMatrix(false), _hasORBranches(false), _hasFinalAnalysisBounds(false),
      _device(LirpaConfiguration::getDevice()) {
    
    log(Stringf("[TorchModel] Constructor called with %u nodes", nodes.size()));
    
    // Set input size from the first input node
    if (!inputIndices.empty() && inputIndices[0] < nodes.size()) {
        auto inputNode = nodes[inputIndices[0]];
        if (inputNode) {
            _input_size = inputNode->getOutputSize();
            log(Stringf("[TorchModel] Set input size to %u from input node %u", _input_size, inputIndices[0]));
        }
    }
    
    // Set output size from the output node
    if (outputIndex < nodes.size()) {
        auto outputNode = nodes[outputIndex];
        if (outputNode) {
            _output_size = outputNode->getOutputSize();
            log(Stringf("[TorchModel] Set output size to %u from output node %u", _output_size, outputIndex));
        }
    }
    
    // Ensure all nodes are on the configured device
    moveToDevice(_device);

    // Build dependency graph
    buildDependencyGraph();
}

// Constructor that loads from ONNX file (mirrors auto_LiRPA BoundedModule)
TorchModel::TorchModel(const String& onnxPath)
    : _device(LirpaConfiguration::getDevice()) {
    log(Stringf("[TorchModel] Constructor called with ONNX path: %s", onnxPath.ascii()));

    // Use OnnxToTorchParser to parse the ONNX file
    std::shared_ptr<TorchModel> parsedModel = OnnxToTorchParser::parse(onnxPath);

    if (!parsedModel) {
        throw LirpaError(LirpaError::ONNX_PARSING_ERROR,
                          Stringf("Failed to parse ONNX file: %s", onnxPath.ascii()).ascii());
    }

    // Copy all members from parsed model to this instance
    _nodes = parsedModel->_nodes;
    _inputIndices = parsedModel->_inputIndices;
    _outputIndex = parsedModel->_outputIndex;
    _dependencies = parsedModel->_dependencies;
    _input_size = parsedModel->_input_size;
    _output_size = parsedModel->_output_size;
    _hasSpecificationMatrix = false;
    _hasORBranches = false;
    _hasFinalAnalysisBounds = false;
    _device = LirpaConfiguration::getDevice();

    // Initialize analysis configuration with defaults
    // Configuration is now accessed via LirpaConfiguration static members

    // Ensure all nodes are on the configured device
    moveToDevice(_device);

    // Build dependency graph
    buildDependencyGraph();

    log(Stringf("[TorchModel] Successfully loaded ONNX model with %u nodes", _nodes.size()));
}

// Constructor that loads from ONNX file and VNN-LIB file for input bounds
TorchModel::TorchModel(const String& onnxPath,
                       const String& vnnlibPath)
    : _device(LirpaConfiguration::getDevice()) {
    log(Stringf("[TorchModel] Constructor called with ONNX path: %s and VNN-LIB path: %s",
                onnxPath.ascii(), vnnlibPath.ascii()));

    // First, parse the ONNX file to build the network structure
    std::shared_ptr<TorchModel> parsedModel = OnnxToTorchParser::parse(onnxPath);

    if (!parsedModel) {
        throw LirpaError(LirpaError::ONNX_PARSING_ERROR,
                          Stringf("Failed to parse ONNX file: %s", onnxPath.ascii()).ascii());
    }

    // Copy all members from parsed model to this instance
    _nodes = parsedModel->_nodes;
    _inputIndices = parsedModel->_inputIndices;
    _outputIndex = parsedModel->_outputIndex;
    _dependencies = parsedModel->_dependencies;
    _input_size = parsedModel->_input_size;
    _output_size = parsedModel->_output_size;
    _hasSpecificationMatrix = false;
    _hasORBranches = false;
    _hasFinalAnalysisBounds = false;
    _device = LirpaConfiguration::getDevice();

    // Initialize analysis configuration with defaults
    // Configuration is now accessed via LirpaConfiguration static members

    // Ensure all nodes are on the configured device
    moveToDevice(_device);

    // Build dependency graph
    buildDependencyGraph();

    log(Stringf("[TorchModel] Successfully loaded ONNX model with %u nodes", _nodes.size()));

    // Now parse the VNN-LIB file to extract input bounds and output constraints
    log(Stringf("[TorchModel] Parsing VNN-LIB file for input bounds: %s", vnnlibPath.ascii()));

    try {
        // Parse input bounds
        BoundedTensor<torch::Tensor> inputBounds =
            VnnLibInputParser::parseInputBounds(vnnlibPath, _input_size);

        // Set the input bounds on this model
        setInputBounds(inputBounds);

        log(Stringf("[TorchModel] Successfully set input bounds from VNN-LIB file"));

        // Parse output constraints (if present)
        try {
            log(Stringf("[TorchModel] Parsing VNN-LIB file for output constraints"));
            OutputConstraintSet outputConstraints = 
                VnnLibInputParser::parseOutputConstraints(vnnlibPath, _output_size);

            if (outputConstraints.hasConstraints()) {
                log(Stringf("[TorchModel] Found %u output constraints", 
                            outputConstraints.getNumConstraints()));

                // Convert output constraints to C matrix
                CMatrixResult cMatrixResult = outputConstraints.toCMatrix();
                torch::Tensor C = cMatrixResult.C;

                // Store the specification matrix in the model
                setSpecificationMatrix(C);

                log(Stringf("[TorchModel] Successfully set specification matrix with %d constraints", 
                            (int)C.size(0)));
            } else {
                log(Stringf("[TorchModel] No output constraints found in VNN-LIB file"));
            }
        } catch (const std::exception& e) {
            // Output constraints are optional, so log but don't fail
            log(Stringf("[TorchModel] Could not parse output constraints: %s (continuing without them)", 
                        e.what()));
        }
    } catch (const LirpaError& e) {
        // Re-throw LirpaError as-is
        throw;
    } catch (const std::exception& e) {
        throw LirpaError(LirpaError::ONNX_PARSING_ERROR,
                          Stringf("Failed to parse VNN-LIB file %s: %s",
                                  vnnlibPath.ascii(), e.what()).ascii());
    }
}


// LOGGING


void TorchModel::log(const String& message) const {
    (void)message;
    if (LirpaConfiguration::NETWORK_LEVEL_REASONER_LOGGING) {
        //printf("TorchModel: %s\n", message.ascii());
    }
}


// GRAPH MANAGEMENT AND TRAVERSAL


void TorchModel::buildDependencyGraph() {
   // Clear existing graph structures
   _dependents.clear();
   _degreeOut.clear();
   _degreeIn.clear();
   _processed.clear();
   
   // Initialize structures for all nodes
   for (const auto& pair : _dependencies) {
       unsigned nodeIndex = pair.first;
       _dependents[nodeIndex] = Vector<unsigned>();
       _degreeOut[nodeIndex] = 0;
       _degreeIn[nodeIndex] = 0;
       _processed[nodeIndex] = false;
   }
   
   // Also initialize nodes that might not be in dependencies
   for (unsigned i = 0; i < _nodes.size(); ++i) {
       if (!_dependents.exists(i)) {
           _dependents[i] = Vector<unsigned>();
           _degreeOut[i] = 0;
           _degreeIn[i] = 0;
           _processed[i] = false;
       }
   }
   
   // Build dependents and compute degrees
   buildDependents();
   computeDegrees();
}

void TorchModel::buildDependents() {
   // For each node, build its dependents (reverse dependencies)
   for (const auto& pair : _dependencies) {
       unsigned nodeIndex = pair.first;
       
       // For each dependency of this node
       for (unsigned inputIndex : _dependencies[nodeIndex]) {
           // This node is a dependent of its input
           _dependents[inputIndex].append(nodeIndex);
       }
   }
}

void TorchModel::computeDegrees() {
   // Compute outgoing degrees (number of dependents)
   for (const auto& pair : _dependents) {
       unsigned nodeIndex = pair.first;
       _degreeOut[nodeIndex] = _dependents[nodeIndex].size();
   }
   
   // Compute incoming degrees (number of dependencies)
   for (const auto& pair : _dependencies) {
       unsigned nodeIndex = pair.first;
       _degreeIn[nodeIndex] = _dependencies[nodeIndex].size();
   }
}

void TorchModel::resetProcessingState() {
   for (auto& pair : _processed) {
       pair.second = false;
   }
}

Vector<unsigned> TorchModel::topologicalSort() const {
   Vector<unsigned> sortedOrder;
   Queue<unsigned> queue;
   
   // Copy degree map for tracking
   Map<unsigned, unsigned> degreeIn = _degreeIn;
   
   // Initialize queue with nodes that have no incoming edges
   // Check all nodes, not just those in dependencies
   for (unsigned i = 0; i < _nodes.size(); ++i) {
       if (degreeIn.exists(i) && degreeIn[i] == 0) {
           queue.push(i);
       }
   }
   
   // Process nodes in topological order
   while (!queue.empty()) {
       unsigned current = queue.peak();
       queue.pop();
       sortedOrder.append(current);
       
       // Update degrees for dependents
       if (_dependents.exists(current)) {
           for (unsigned dependent : _dependents[current]) {
               if (degreeIn.exists(dependent)) {
                   degreeIn[dependent]--;
                   if (degreeIn[dependent] == 0) {
                       queue.push(dependent);
                   }
               }
           }
       }
   }
   
   return sortedOrder;
}

Vector<unsigned> TorchModel::getRoots() const {
   Vector<unsigned> roots;
   for (const auto& pair : _degreeIn) {
       if (pair.second == 0) {
           roots.append(pair.first);
       }
   }
   return roots;
}

Vector<unsigned> TorchModel::getLeaves() const {
   Vector<unsigned> leaves;
   for (const auto& pair : _degreeOut) {
       if (pair.second == 0) {
           leaves.append(pair.first);
       }
   }
   return leaves;
}

Vector<unsigned> TorchModel::getDependents(unsigned nodeIndex) const {
   if (_dependents.exists(nodeIndex)) {
       return _dependents[nodeIndex];
   }
   return Vector<unsigned>();
}

Vector<unsigned> TorchModel::getDependencies(unsigned nodeIndex) const {
   if (_dependencies.exists(nodeIndex)) {
       return _dependencies[nodeIndex];
   }
   return Vector<unsigned>();
}

unsigned TorchModel::getDegreeOut(unsigned nodeIndex) const {
   if (_degreeOut.exists(nodeIndex)) {
       return _degreeOut[nodeIndex];
   }
   return 0;
}

unsigned TorchModel::getDegreeIn(unsigned nodeIndex) const {
   if (_degreeIn.exists(nodeIndex)) {
       return _degreeIn[nodeIndex];
   }
   return 0;
}

bool TorchModel::isProcessed(unsigned nodeIndex) const {
   if (_processed.exists(nodeIndex)) {
       return _processed[nodeIndex];
   }
   return false;
}

void TorchModel::markProcessed(unsigned nodeIndex) {
   if (_processed.exists(nodeIndex)) {
       _processed[nodeIndex] = true;
   }
}


// NODE ACCESS AND INFORMATION


std::shared_ptr<BoundedTorchNode> TorchModel::getNode(unsigned index) const {
    if ( index < _nodes.size() ) 
    {
        return _nodes[index];
    }
    return nullptr;
}

Vector<unsigned> TorchModel::getAllNodeIndices() const {
    Vector<unsigned> indices;
    for ( unsigned i = 0; i < _nodes.size(); ++i ) 
    {
        indices.append(i);
    }
    return indices;
}

Vector<unsigned> TorchModel::getNodesByType(NodeType type) const {
    Vector<unsigned> indices;
    for ( unsigned i = 0; i < _nodes.size(); ++i ) 
    {
        if ( _nodes[i]->getNodeType() == type ) 
        {
            indices.append(i);
        }
    }
    return indices;
}

void TorchModel::moveToDevice(const torch::Device& device)
{
    _device = device;
    for (unsigned i = 0; i < _nodes.size(); ++i) {
        if (_nodes[i]) {
            _nodes[i]->moveToDevice(device);
        }
    }
    if (_inputBounds.lower().defined() || _inputBounds.upper().defined()) {
        _inputBounds = BoundedTensor<torch::Tensor>(
            _inputBounds.lower().to(device),
            _inputBounds.upper().to(device));
    }
    if (_specificationMatrix.defined()) {
        _specificationMatrix = _specificationMatrix.to(device);
    }
    if (_specificationThresholds.defined()) {
        _specificationThresholds = _specificationThresholds.to(device);
    }
    if (_finalAnalysisBounds.lower().defined() || _finalAnalysisBounds.upper().defined()) {
        _finalAnalysisBounds = BoundedTensor<torch::Tensor>(
            _finalAnalysisBounds.lower().to(device),
            _finalAnalysisBounds.upper().to(device));
    }
}


// FORWARD PASS AND COMPUTATION


torch::Tensor TorchModel::forward(unsigned nodeIndex, Map<unsigned, torch::Tensor>& activations, 
    const Map<unsigned, torch::Tensor>& inputs) {
    // Return cached result if available
    if ( activations.exists(nodeIndex) ) 
    {
        return activations[nodeIndex];
    }

    if ( nodeIndex >= _nodes.size() ) 
    {
        throw LirpaError(LirpaError::INVALID_MODEL_STRUCTURE, (String("Node index not found: ") + std::to_string(nodeIndex)).ascii());
    }

    auto& node = _nodes[nodeIndex];
    NodeType nodeType = node->getNodeType();

    // Handle based on node type
    switch (nodeType) 
    {
        case NodeType::INPUT: 
        {
        // Input nodes get their value from the inputs map
        unsigned inputIndex = nodeIndex; // Assuming input indices match node indices
        if ( !inputs.exists(inputIndex) ) 
        {
            throw LirpaError(LirpaError::INVALID_MODEL_STRUCTURE, (String("Input index not found: ") + std::to_string(inputIndex)).ascii());
        }
        activations[nodeIndex] = inputs[inputIndex];
        return inputs[inputIndex];
        }
        
        case NodeType::CONSTANT: 
        {
            // Constants return their fixed value
            torch::Tensor result = node->forward(torch::Tensor());
            activations[nodeIndex] = result;
            return result;
        }

        case NodeType::LINEAR:
        case NodeType::RELU:
        case NodeType::RESHAPE:
        case NodeType::FLATTEN:
        case NodeType::IDENTITY:
        case NodeType::SUB:
        case NodeType::ADD:
        case NodeType::CONV:
        case NodeType::BATCHNORM:
        case NodeType::SIGMOID:
        case NodeType::CONCAT:
        case NodeType::CONVTRANSPOSE:
        case NodeType::SLICE: {
            // Module nodes need to compute their inputs first
            if ( !_dependencies.exists(nodeIndex) ) 
            {
                throw LirpaError(LirpaError::INVALID_MODEL_STRUCTURE, (String("No dependencies found for node at index: ") + std::to_string(nodeIndex)).ascii());
            }
            
            Vector<unsigned> deps = _dependencies[nodeIndex];
            
            // Recursively compute all input activations
            std::vector<torch::Tensor> inputTensors;
            for ( unsigned dep : deps ) 
            {
                inputTensors.push_back(forward(dep, activations, inputs));
            }

            if ( inputTensors.empty() ) 
            {
                throw LirpaError(LirpaError::INVALID_MODEL_STRUCTURE, (String("No input tensors for node at index: ") + std::to_string(nodeIndex)).ascii());
            }

            torch::Tensor result;
            if ( inputTensors.size() == 1 ) 
            {
                result = node->forward(inputTensors[0]);
            } 
            else 
            {
                result = node->forward(inputTensors);
            }

            activations[nodeIndex] = result;
            return result;
        }
    }
    
    // This should never be reached
    throw LirpaError(LirpaError::INVALID_MODEL_STRUCTURE, (String("Unknown node type for node index: ") + std::to_string(nodeIndex)).ascii());
}

/*
torch::Tensor TorchModel::forward(const torch::Tensor& input) {
    // Forward pass through the entire model
    Map<unsigned, torch::Tensor> activations;
    Map<unsigned, torch::Tensor> inputs_map;
    
    // Set input for input nodes
    for (unsigned inputIndex : _inputIndices) {
        inputs_map[inputIndex] = input;
    }
    
    // Forward through all nodes
    for (unsigned i = 0; i < _nodes.size(); ++i) {
        forward(i, activations, inputs_map);
    }
    
    // Return output from the output index
    return activations[_outputIndex];
}

torch::Tensor TorchModel::forward(const Map<unsigned, torch::Tensor>& inputs) {
    // Forward pass with pre-defined inputs
    Map<unsigned, torch::Tensor> activations;
    
    // Forward through all nodes
    for (unsigned i = 0; i < _nodes.size(); ++i) {
        forward(i, activations, inputs);
    }
    
    // Return output from the output index
    return activations[_outputIndex];
}
*/


// NEW: Forward pass that stores and returns activations for all nodes
Map<unsigned, torch::Tensor> TorchModel::forwardAndStoreActivations(const torch::Tensor& input) {
    Map<unsigned, torch::Tensor> activations;
    Map<unsigned, torch::Tensor> inputs_map;
    torch::Tensor deviceInput = input.to(_device);
    
    // Set input for input nodes
    for (unsigned inputIndex : _inputIndices) {
        inputs_map[inputIndex] = deviceInput;
    }
    
    // Forward through all nodes
    for (unsigned i = 0; i < _nodes.size(); ++i) {
        forward(i, activations, inputs_map);
    }
    
    return activations;
}

Map<unsigned, torch::Tensor> TorchModel::forwardAndStoreActivations(const Map<unsigned, torch::Tensor>& inputs) {
    Map<unsigned, torch::Tensor> activations;
    Map<unsigned, torch::Tensor> deviceInputs;

    for (auto it = inputs.begin(); it != inputs.end(); ++it) {
        deviceInputs[it->first] = it->second.to(_device);
    }
    
    // Forward through all nodes
    for (unsigned i = 0; i < _nodes.size(); ++i) {
        forward(i, activations, deviceInputs);
    }
    
    return activations;
}

// BOUND MANAGEMENT INTERFACE

void TorchModel::setInputBounds(const BoundedTensor<torch::Tensor>& inputBounds) {
    log(Stringf("[TorchModel] Setting input bounds"));
    
    // Store the input bounds in the TorchModel
    _inputBounds = BoundedTensor<torch::Tensor>(
        inputBounds.lower().to(_device),
        inputBounds.upper().to(_device));
    
    log(Stringf("[TorchModel] Input bounds set with shape: %s", 
                inputBounds.lower().sizes().vec().data()));
}


// CONCRETE BOUND STORAGE METHODS

void TorchModel::setConcreteBounds(unsigned nodeIndex, const BoundedTensor<torch::Tensor>& concreteBounds) {
    validateNodeIndex(nodeIndex);

    _concreteBounds[nodeIndex] = concreteBounds;
    log(Stringf("[TorchModel] Set concrete bounds for node %u", nodeIndex));
}

void TorchModel::clearConcreteBounds() {
    _concreteBounds.clear();
    log(Stringf("[TorchModel] Cleared all concrete bounds"));
}

BoundedTensor<torch::Tensor> TorchModel::getConcreteBounds(unsigned nodeIndex) const {
    validateNodeIndex(nodeIndex);
    
    if (!hasConcreteBounds(nodeIndex)) {
        throw LirpaError(LirpaError::INVALID_MODEL_STRUCTURE, 
                          Stringf("Concrete bounds not computed for node %u", nodeIndex).ascii());
    }
    
    return _concreteBounds[nodeIndex];
}

bool TorchModel::hasConcreteBounds(unsigned nodeIndex) const {
    validateNodeIndex(nodeIndex);
    return _concreteBounds.exists(nodeIndex);
}

// INPUT BOUND ACCESS METHODS

BoundedTensor<torch::Tensor> TorchModel::getInputBounds() const {
    if (!hasInputBounds()) {
        throw LirpaError(LirpaError::INVALID_MODEL_STRUCTURE, "No input bounds set");
    }
    return _inputBounds;
}

bool TorchModel::hasInputBounds() const {
    return _inputBounds.lower().defined() && _inputBounds.upper().defined();
}

torch::Tensor TorchModel::getInputLowerBounds() const {
    if (!hasInputBounds()) {
        throw LirpaError(LirpaError::INVALID_MODEL_STRUCTURE, "No input bounds set");
    }
    return _inputBounds.lower();
}

torch::Tensor TorchModel::getInputUpperBounds() const {
    if (!hasInputBounds()) {
        throw LirpaError(LirpaError::INVALID_MODEL_STRUCTURE, "No input bounds set");
    }
    return _inputBounds.upper();
}

// SPECIFICATION MATRIX MANAGEMENT

void TorchModel::setSpecificationMatrix(const torch::Tensor& specMatrix) {
    log(Stringf("[TorchModel] Setting specification matrix with shape [%ld, %ld, %ld]",
                specMatrix.size(0), specMatrix.size(1), specMatrix.size(2)));
    _specificationMatrix = specMatrix.to(_device);
    _hasSpecificationMatrix = true;
    // Thresholds remain empty when setting matrix directly
    _specificationThresholds = torch::Tensor();
    _specificationBranchMapping.clear();
    _specificationBranchSizes.clear();
    _hasORBranches = false;
}

void TorchModel::setSpecificationFromConstraints(const OutputConstraintSet& constraints) {
    log("[TorchModel] Setting specification from OutputConstraintSet");
    
    if (!constraints.hasConstraints()) {
        throw LirpaError(LirpaError::INVALID_MODEL_STRUCTURE, "OutputConstraintSet is empty");
    }
    
    CMatrixResult result = constraints.toCMatrix();
    _specificationMatrix = result.C.to(_device);
    _specificationThresholds = result.thresholds.to(_device);
    _specificationBranchMapping = result.branchMapping;
    _specificationBranchSizes = result.branchSizes;
    _hasORBranches = result.hasORBranches;
    _hasSpecificationMatrix = true;
    
    log(Stringf("[TorchModel] Specification matrix set: shape [%ld, %ld, %ld], %u constraints%s",
                result.C.size(0), result.C.size(1), result.C.size(2), (unsigned)result.thresholds.size(0),
                result.hasORBranches ? Stringf(", %u OR branches", result.branchSizes.size()).ascii() : ""));
}

torch::Tensor TorchModel::getSpecificationMatrix() const {
    if (!_hasSpecificationMatrix) {
        throw LirpaError(LirpaError::INVALID_MODEL_STRUCTURE, "No specification matrix set");
    }
    return _specificationMatrix;
}

torch::Tensor TorchModel::getSpecificationThresholds() const {
    if (!_hasSpecificationMatrix) {
        throw LirpaError(LirpaError::INVALID_MODEL_STRUCTURE, "No specification matrix set");
    }
    return _specificationThresholds;
}

CMatrixResult TorchModel::getSpecificationMatrixResult() const {
    if (!_hasSpecificationMatrix) {
        throw LirpaError(LirpaError::INVALID_MODEL_STRUCTURE, "No specification matrix set");
    }
    CMatrixResult result;
    result.C = _specificationMatrix;
    result.thresholds = _specificationThresholds;
    result.branchMapping = _specificationBranchMapping;
    result.branchSizes = _specificationBranchSizes;
    result.hasORBranches = _hasORBranches;
    return result;
}

bool TorchModel::hasSpecificationMatrix() const {
    return _hasSpecificationMatrix;
}

// Configuration is now accessed via LirpaConfiguration static members
// Removed setAnalysisConfig, getAnalysisConfig, setAnalysisMethod, setVerbose methods

// UNIFIED ANALYSIS ENTRY METHOD

BoundedTensor<torch::Tensor> TorchModel::compute_bounds(
    const BoundedTensor<torch::Tensor>& input_bounds,
    const torch::Tensor* specification_matrix,
    LirpaConfiguration::AnalysisMethod method,
    bool bound_lower,
    bool bound_upper) {

    log("[TorchModel] compute_bounds() called - unified analysis entry point");

    // Set input bounds
    setInputBounds(input_bounds);

    // Set specification matrix if provided
    if (specification_matrix != nullptr) {
        setSpecificationMatrix(*specification_matrix);
        log("[TorchModel] Specification matrix set from compute_bounds parameter");
    }
    // If no specification matrix provided but one is already set in TorchModel, it will be used by CROWN analysis

    // Update LirpaConfiguration with method and bound flags
    LirpaConfiguration::ANALYSIS_METHOD = method;
    LirpaConfiguration::COMPUTE_LOWER = bound_lower;
    LirpaConfiguration::COMPUTE_UPPER = bound_upper;

    // Dispatch to appropriate analysis method based on configuration
    BoundedTensor<torch::Tensor> result;

    if (method == LirpaConfiguration::AnalysisMethod::CROWN) {
        log("[TorchModel] Running CROWN analysis via compute_bounds");
        result = runCROWN(input_bounds);
    } else if (method == LirpaConfiguration::AnalysisMethod::AlphaCROWN) {
        log("[TorchModel] Running AlphaCROWN analysis via compute_bounds");

        // Configure AlphaCROWN optimization flags
        bool optimizeLower = bound_lower && LirpaConfiguration::OPTIMIZE_LOWER;
        bool optimizeUpper = bound_upper && LirpaConfiguration::OPTIMIZE_UPPER;

        log(Stringf("[TorchModel] AlphaCROWN config: optimize_lower=%s, optimize_upper=%s",
                    optimizeLower ? "true" : "false", optimizeUpper ? "true" : "false"));

        result = runAlphaCROWN(input_bounds, optimizeLower, optimizeUpper);
    } else {
        throw LirpaError(LirpaError::INVALID_MODEL_STRUCTURE, "Unknown analysis method");
    }

    log("[TorchModel] compute_bounds() completed successfully");
    return result;
}

// ANALYSIS ENTRY METHODS

BoundedTensor<torch::Tensor> TorchModel::runCROWN() {
    if (!hasInputBounds()) {
        throw LirpaError(LirpaError::INVALID_MODEL_STRUCTURE, "Input bounds must be set before running CROWN analysis");
    }
    return runCROWN(_inputBounds);
}

BoundedTensor<torch::Tensor> TorchModel::runCROWN(const BoundedTensor<torch::Tensor>& inputBounds) {
    log("[TorchModel] Running CROWN analysis");

    // Cache node shapes using a single center-point forward pass to avoid shape inference
    // heuristics in Conv/BN during backward bound propagation.
    setInputBounds(inputBounds);
    cacheForwardShapesFromCenter();

    // Create CROWN analysis instance
    std::unique_ptr<CROWNAnalysis> crownAnalysis = std::make_unique<CROWNAnalysis>(this);

    // Set input bounds and run analysis
    crownAnalysis->setInputBounds(inputBounds);
    crownAnalysis->run();

    // Get output bounds
    BoundedTensor<torch::Tensor> outputBounds = crownAnalysis->getOutputBounds();

    // Store final analysis bounds
    setFinalAnalysisBounds(outputBounds);

    log("[TorchModel] CROWN analysis completed");
    return outputBounds;
}

void TorchModel::cacheForwardShapesFromCenter() {
    if (!hasInputBounds()) return;

    torch::NoGradGuard no_grad;
    torch::Tensor lb = _inputBounds.lower().to(torch::kFloat32);
    torch::Tensor ub = _inputBounds.upper().to(torch::kFloat32);
    if (!lb.defined() || !ub.defined()) return;

    torch::Tensor center = (lb + ub) / 2.0;

    // Heuristic reshape to common image formats (matches RunFullOptimized.cpp).
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

    try {
        // This will call each node's forward() and populate any internal cached shapes.
        (void)forwardAndStoreActivations(center);
    } catch (...) {
        // Shape caching is a best-effort optimization; do not fail analysis if it breaks.
        return;
    }
}


// NEW REFACTORED ALPHA-CROWN IMPLEMENTATION - Pure delegator pattern
BoundedTensor<torch::Tensor> TorchModel::runAlphaCROWN(bool optimizeLower, bool optimizeUpper) {
    if (!hasInputBounds()) {
        throw LirpaError(LirpaError::INVALID_MODEL_STRUCTURE, "Input bounds must be set before running AlphaCROWN analysis");
    }
    return runAlphaCROWN(_inputBounds, optimizeLower, optimizeUpper);
}

BoundedTensor<torch::Tensor> TorchModel::runAlphaCROWN(const BoundedTensor<torch::Tensor>& inputBounds,
                                                        bool optimizeLower, bool optimizeUpper) {
    log("[TorchModel] Running AlphaCROWN analysis - Delegating to AlphaCROWNAnalysis");

    // Create AlphaCROWN analysis instance
    std::unique_ptr<AlphaCROWNAnalysis> alphaCrownAnalysis = std::make_unique<AlphaCROWNAnalysis>(this);

    // Set input bounds on the internal CROWN analysis
    alphaCrownAnalysis->getCROWNAnalysis()->setInputBounds(inputBounds);

    // Initialize bound tensors
    torch::Tensor finalLower, finalUpper;

    // Call AlphaCROWN to compute optimized upper bounds (if requested)
    if (optimizeUpper) {
        log("[TorchModel] Requesting optimized upper bounds from AlphaCROWNAnalysis");
        finalUpper = alphaCrownAnalysis->computeOptimizedBounds(LirpaConfiguration::BoundSide::Upper);
        log("[TorchModel] Received optimized upper bounds");
    }

    // Call AlphaCROWN to compute optimized lower bounds (if requested)
    if (optimizeLower) {
        log("[TorchModel] Requesting optimized lower bounds from AlphaCROWNAnalysis");
        finalLower = alphaCrownAnalysis->computeOptimizedBounds(LirpaConfiguration::BoundSide::Lower);
        log("[TorchModel] Received optimized lower bounds");
    }

    // If neither optimization was requested, fall back to CROWN
    if (!optimizeLower && !optimizeUpper) {
        log("[TorchModel] No optimization requested, delegating to CROWN analysis");
        return runCROWN(inputBounds);
    }

    // If only one bound was requested, use CROWN for the other
    if (!optimizeLower) {
        log("[TorchModel] Computing lower bounds with CROWN (not optimized)");
        auto crownBounds = runCROWN(inputBounds);
        finalLower = crownBounds.lower();
    }
    if (!optimizeUpper) {
        log("[TorchModel] Computing upper bounds with CROWN (not optimized)");
        auto crownBounds = runCROWN(inputBounds);
        finalUpper = crownBounds.upper();
    }

    // Create final bounded tensor
    BoundedTensor<torch::Tensor> outputBounds(finalLower, finalUpper);

    // Store final analysis bounds
    setFinalAnalysisBounds(outputBounds);

    log("[TorchModel] AlphaCROWN analysis completed");
    return outputBounds;
}

// ANALYSIS BOUNDS STORAGE

void TorchModel::setCROWNBounds(unsigned nodeIndex, const BoundedTensor<torch::Tensor>& bounds) {
    validateNodeIndex(nodeIndex);
    _crownBounds[nodeIndex] = bounds;
    log(Stringf("[TorchModel] Set CROWN bounds for node %u", nodeIndex));
}

void TorchModel::setAlphaCROWNBounds(unsigned nodeIndex, const BoundedTensor<torch::Tensor>& bounds) {
    validateNodeIndex(nodeIndex);
    _alphaCrownBounds[nodeIndex] = bounds;
    log(Stringf("[TorchModel] Set AlphaCROWN bounds for node %u", nodeIndex));
}

BoundedTensor<torch::Tensor> TorchModel::getCROWNBounds(unsigned nodeIndex) const {
    validateNodeIndex(nodeIndex);
    if (!_crownBounds.exists(nodeIndex)) {
        throw LirpaError(LirpaError::INVALID_MODEL_STRUCTURE,
                          Stringf("No CROWN bounds available for node %u", nodeIndex).ascii());
    }
    return _crownBounds.at(nodeIndex);
}

BoundedTensor<torch::Tensor> TorchModel::getAlphaCROWNBounds(unsigned nodeIndex) const {
    validateNodeIndex(nodeIndex);
    if (!_alphaCrownBounds.exists(nodeIndex)) {
        throw LirpaError(LirpaError::INVALID_MODEL_STRUCTURE,
                          Stringf("No AlphaCROWN bounds available for node %u", nodeIndex).ascii());
    }
    return _alphaCrownBounds.at(nodeIndex);
}

bool TorchModel::hasCROWNBounds(unsigned nodeIndex) const {
    return nodeIndex < _nodes.size() && _crownBounds.exists(nodeIndex);
}

bool TorchModel::hasAlphaCROWNBounds(unsigned nodeIndex) const {
    return nodeIndex < _nodes.size() && _alphaCrownBounds.exists(nodeIndex);
}

// FINAL ANALYSIS BOUNDS

void TorchModel::setFinalAnalysisBounds(const BoundedTensor<torch::Tensor>& bounds) {
    _finalAnalysisBounds = bounds;
    _hasFinalAnalysisBounds = true;
    log("[TorchModel] Set final analysis bounds");
}

BoundedTensor<torch::Tensor> TorchModel::getFinalAnalysisBounds() const {
    if (!_hasFinalAnalysisBounds) {
        throw LirpaError(LirpaError::INVALID_MODEL_STRUCTURE, "No final analysis bounds available");
    }
    return _finalAnalysisBounds;
}

bool TorchModel::hasFinalAnalysisBounds() const {
    return _hasFinalAnalysisBounds;
}
 
void TorchModel::validateNodeIndex(unsigned nodeIndex) const {
    if (nodeIndex >= _nodes.size()) {
        throw LirpaError(LirpaError::INVALID_MODEL_STRUCTURE,
                          Stringf("Node index %u out of bounds for model with %u nodes", nodeIndex, _nodes.size()).ascii());
    }
}

} // namespace NLR