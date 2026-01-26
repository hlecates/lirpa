#ifndef __OnnxToTorchParser_h__
#define __OnnxToTorchParser_h__

#include "Map.h"
#include "Set.h"
#include "MString.h"
#include "Vector.h"
#include "onnx.proto3.pb.h"
#include "../engine/nodes/BoundedTorchNode.h"
#include "../engine/nodes/BoundedConstantNode.h"
#include "../engine/nodes/BoundedInputNode.h"
#include "../engine/nodes/BoundedLinearNode.h"
#include "../engine/nodes/BoundedReLUNode.h"
#include "../engine/nodes/BoundedIdentityNode.h"
#include "../engine/nodes/BoundedReshapeNode.h"
#include "../engine/nodes/BoundedFlattenNode.h"
#include "../engine/nodes/BoundedSubNode.h"
#include "../engine/nodes/BoundedBatchNormNode.h"

// Undefine Warning macro to avoid conflict with PyTorch
#ifdef Warning
#undef Warning
#endif

#include <torch/torch.h>
#include <memory>

// Forward declarations
namespace NLR {
    class TorchModel;
}

using TensorShape = Vector<unsigned int>;

// Error handling functions for OnnxToTorch parser
void onnxToTorchMissingAttributeError(const onnx::NodeProto &node, const String &attributeName);
void onnxToTorchUnimplementedOperationError(const onnx::NodeProto &node);
void onnxToTorchUnimplementedAttributeError(const onnx::NodeProto &node, const String &attributeName);
void onnxToTorchUnsupportedOperationError(const onnx::NodeProto &node);
void onnxToTorchMissingNodeError(const String &missingNodeName);
void onnxToTorchUnexpectedNumberOfInputs(const onnx::NodeProto &node, 
                                        unsigned int actualNumberOfInputs,
                                        unsigned int lowerBound, 
                                        unsigned int upperBound);
void onnxToTorchInvalidTensorShapeError(const String &nodeName, const String &reason);
void onnxToTorchUnsupportedDataTypeError(const onnx::TensorProto_DataType &dataType);
void onnxToTorchInvalidConstantNodeError(const onnx::NodeProto &node, const String &reason);
void onnxToTorchTopologicalSortError(const String &reason);
void onnxToTorchBoundedModuleCreationError(const String &operationType, const String &reason);
void onnxToTorchFileReadError(const String &filename, const String &reason);
void onnxToTorchModelParseError(const String &filename, const String &reason);
void onnxToTorchGraphProcessingError(const String &reason);
void onnxToTorchTensorConversionError(const String &tensorName, const String &reason);
void onnxToTorchAttributeProcessingError(const onnx::NodeProto &node, const String &attributeName, const String &reason);
void onnxToTorchShapeMismatchError(const String &operation, const TensorShape &expectedShape, const TensorShape &actualShape);
void onnxToTorchDimensionMismatchError(const String &operation, unsigned int expectedDim, unsigned int actualDim);
void onnxToTorchInvalidBroadcastError(const String &operation, const TensorShape &shape1, const TensorShape &shape2);
void onnxToTorchUnsupportedActivationError(const String &activationType);
void onnxToTorchInvalidWeightBiasError(const String &operation, const String &reason);
void onnxToTorchMemoryAllocationError(const String &operation, const String &reason);
void onnxToTorchPyTorchError(const String &operation, const String &pytorchError);

class OnnxToTorchParser
{
public:
    static std::shared_ptr<NLR::TorchModel> parse(const String &path);
private:
    OnnxToTorchParser(const String &path);
    std::shared_ptr<NLR::TorchModel> processGraph();
    onnx::ModelProto _onnx_model;
};


namespace AttributeUtils {
    Map<String, torch::IValue> extractAttributes(onnx::NodeProto &node);
    float getFloatAttribute(const onnx::NodeProto &node, const String &name, float defaultValue = 0.0f);
    int getIntAttribute(const onnx::NodeProto &node, const String &name, int defaultValue = 0);
    Vector<int> getIntsAttribute(onnx::NodeProto &node, const String &name, const Vector<int> &defaultValue = {});
    String getStringAttribute(onnx::NodeProto &node, const String &name, const String &defaultValue = "");
}

namespace GraphUtils {
    Vector<String> computeTopologicalOrder(
        const Map<String, onnx::NodeProto>& name_to_node,
        const Map<String, onnx::ValueInfoProto>& name_to_input,
        const Map<String, onnx::TensorProto>& name_to_initializer
    );

    Map<String, Set<String>> computeActivationDependencies(const onnx::GraphProto& graph);

    std::vector<int64_t> instantiateReshapeTemplate(
        const torch::Tensor& input, 
        const torch::Tensor& shape_tensor
    );
}

namespace ConstantProcessor {
    torch::Tensor processInitializer(const onnx::TensorProto& tensor);
    torch::Tensor processConstantNode(const onnx::NodeProto& node);
}

// New namespace for bounded node conversion
namespace BoundedOperationConverter {
    // Helper functions for shape extraction
    TensorShape extractShapeFromNode(const onnx::NodeProto& node, 
                                   const Map<String, onnx::ValueInfoProto>& name_to_input,
                                   const Map<String, onnx::TensorProto>& name_to_initializer,
                                   const String& tensorName);
    unsigned computeTensorSize(const TensorShape& shape);
    
    std::shared_ptr<NLR::BoundedTorchNode> convertGemm(const onnx::NodeProto& node,
                                                     const Map<String, torch::Tensor>& constants,
                                                     const Map<String, onnx::ValueInfoProto>& name_to_input,
                                                     const Map<String, onnx::TensorProto>& name_to_initializer);
    std::shared_ptr<NLR::BoundedTorchNode> convertMatMul(const onnx::NodeProto& node,
                                                        const Map<String, torch::Tensor>& constants,
                                                        const Map<String, onnx::ValueInfoProto>& name_to_input,
                                                        const Map<String, onnx::TensorProto>& name_to_initializer);
    std::shared_ptr<NLR::BoundedTorchNode> convertAdd(const onnx::NodeProto& node,
                                                     const Map<String, torch::Tensor>& constants,
                                                     const Map<String, onnx::ValueInfoProto>& name_to_input,
                                                     const Map<String, onnx::TensorProto>& name_to_initializer,
                                                     const Vector<std::shared_ptr<NLR::BoundedTorchNode>>& existingNodes,
                                                     const Map<String, unsigned>& nameToIndex);
    std::shared_ptr<NLR::BoundedTorchNode> convertRelu(const onnx::NodeProto& node,
                                                     const Map<String, onnx::ValueInfoProto>& name_to_input,
                                                     const Map<String, onnx::TensorProto>& name_to_initializer);
    std::shared_ptr<NLR::BoundedTorchNode> convertSigmoid(const onnx::NodeProto& node,
                                                     const Map<String, onnx::ValueInfoProto>& name_to_input,
                                                     const Map<String, onnx::TensorProto>& name_to_initializer);
    std::shared_ptr<NLR::BoundedTorchNode> convertIdentity(const onnx::NodeProto& node,
                                                         const Map<String, onnx::ValueInfoProto>& name_to_input,
                                                         const Map<String, onnx::TensorProto>& name_to_initializer);
    std::shared_ptr<NLR::BoundedTorchNode> convertReshape(const onnx::NodeProto& node,
                                                        const Map<String, onnx::ValueInfoProto>& name_to_input,
                                                        const Map<String, onnx::TensorProto>& name_to_initializer);
    std::shared_ptr<NLR::BoundedTorchNode> convertFlatten(const onnx::NodeProto& node,
                                                        const Map<String, onnx::ValueInfoProto>& name_to_input,
                                                        const Map<String, onnx::TensorProto>& name_to_initializer);
    std::shared_ptr<NLR::BoundedTorchNode> convertSub(const onnx::NodeProto& node,
                                                     const Map<String, torch::Tensor>& constants,
                                                     const Map<String, onnx::ValueInfoProto>& name_to_input,
                                                     const Map<String, onnx::TensorProto>& name_to_initializer);
    std::shared_ptr<NLR::BoundedTorchNode> convertConv(const onnx::NodeProto& node,
                                                     const Map<String, torch::Tensor>& constants,
                                                     const Map<String, onnx::ValueInfoProto>& name_to_input,
                                                     const Map<String, onnx::TensorProto>& name_to_initializer,
                                                     const Map<String, Vector<int>>& shape_metadata = Map<String, Vector<int>>());
    std::shared_ptr<NLR::BoundedTorchNode> convertConvTranspose(const onnx::NodeProto& node,
                                                     const Map<String, torch::Tensor>& constants,
                                                     const Map<String, onnx::ValueInfoProto>& name_to_input,
                                                     const Map<String, onnx::TensorProto>& name_to_initializer);
    std::shared_ptr<NLR::BoundedTorchNode> convertBatchNormalization(const onnx::NodeProto& node,
                                                     const Map<String, torch::Tensor>& constants,
                                                     const Map<String, onnx::ValueInfoProto>& name_to_input,
                                                     const Map<String, onnx::TensorProto>& name_to_initializer);
    std::shared_ptr<NLR::BoundedTorchNode> convertConcat(const onnx::NodeProto& node,
                                                     const Map<String, onnx::ValueInfoProto>& name_to_input,
                                                     const Map<String, onnx::TensorProto>& name_to_initializer,
                                                     const Map<String, Vector<int>>& shape_metadata);
    std::shared_ptr<NLR::BoundedTorchNode> convertSlice(const onnx::NodeProto& node,
                                                        const Map<String, torch::Tensor>& constants,
                                                        const Map<String, onnx::ValueInfoProto>& name_to_input,
                                                        const Map<String, onnx::TensorProto>& name_to_initializer,
                                                        const Map<String, Vector<int>>& shape_metadata);
    std::shared_ptr<NLR::BoundedTorchNode> convertConstant(const torch::Tensor& value);
}

#endif // __OnnxToTorchParser_h__