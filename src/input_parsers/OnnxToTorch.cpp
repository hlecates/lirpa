// Handles ONNX file parsing and entry to TorchModel

/*
 * Important Operations to Add (Before integrating to pipeline):
 * Mul/Div
 * Sigmoid
 *
 * Supported Operations:
 * Identity
 * Reshape
 * Gemm
 * MatMul
 * Add (element-wise, fused with Linear when possible)
 * Sub (element-wise)
 * Relu
 * Conv (2D Convolution)
 * Dropout (treated as Identity during inference)
 */


#include "OnnxToTorch.h"
#include "../engine/TorchModel.h"
#include "../engine/nodes/BoundedTorchNode.h"
#include "../engine/nodes/BoundedConstantNode.h"
#include "../engine/nodes/BoundedInputNode.h"
#include "../engine/nodes/BoundedLinearNode.h"
#include "../engine/nodes/BoundedReLUNode.h"
#include "../engine/nodes/BoundedSigmoidNode.h"
#include "../engine/nodes/BoundedIdentityNode.h"
#include "../engine/nodes/BoundedReshapeNode.h"
#include "../engine/nodes/BoundedSubNode.h"
#include "../engine/nodes/BoundedAddNode.h"
#include "../engine/nodes/BoundedConvNode.h"
#include "../engine/nodes/BoundedConvTransposeNode.h"
#include "../engine/nodes/BoundedConcatNode.h"
#include "../engine/nodes/BoundedSliceNode.h"
#include "../engine/LirpaError.h"
#include "File.h"
#include "MString.h"
#include "Vector.h"
#include "Map.h"
#include "Set.h"
#include <fstream>
#include <memory>
#include <cstdlib>
#include <cstdio>
#include <torch/torch.h>
#include <onnx.proto3.pb.h>

#include "TensorUtils.h"

#include <math.h>

#include "Debug.h"


void onnxToTorchMissingAttributeError(const onnx::NodeProto &node, const String &attributeName)
{
    String errorMessage = Stringf("OnnxToTorch: Onnx node of type %s is missing the expected attribute %s",
                                   node.op_type().c_str(),
                                   attributeName.ascii());
    throw LirpaError(LirpaError::ONNX_PARSING_ERROR, errorMessage.ascii());
}

void onnxToTorchUnimplementedOperationError(const onnx::NodeProto &node)
{
    String errorMessage = Stringf("OnnxToTorch: Onnx '%s' operation not yet implemented for TorchModel conversion. Should be relatively easy to add.",
                                   node.op_type().c_str());
    throw LirpaError(LirpaError::ONNX_PARSING_ERROR, errorMessage.ascii());
}

void onnxToTorchUnimplementedAttributeError(const onnx::NodeProto &node, const String &attributeName)
{
    String errorMessage = Stringf("OnnxToTorch: Onnx '%s' operation with non-default value for attribute '%s' not yet supported.",
                                   node.op_type().c_str(),
                                   attributeName.ascii());
    throw LirpaError(LirpaError::ONNX_PARSING_ERROR, errorMessage.ascii());
}

void onnxToTorchUnsupportedOperationError(const onnx::NodeProto &node)
{
    String errorMessage = Stringf("OnnxToTorch: Onnx operation %s not currently supported by TorchModel conversion",
                                   node.op_type().c_str());
    throw LirpaError(LirpaError::ONNX_PARSING_ERROR, errorMessage.ascii());
}

void onnxToTorchMissingNodeError(const String &missingNodeName)
{
    String errorMessage = Stringf("OnnxToTorch: Internal invariant violated: missing node '%s' not found",
                                   missingNodeName.ascii());
    throw LirpaError(LirpaError::ONNX_PARSING_ERROR, errorMessage.ascii());
}

void onnxToTorchUnexpectedNumberOfInputs(const onnx::NodeProto &node,
                                        unsigned int actualNumberOfInputs,
                                        unsigned int lowerBound,
                                        unsigned int upperBound)
{
    String errorMessage;
    if (lowerBound == upperBound)
    {
        errorMessage = Stringf("OnnxToTorch: %s node expected to have exactly %d inputs, but found %d",
                                node.op_type().c_str(),
                                lowerBound,
                                actualNumberOfInputs);
    }
    else
    {
        errorMessage = Stringf("OnnxToTorch: %s node expected to have between %d and %d inputs, but found %d",
                                node.op_type().c_str(),
                                lowerBound,
                                upperBound,
                                actualNumberOfInputs);
    }
    throw LirpaError(LirpaError::ONNX_PARSING_ERROR, errorMessage.ascii());
}

void onnxToTorchInvalidTensorShapeError(const String &nodeName, const String &reason)
{
    String errorMessage = Stringf("OnnxToTorch: Invalid tensor shape for node '%s': %s",
                                   nodeName.ascii(),
                                   reason.ascii());
    throw LirpaError(LirpaError::ONNX_PARSING_ERROR, errorMessage.ascii());
}

void onnxToTorchUnsupportedDataTypeError(const onnx::TensorProto_DataType &dataType)
{
    String errorMessage = Stringf("OnnxToTorch: Support for Onnx constants of type '%s' not yet implemented.",
                                   TensorProto_DataType_Name(dataType).c_str());
    throw LirpaError(LirpaError::ONNX_PARSING_ERROR, errorMessage.ascii());
}

void onnxToTorchInvalidConstantNodeError(const onnx::NodeProto &node, const String &reason)
{
    String errorMessage = Stringf("OnnxToTorch: Invalid Constant node '%s': %s",
                                   node.name().c_str(),
                                   reason.ascii());
    throw LirpaError(LirpaError::ONNX_PARSING_ERROR, errorMessage.ascii());
}

void onnxToTorchTopologicalSortError(const String &reason)
{
    String errorMessage = Stringf("OnnxToTorch: Topological sort failed: %s",
                                   reason.ascii());
    throw LirpaError(LirpaError::ONNX_PARSING_ERROR, errorMessage.ascii());
}

void onnxToTorchBoundedModuleCreationError(const String &operationType, const String &reason)
{
    String errorMessage = Stringf("OnnxToTorch: Failed to create bounded module for operation '%s': %s",
                                   operationType.ascii(),
                                   reason.ascii());
    throw LirpaError(LirpaError::ONNX_PARSING_ERROR, errorMessage.ascii());
}

void onnxToTorchFileReadError(const String &filename, const String &reason)
{
    String errorMessage = Stringf("OnnxToTorch: Failed to read file '%s': %s",
                                   filename.ascii(),
                                   reason.ascii());
    throw LirpaError(LirpaError::ONNX_PARSING_ERROR, errorMessage.ascii());
}

void onnxToTorchModelParseError(const String &filename, const String &reason)
{
    String errorMessage = Stringf("OnnxToTorch: Failed to parse ONNX model from file '%s': %s",
                                   filename.ascii(),
                                   reason.ascii());
    throw LirpaError(LirpaError::ONNX_PARSING_ERROR, errorMessage.ascii());
}

void onnxToTorchGraphProcessingError(const String &reason)
{
    String errorMessage = Stringf("OnnxToTorch: Graph processing failed: %s",
                                   reason.ascii());
    throw LirpaError(LirpaError::ONNX_PARSING_ERROR, errorMessage.ascii());
}

void onnxToTorchTensorConversionError(const String &tensorName, const String &reason)
{
    String errorMessage = Stringf("OnnxToTorch: Failed to convert tensor '%s': %s",
                                   tensorName.ascii(),
                                   reason.ascii());
    throw LirpaError(LirpaError::ONNX_PARSING_ERROR, errorMessage.ascii());
}

void onnxToTorchAttributeProcessingError(const onnx::NodeProto &node, const String &attributeName, const String &reason)
{
    String errorMessage = Stringf("OnnxToTorch: Failed to process attribute '%s' for node '%s': %s",
                                   attributeName.ascii(),
                                   node.op_type().c_str(),
                                   reason.ascii());
    throw LirpaError(LirpaError::ONNX_PARSING_ERROR, errorMessage.ascii());
}

void onnxToTorchShapeMismatchError(const String &operation, const TensorShape &expectedShape, const TensorShape &actualShape)
{
    String expectedStr = "[";
    for (unsigned int i = 0; i < expectedShape.size(); ++i) {
        if (i > 0) expectedStr += ", ";
        expectedStr += Stringf("%u", expectedShape[i]);
    }
    expectedStr += "]";
    
    String actualStr = "[";
    for (unsigned int i = 0; i < actualShape.size(); ++i) {
        if (i > 0) actualStr += ", ";
        actualStr += Stringf("%u", actualShape[i]);
    }
    actualStr += "]";
    
    String errorMessage = Stringf("OnnxToTorch: Shape mismatch in operation '%s': expected %s, got %s",
                                   operation.ascii(),
                                   expectedStr.ascii(),
                                   actualStr.ascii());
    throw LirpaError(LirpaError::ONNX_PARSING_ERROR, errorMessage.ascii());
}

void onnxToTorchDimensionMismatchError(const String &operation, unsigned int expectedDim, unsigned int actualDim)
{
    String errorMessage = Stringf("OnnxToTorch: Dimension mismatch in operation '%s': expected %u dimensions, got %u",
                                   operation.ascii(),
                                   expectedDim,
                                   actualDim);
    throw LirpaError(LirpaError::ONNX_PARSING_ERROR, errorMessage.ascii());
}

void onnxToTorchInvalidBroadcastError(const String &operation, const TensorShape &shape1, const TensorShape &shape2)
{
    String shape1Str = "[";
    for (unsigned int i = 0; i < shape1.size(); ++i) {
        if (i > 0) shape1Str += ", ";
        shape1Str += Stringf("%u", shape1[i]);
    }
    shape1Str += "]";
    
    String shape2Str = "[";
    for (unsigned int i = 0; i < shape2.size(); ++i) {
        if (i > 0) shape2Str += ", ";
        shape2Str += Stringf("%u", shape2[i]);
    }
    shape2Str += "]";
    
    String errorMessage = Stringf("OnnxToTorch: Invalid broadcast in operation '%s': cannot broadcast shapes %s and %s",
                                   operation.ascii(),
                                   shape1Str.ascii(),
                                   shape2Str.ascii());
    throw LirpaError(LirpaError::ONNX_PARSING_ERROR, errorMessage.ascii());
}

void onnxToTorchUnsupportedActivationError(const String &activationType)
{
    String errorMessage = Stringf("OnnxToTorch: Unsupported activation function '%s'",
                                   activationType.ascii());
    throw LirpaError(LirpaError::ONNX_PARSING_ERROR, errorMessage.ascii());
}

void onnxToTorchInvalidWeightBiasError(const String &operation, const String &reason)
{
    String errorMessage = Stringf("OnnxToTorch: Invalid weight/bias configuration in operation '%s': %s",
                                   operation.ascii(),
                                   reason.ascii());
    throw LirpaError(LirpaError::ONNX_PARSING_ERROR, errorMessage.ascii());
}

void onnxToTorchMemoryAllocationError(const String &operation, const String &reason)
{
    String errorMessage = Stringf("OnnxToTorch: Memory allocation failed in operation '%s': %s",
                                   operation.ascii(),
                                   reason.ascii());
    throw LirpaError(LirpaError::ONNX_PARSING_ERROR, errorMessage.ascii());
}

void onnxToTorchPyTorchError(const String &operation, const String &pytorchError)
{
    String errorMessage = Stringf("OnnxToTorch: PyTorch error in operation '%s': %s",
                                   operation.ascii(),
                                   pytorchError.ascii());
    throw LirpaError(LirpaError::ONNX_PARSING_ERROR, errorMessage.ascii());
}

//Public
using namespace Operations;
using namespace AttributeUtils;

using TensorShape = Vector<unsigned int>;

OnnxToTorchParser::OnnxToTorchParser(const String &path) {
    // std::cerr << "[OnnxToTorchParser] Loading ONNX file: " << path << std::endl << std::flush;
    std::ifstream input(path.ascii(), std::ios::ate | std::ios::binary);
    if (!input.is_open()) {
        onnxToTorchFileReadError(path, "Could not open file");
    }
    std::streamsize size = input.tellg();
    // std::cerr << "[OnnxToTorchParser] File size: " << size << " bytes" << std::endl;

    input.seekg(0, std::ios::beg);

    // Use std::vector instead of Vector to ensure proper memory handling
    std::vector<char> buffer(size);
    input.read(buffer.data(), size);
    if (input.gcount() != size) {
        onnxToTorchFileReadError(path, Stringf("Failed to read entire file: expected %ld bytes, got %ld",
                                               (long)size, (long)input.gcount()));
    }

    // Parse using string to better handle raw_data
    std::string model_string(buffer.data(), size);
    if (!_onnx_model.ParseFromString(model_string)) {
        onnxToTorchModelParseError(path, "Failed to parse ONNX protobuf");
    }
    // std::cerr << "[OnnxToTorchParser] Successfully parsed ONNX model with " << _onnx_model.graph().node_size() << " nodes" << std::endl;
    // std::cerr << "[OnnxToTorchParser] Model has " << _onnx_model.graph().initializer_size() << " initializers" << std::endl;

    // Run ONNX shape inference using Python if no intermediate shapes are present
    if (_onnx_model.graph().value_info_size() == 0) {
        // Create temporary output path
        std::string temp_path = std::string(path.ascii()) + ".inferred.onnx";
        
        // Build Python command to run shape inference
        std::string python_cmd = "python3 -c \"import onnx; from onnx import shape_inference; "
                                "model = onnx.load('" + std::string(path.ascii()) + "'); "
                                "inferred = shape_inference.infer_shapes(model); "
                                "onnx.save(inferred, '" + temp_path + "')\" 2>/dev/null";
        
        int result = system(python_cmd.c_str());
        if (result == 0) {
            // Reload the inferred model
            std::ifstream inferred_input(temp_path, std::ios::ate | std::ios::binary);
            if (inferred_input.is_open()) {
                std::streamsize inferred_size = inferred_input.tellg();
                inferred_input.seekg(0, std::ios::beg);
                std::vector<char> inferred_buffer(inferred_size);
                inferred_input.read(inferred_buffer.data(), inferred_size);
                
                std::string inferred_string(inferred_buffer.data(), inferred_size);
                _onnx_model.ParseFromString(inferred_string);
                inferred_input.close();
                
                // Clean up temporary file
                std::remove(temp_path.c_str());
            }
        }
    }

    // Debug: check raw_data for each initializer immediately after parsing (commented out)
    // for (const auto& init : _onnx_model.graph().initializer()) {
    //     if (init.name() == "1" || init.name() == "3") {  // Conv weights for CIFAR
    //         std::cerr << "[OnnxToTorchParser] Conv weight " << init.name()
    //                   << " raw_data size: " << init.raw_data().size() << " bytes" << std::endl;
    //     }
    // }

    // Debug: Print all nodes and their attributes
    for (int i = 0; i < _onnx_model.graph().node_size(); ++i) {
        const auto& node = _onnx_model.graph().node(i);
        // std::cerr << "[OnnxToTorchParser] Node " << i << " op_type: " << node.op_type() << std::endl << std::flush;
        for (int j = 0; j < node.attribute_size(); ++j) {
            // const auto& attr = node.attribute(j);
            // std::cerr << "[OnnxToTorchParser]   Attribute " << j << " name: " << attr.name() << " type: " << attr.type() << std::endl << std::flush;
            (void)j; // Suppress unused variable warning
        }
    }
}

std::shared_ptr<NLR::TorchModel> OnnxToTorchParser::parse(const String &path) {
    OnnxToTorchParser parser(path);
    return parser.processGraph();
}

std::shared_ptr<NLR::TorchModel> OnnxToTorchParser::processGraph() {
    // Process initializers
    Map<String, onnx::TensorProto> name_to_initializer;
    for (const auto& initializer : _onnx_model.graph().initializer()) {
        name_to_initializer[initializer.name()] = initializer;
    }

    // Process inputs
    Map<String, onnx::ValueInfoProto> name_to_input;
    for (const auto& input : _onnx_model.graph().input()) {
        name_to_input[input.name()] = input;
    }

    // Process nodes - store by output name for lookup
    Map<String, onnx::NodeProto> name_to_node;
    for (const auto& node : _onnx_model.graph().node()) {
        for (int i = 0; i < node.output_size(); ++i) {
            name_to_node[node.output(i)] = node;
        }
    }
    
    // Extract shape metadata from ONNX value_info
    // This includes shapes for intermediate tensors
    Map<String, Vector<int>> shape_metadata;
    
    // Add input shapes
    for (const auto& input : _onnx_model.graph().input()) {
        if (input.type().tensor_type().has_shape()) {
            const auto& shape = input.type().tensor_type().shape();
            Vector<int> dims;
            for (int i = 0; i < shape.dim_size(); ++i) {
                if (shape.dim(i).has_dim_value()) {
                    dims.append(static_cast<int>(shape.dim(i).dim_value()));
                } else {
                    // Dynamic dimension - use 1 as default (typically batch dimension)
                    dims.append(1);
                }
            }
            if (!dims.empty()) {
                shape_metadata[input.name()] = dims;
            }
        }
    }
    
    // Add output shapes
    for (const auto& output : _onnx_model.graph().output()) {
        if (output.type().tensor_type().has_shape()) {
            const auto& shape = output.type().tensor_type().shape();
            Vector<int> dims;
            for (int i = 0; i < shape.dim_size(); ++i) {
                if (shape.dim(i).has_dim_value()) {
                    dims.append(static_cast<int>(shape.dim(i).dim_value()));
                } else {
                    dims.append(-1);
                }
            }
            if (!dims.empty()) {
                shape_metadata[output.name()] = dims;
            }
        }
    }
    
    // FIRST PASS: Infer shapes for all operations
    // This ensures Concat and other operations have access to all input shapes
    // Iterate multiple times to handle dependencies
    for (int pass = 0; pass < 3; ++pass) {
        int shapes_added = 0;
        
        for (const auto& node : _onnx_model.graph().node()) {
            if (node.output_size() > 0) {
                String outputName = node.output(0);
                
                // Skip if already computed
                if (shape_metadata.exists(outputName)) continue;
                
                Vector<int> output_shape;
                
                if (node.op_type() == "MatMul" || node.op_type() == "Gemm") {
                    if (node.input_size() >= 2) {
                        Vector<int> input0_shape = shape_metadata.exists(node.input(0)) ? 
                                                   shape_metadata[node.input(0)] : Vector<int>();
                        String weight_name = node.input(1);
                        
                        if (!input0_shape.empty() && name_to_initializer.exists(weight_name)) {
                            const auto& weight_tensor = name_to_initializer[weight_name];
                            if (weight_tensor.dims_size() == 2) {
                                int batch = input0_shape[0];
                                int N = weight_tensor.dims(1);
                                output_shape.append(batch);
                                output_shape.append(N);
                            }
                        }
                    }
                } else if (node.op_type() == "Relu" || node.op_type() == "Sigmoid" || 
                           node.op_type() == "Dropout" || node.op_type() == "BatchNormalization") {
                    // Shape-preserving operations
                    if (node.input_size() > 0 && shape_metadata.exists(node.input(0))) {
                        output_shape = shape_metadata[node.input(0)];
                    }
                } else if (node.op_type() == "Conv") {
                    // Compute Conv output shape
                    if (node.input_size() >= 2 && shape_metadata.exists(node.input(0))) {
                        Vector<int> input_shape = shape_metadata[node.input(0)];
                        String weight_name = node.input(1);
                        
                        if (name_to_initializer.exists(weight_name) && input_shape.size() >= 3) {
                            const auto& weight_tensor = name_to_initializer[weight_name];
                            int weight_dims = weight_tensor.dims_size();
                            
                            // Get conv attributes
                            std::vector<int64_t> strides = {1, 1};
                            std::vector<int64_t> pads = {0, 0, 0, 0};
                            std::vector<int64_t> dilations = {1, 1};
                            
                            for (const auto& attr : node.attribute()) {
                                if (attr.name() == "strides") {
                                    strides.clear();
                                    for (int k = 0; k < attr.ints_size(); ++k) strides.push_back(attr.ints(k));
                                } else if (attr.name() == "pads") {
                                    pads.clear();
                                    for (int k = 0; k < attr.ints_size(); ++k) pads.push_back(attr.ints(k));
                                } else if (attr.name() == "dilations") {
                                    dilations.clear();
                                    for (int k = 0; k < attr.ints_size(); ++k) dilations.push_back(attr.ints(k));
                                }
                            }
                            
                            int out_channels = weight_tensor.dims(0);
                            
                            if (weight_dims == 3 && input_shape.size() >= 3) {
                                // Conv1D: [N, C, L] -> [N, out_c, out_l]
                                int N = input_shape[0] > 0 ? input_shape[0] : 1;
                                int L = input_shape[2];
                                int k_l = weight_tensor.dims(2);
                                int s = strides.size() > 0 ? strides[0] : 1;
                                int p = pads.size() > 0 ? pads[0] : 0;
                                int d = dilations.size() > 0 ? dilations[0] : 1;
                                int out_l = (L + 2 * p - d * (k_l - 1) - 1) / s + 1;
                                output_shape.append(N);
                                output_shape.append(out_channels);
                                output_shape.append(out_l);
                            } else if (weight_dims == 4 && input_shape.size() >= 4) {
                                // Conv2D: [N, C, H, W] -> [N, out_c, out_h, out_w]
                                int N = input_shape[0] > 0 ? input_shape[0] : 1;
                                int H = input_shape[2];
                                int W = input_shape[3];
                                int k_h = weight_tensor.dims(2);
                                int k_w = weight_tensor.dims(3);
                                int s_h = strides.size() > 0 ? strides[0] : 1;
                                int s_w = strides.size() > 1 ? strides[1] : s_h;
                                int p_h = pads.size() > 0 ? pads[0] : 0;
                                int p_w = pads.size() > 1 ? pads[1] : p_h;
                                int d_h = dilations.size() > 0 ? dilations[0] : 1;
                                int d_w = dilations.size() > 1 ? dilations[1] : d_h;
                                int out_h = (H + 2 * p_h - d_h * (k_h - 1) - 1) / s_h + 1;
                                int out_w = (W + 2 * p_w - d_w * (k_w - 1) - 1) / s_w + 1;
                                output_shape.append(N);
                                output_shape.append(out_channels);
                                output_shape.append(out_h);
                                output_shape.append(out_w);
                            }
                        }
                    }
                } else if (node.op_type() == "Flatten") {
                    // Flatten preserves batch, flattens the rest
                    if (node.input_size() > 0 && shape_metadata.exists(node.input(0))) {
                        Vector<int> input_shape = shape_metadata[node.input(0)];
                        if (!input_shape.empty()) {
                            int N = input_shape[0] > 0 ? input_shape[0] : 1;
                            int flat_size = 1;
                            for (unsigned k = 1; k < input_shape.size(); ++k) {
                                if (input_shape[k] > 0) flat_size *= input_shape[k];
                            }
                            output_shape.append(N);
                            output_shape.append(flat_size);
                        }
                    }
                }
                if (!output_shape.empty()) {
                    shape_metadata[outputName] = output_shape;
                    shapes_added++;
                }
            }
        }
        
        if (shapes_added == 0) break; // No progress, stop iterating
    }
    
    // Add intermediate tensor shapes from value_info
    // Only add if not already computed in the inference pass, and replace dynamic dims with 1
    for (const auto& value_info : _onnx_model.graph().value_info()) {
        if (value_info.type().tensor_type().has_shape()) {
            // Skip if already computed during shape inference
            if (shape_metadata.exists(value_info.name())) continue;
            
            const auto& shape = value_info.type().tensor_type().shape();
            Vector<int> dims;
            for (int i = 0; i < shape.dim_size(); ++i) {
                if (shape.dim(i).has_dim_value()) {
                    dims.append(static_cast<int>(shape.dim(i).dim_value()));
                } else {
                    // Dynamic dimension - use 1 as default
                    dims.append(1);
                }
            }
            if (!dims.empty()) {
                shape_metadata[value_info.name()] = dims;
            }
        }
    }

    // Build a complete list of all tensors in processing order
    // This mimics the Python approach of processing in graph order
    Vector<String> processingOrder;
    Set<String> seenNames;  // Track names to avoid duplicates
    
    // Add inputs first, but skip those that are also initializers (weights listed as inputs)
    // This handles ONNX models where weights appear in both graph.input and graph.initializer
    for (const auto& input : _onnx_model.graph().input()) {
        String inputName = input.name();
        // Only add as input if it's NOT an initializer (actual network input)
        if (!name_to_initializer.exists(inputName)) {
            processingOrder.append(inputName);
            seenNames.insert(inputName);
        }
    }
    
    // Add initializers (weights/biases)
    for (const auto& initializer : _onnx_model.graph().initializer()) {
        String initName = initializer.name();
        if (!seenNames.exists(initName)) {
            processingOrder.append(initName);
            seenNames.insert(initName);
        }
    }
    
    // Add node outputs in the order they appear in the graph
    for (const auto& node : _onnx_model.graph().node()) {
        for (int i = 0; i < node.output_size(); ++i) {
            String outputName = node.output(i);
            // Avoid duplicates using the Set for O(1) lookup
            if (!seenNames.exists(outputName)) {
                processingOrder.append(outputName);
                seenNames.insert(outputName);
            }
        }
    }

    // std::cerr << "[OnnxToTorchParser] Processing order:";
    // for (unsigned i = 0; i < processingOrder.size(); ++i) {
    //     std::cerr << " " << processingOrder[i];
    // }
    // std::cerr << std::endl << std::flush;

    // Build model components using unified nodes
    Vector<std::shared_ptr<NLR::BoundedTorchNode>> nodes;
    Vector<unsigned> inputIndices;
    unsigned outputIndex = 0;

    // Debug: Show expected output tensor name
    String expectedOutputName;
    if (_onnx_model.graph().output_size() > 0) {
        expectedOutputName = _onnx_model.graph().output(0).name();
//         std::cerr << "[OnnxToTorchParser] Expected output tensor name: " << expectedOutputName << std::endl << std::flush;
    } else {
//         std::cerr << "[OnnxToTorchParser] No outputs found in graph" << std::endl << std::flush;
    }

    // Create constants map for operations
    Map<String, torch::Tensor> constantsMap;

    // Map input names to their corresponding constants for operations
    // This needs to happen BEFORE node processing so constants are available
    for (const auto& node : _onnx_model.graph().node()) {
        for (int i = 0; i < node.input_size(); ++i) {
            String inputName = node.input(i);
            // If this input name corresponds to a constant, add it to the constants map
            if (name_to_initializer.exists(inputName) && !constantsMap.exists(inputName)) {
                torch::Tensor constant = ConstantProcessor::processInitializer(name_to_initializer[inputName]);
                constantsMap[inputName] = constant;
            }
            // Also treat outputs of ONNX Constant nodes as constants (common for BN params).
            if (name_to_node.exists(inputName) && !constantsMap.exists(inputName)) {
                const auto& producer = name_to_node[inputName];
                if (producer.op_type() == "Constant") {
                    torch::Tensor constant = ConstantProcessor::processConstantNode(producer);
                    constantsMap[inputName] = constant;
                }
            }
        }
    }

    // Map names to indices for lookup
    Map<String, unsigned> nameToIndex;
    for (unsigned i = 0; i < processingOrder.size(); ++i) {
        nameToIndex[processingOrder[i]] = i;
    }

    // Build dependency map: node index -> input node indices
    Map<unsigned, Vector<unsigned>> dependencies;

    // Process each tensor in processing order
    for (unsigned i = 0; i < processingOrder.size(); ++i) {
        const String& tensorName = processingOrder[i];
//         std::cerr << "[OnnxToTorchParser] Processing tensor " << i << ": " << tensorName << std::endl << std::flush;

        // Handle initializers (constants)
        if (name_to_initializer.exists(tensorName)) {
            torch::Tensor constant = ConstantProcessor::processInitializer(name_to_initializer[tensorName]);
            auto constantNode = std::make_shared<NLR::BoundedConstantNode>(constant, tensorName);
            constantNode->setNodeIndex(i);
            nodes.append(constantNode);
            continue;
        }

        // Handle Constant nodes
        if (name_to_node.exists(tensorName)) {
            const auto& node = name_to_node[tensorName];
            // std::cerr << "  Found node with op_type: " << node.op_type() << std::endl << std::flush;
            if (node.op_type() == "Constant") {
                // std::cerr << "  Processing as Constant node" << std::endl << std::flush;
                try {
                    torch::Tensor constant = ConstantProcessor::processConstantNode(node);
                    auto constantNode = std::make_shared<NLR::BoundedConstantNode>(constant, tensorName);
                    constantNode->setNodeIndex(i);
                    nodes.append(constantNode);
                } catch (const LirpaError& e) {
                    // Re-throw LirpaError exceptions as they are already properly formatted
                    throw;
                } catch (const std::exception& e) {
                    // Convert other exceptions to OnnxToTorch specific errors
                    onnxToTorchInvalidConstantNodeError(node, e.what());
                }
                continue;
            }
        }

        // Handle inputs
        if (name_to_input.exists(tensorName) && !name_to_initializer.exists(tensorName)) {
            inputIndices.append(i);

            // Get input size from the input info
            unsigned inputSize = 1; // Default, should be extracted from input info
            TensorShape inputShape = BoundedOperationConverter::extractShapeFromNode(onnx::NodeProto(), name_to_input, name_to_initializer, tensorName);
            if (!inputShape.empty()) {
                inputSize = BoundedOperationConverter::computeTensorSize(inputShape);
//                 std::cout << "[DEBUG] Input tensor " << tensorName << " shape: ";
                // for (unsigned dim : inputShape) {
                //     std::cout << dim << " ";
                // }
                // std::cout << ", computed size: " << inputSize << std::endl;
            }
            auto inputNode = std::make_shared<NLR::BoundedInputNode>(i, inputSize, tensorName);
            inputNode->setNodeIndex(i);
            nodes.append(inputNode);
            continue;
        }

        // Handle node outputs - create bounded nodes
        if (name_to_node.exists(tensorName)) {
            const auto& node = name_to_node[tensorName];

            // Build input dependencies for this node
            Vector<unsigned> deps;

            // Special handling for Sub nodes that will be converted to Linear
            if (node.op_type() == "Sub" && node.input_size() == 2) {
                // Check if second input is a constant
                String input2Name = node.input(1);
                bool isSecondInputConstant = (constantsMap.exists(input2Name) || name_to_initializer.exists(input2Name));

                if (isSecondInputConstant) {
                    // Sub will be converted to Linear, only needs first input as dependency
                    String input1Name = node.input(0);
                    if (nameToIndex.exists(input1Name)) {
                        unsigned inputIndex = nameToIndex[input1Name];
                        deps.append(inputIndex);
                    }
                } else {
                    // Sub with two variables - include both dependencies
                    for (int j = 0; j < node.input_size(); ++j) {
                        String inputName = node.input(j);
                        if (nameToIndex.exists(inputName)) {
                            unsigned inputIndex = nameToIndex[inputName];
                            deps.append(inputIndex);
                        }
                    }
                }
            } else if (node.op_type() == "Add" && node.input_size() == 2) {
                // Special handling for Add nodes
                String input1Name = node.input(0);
                String input2Name = node.input(1);
                bool isFirstInputConstant = (constantsMap.exists(input1Name) || name_to_initializer.exists(input1Name));
                bool isSecondInputConstant = (constantsMap.exists(input2Name) || name_to_initializer.exists(input2Name));

                if (isFirstInputConstant && !isSecondInputConstant) {
                    // Only second input (variable) as dependency
                    if (nameToIndex.exists(input2Name)) {
                        unsigned inputIndex = nameToIndex[input2Name];
                        deps.append(inputIndex);
                    }
                } else if (isSecondInputConstant && !isFirstInputConstant) {
                    // Only first input (variable) as dependency
                    if (nameToIndex.exists(input1Name)) {
                        unsigned inputIndex = nameToIndex[input1Name];
                        deps.append(inputIndex);
                    }
                } else if (!isFirstInputConstant && !isSecondInputConstant) {
                    // Both are variables - include both as dependencies
                    for (int j = 0; j < node.input_size(); ++j) {
                        String inputName = node.input(j);
                        if (nameToIndex.exists(inputName)) {
                            unsigned inputIndex = nameToIndex[inputName];
                            deps.append(inputIndex);
                        }
                    }
                }
                // If both are constants, no dependencies (handled in convertAdd as constant folding)
            } else if (node.op_type() == "BatchNormalization") {
                // BatchNormalization params are embedded into the bounded node; only X is a dependency.
                if (node.input_size() >= 1) {
                    String xName = node.input(0);
                    if (nameToIndex.exists(xName)) {
                        deps.append(nameToIndex[xName]);
                    }
                }
            } else {
                // Standard dependency handling for other nodes
                for (int j = 0; j < node.input_size(); ++j) {
                    String inputName = node.input(j);
                    if (nameToIndex.exists(inputName)) {
                        unsigned inputIndex = nameToIndex[inputName];
                        // Only include non-constant inputs as dependencies (constants are embedded in the node)
                        if (!name_to_initializer.exists(inputName)) {
                            deps.append(inputIndex);
                        }
                    }
                }
            }
            if (!deps.empty()) {
                dependencies[i] = deps;
                // std::cerr << "  Element dependencies: ";
                // for (unsigned d = 0; d < deps.size(); ++d) std::cerr << deps[d] << " ";
                // std::cerr << std::endl << std::flush;
            }

            // Convert node to bounded node with enhanced size setting
            std::shared_ptr<NLR::BoundedTorchNode> boundedNode;
            
            try {
                if (node.op_type() == "Identity") {
                    boundedNode = BoundedOperationConverter::convertIdentity(node, name_to_input, name_to_initializer);
                } else if (node.op_type() == "Gemm") {
                    boundedNode = BoundedOperationConverter::convertGemm(node, constantsMap, name_to_input, name_to_initializer);
                } else if (node.op_type() == "MatMul") {
                    boundedNode = BoundedOperationConverter::convertMatMul(node, constantsMap, name_to_input, name_to_initializer);
                } else if (node.op_type() == "Add") {
                    boundedNode = BoundedOperationConverter::convertAdd(node, constantsMap, name_to_input, name_to_initializer, nodes, nameToIndex);
                } else if (node.op_type() == "Relu" || node.op_type() == "relu") {
                    boundedNode = BoundedOperationConverter::convertRelu(node, name_to_input, name_to_initializer);
                } else if (node.op_type() == "Sigmoid" || node.op_type() == "sigmoid") {
                    boundedNode = BoundedOperationConverter::convertSigmoid(node, name_to_input, name_to_initializer);
                } else if (node.op_type() == "Reshape") {
                    boundedNode = BoundedOperationConverter::convertReshape(node, name_to_input, name_to_initializer);
                } else if (node.op_type() == "Flatten") {
                    boundedNode = BoundedOperationConverter::convertFlatten(node, name_to_input, name_to_initializer);
                } else if (node.op_type() == "Sub") {
                    boundedNode = BoundedOperationConverter::convertSub(node, constantsMap, name_to_input, name_to_initializer);
                } else if (node.op_type() == "Conv") {
                    boundedNode = BoundedOperationConverter::convertConv(node, constantsMap, name_to_input, name_to_initializer, shape_metadata);
                    if (!boundedNode) {
                        onnxToTorchBoundedModuleCreationError("Conv", "Conversion returned nullptr");
                    }
                } else if (node.op_type() == "ConvTranspose") {
                    boundedNode = BoundedOperationConverter::convertConvTranspose(node, constantsMap, name_to_input, name_to_initializer);
                    if (!boundedNode) {
                        onnxToTorchBoundedModuleCreationError("ConvTranspose", "Conversion returned nullptr");
                    }
                } else if (node.op_type() == "BatchNormalization") {
                    boundedNode = BoundedOperationConverter::convertBatchNormalization(node, constantsMap, name_to_input, name_to_initializer);
                } else if (node.op_type() == "Gather") {
                    // TODO: Implement proper BoundedGatherNode with correct bound propagation
                    // For now, treat as Identity (conservative but sound)
                    boundedNode = BoundedOperationConverter::convertIdentity(node, name_to_input, name_to_initializer);
                } else if (node.op_type() == "Cast") {
                    // Cast operation - bounds pass through unchanged (data type conversion doesn't affect bounds)
                    boundedNode = BoundedOperationConverter::convertIdentity(node, name_to_input, name_to_initializer);
                } else if (node.op_type() == "Dropout") {
                    // Dropout is disabled during inference - acts as identity pass-through
                    boundedNode = BoundedOperationConverter::convertIdentity(node, name_to_input, name_to_initializer);
                } else if (node.op_type() == "Concat") {
                    boundedNode = BoundedOperationConverter::convertConcat(node, name_to_input, name_to_initializer, shape_metadata);
                    if (!boundedNode) {
                        onnxToTorchBoundedModuleCreationError("Concat", "Conversion returned nullptr");
                    }
                } else if (node.op_type() == "Slice") {
                    boundedNode = BoundedOperationConverter::convertSlice(node, constantsMap, name_to_input, name_to_initializer, shape_metadata);
                    if (!boundedNode) {
                        onnxToTorchBoundedModuleCreationError("Slice", "Conversion returned nullptr");
                    }
                } else {
                    onnxToTorchUnsupportedOperationError(node);
                }
                
                // Infer and store output shape for this node BEFORE setting metadata
                if (node.output_size() > 0) {
                    String outputName = node.output(0);
                    Vector<int> output_shape;
                    
                    if (node.op_type() == "MatMul" || node.op_type() == "Gemm") {
                        // MatMul: [M, K] x [K, N] -> [M, N]
                        if (node.input_size() >= 2) {
                            Vector<int> input0_shape = shape_metadata.exists(node.input(0)) ? 
                                                       shape_metadata[node.input(0)] : Vector<int>();
                            String weight_name = node.input(1);
                            
                            // Get weight shape from initializer
                            if (name_to_initializer.exists(weight_name)) {
                                const auto& weight_tensor = name_to_initializer[weight_name];
                                if (weight_tensor.dims_size() == 2 && !input0_shape.empty()) {
                                    // Input: [batch, K], Weight: [K, N] -> Output: [batch, N]
                                    int batch = input0_shape[0];
                                    int N = weight_tensor.dims(1);
                                    output_shape.append(batch);
                                    output_shape.append(N);
                                    
                                }
                            }
                        }
                    } else if (node.op_type() == "Relu" || node.op_type() == "Sigmoid") {
                        // Activation functions preserve shape
                        if (node.input_size() > 0 && shape_metadata.exists(node.input(0))) {
                            output_shape = shape_metadata[node.input(0)];
                        }
                    }
                    
                    // Store computed output shape
                    if (!output_shape.empty()) {
                        shape_metadata[outputName] = output_shape;
                    }
                }
                
                // Set node metadata
                boundedNode->setNodeIndex(i);
                boundedNode->setNodeName(tensorName);

            } catch (const LirpaError& e) {
                throw;
            } catch (const std::exception& e) {
                onnxToTorchBoundedModuleCreationError(node.op_type(), e.what());
            }

            nodes.append(boundedNode);
            continue;
        }

        // Track output index
        if (_onnx_model.graph().output_size() > 0 && tensorName == _onnx_model.graph().output(0).name()) {
            outputIndex = i;
            // std::cerr << "  Output index set to " << i << " for tensor " << tensorName << std::endl << std::flush;
        }
    }

    // Infer sizes for dimension-preserving nodes and Conv nodes based on their input dependencies
//     std::cout << "[OnnxToTorch] Inferring sizes for nodes..." << std::endl;
    for (unsigned i = 0; i < nodes.size(); ++i) {
        auto node = nodes[i];
        if (!node) continue;

        // Process nodes that need size inference
            if ((node->getNodeType() == NLR::NodeType::RELU ||
             node->getNodeType() == NLR::NodeType::IDENTITY ||
             node->getNodeType() == NLR::NodeType::RESHAPE ||
             node->getNodeType() == NLR::NodeType::FLATTEN ||
             node->getNodeType() == NLR::NodeType::ADD ||
             node->getNodeType() == NLR::NodeType::SUB ||
                 node->getNodeType() == NLR::NodeType::CONV ||
                 node->getNodeType() == NLR::NodeType::CONVTRANSPOSE ||
                 node->getNodeType() == NLR::NodeType::BATCHNORM ||
                 node->getNodeType() == NLR::NodeType::LINEAR ||
                 node->getNodeType() == NLR::NodeType::CONCAT ||
                 node->getNodeType() == NLR::NodeType::SLICE) &&
            node->getInputSize() == 0) {

            // Get the input dependency
            if (dependencies.exists(i) && !dependencies[i].empty()) {
                unsigned inputIdx = dependencies[i][0];
                if (inputIdx < nodes.size() && nodes[inputIdx]) {
                    unsigned inferredSize = nodes[inputIdx]->getOutputSize();
                    if (inferredSize > 0) {
                        if (node->getNodeType() == NLR::NodeType::CONV) {
                            // For Conv nodes, we need special handling
                            auto convNode = std::dynamic_pointer_cast<NLR::BoundedConvNode>(node);
                            if (convNode) {
                                unsigned outputSize = convNode->inferOutputSize(inferredSize);
                                node->setInputSize(inferredSize);
                                node->setOutputSize(outputSize);
                                // std::cout << "[OnnxToTorch] Inferred Conv size for node " << i 
                                //           << ": input=" << inferredSize << " -> output=" << outputSize << std::endl;
                            } else {
                                node->setInputSize(inferredSize);
                                node->setOutputSize(2); // Fallback placeholder
                            }
                        } else if (node->getNodeType() == NLR::NodeType::CONVTRANSPOSE) {
                            // For ConvTranspose nodes, we need special handling
                            auto convtNode = std::dynamic_pointer_cast<NLR::BoundedConvTransposeNode>(node);
                            if (convtNode) {
                                unsigned outputSize = convtNode->inferOutputSize(inferredSize);
                                node->setInputSize(inferredSize);
                                node->setOutputSize(outputSize);
                                // std::cout << "[OnnxToTorch] Inferred ConvTranspose size for node " << i 
                                //           << ": input=" << inferredSize << " -> output=" << outputSize << std::endl;
                            } else {
                                node->setInputSize(inferredSize);
                                node->setOutputSize(2); // Fallback placeholder
                            }
                        } else if (node->getNodeType() == NLR::NodeType::LINEAR) {
                            // For Linear/MatMul nodes, just update the input size
                            // The output size should already be set during node creation based on weight dimensions
                            node->setInputSize(inferredSize);
                            // Output size was already set in convertMatMul/convertGemm, don't override it
                        } else if (node->getNodeType() == NLR::NodeType::CONCAT) {
                            // For Concat nodes, output size should already be set during conversion
                            // Just update input size
                            node->setInputSize(inferredSize);
                            // Output size was set in convertConcat, don't override it
                        } else if (node->getNodeType() == NLR::NodeType::SLICE) {
                            // For Slice nodes, output size should already be set during conversion
                            // Just update input size
                            node->setInputSize(inferredSize);
                            // Output size was set in convertSlice, don't override it
                        } else {
                            // For dimension-preserving nodes (RELU, IDENTITY, RESHAPE, etc.)
                            node->setInputSize(inferredSize);
                            if (node->getOutputSize() == 0) {
                                node->setOutputSize(inferredSize);
                            }
                        }
//                         std::cout << "[OnnxToTorch] Inferred size for node " << i
//                                   << " from input node " << inputIdx
//                                   << ": " << inferredSize << std::endl;
                    }
                }
            }
        }
    }

    // If output index wasn't set (ONNX model doesn't specify outputs), use the last non-constant node
    if (outputIndex == 0 && nodes.size() > 1) {
        // Find the last computational node (skip constants)
        for (int i = nodes.size() - 1; i >= 0; --i) {
            if (nodes[i] && nodes[i]->getNodeType() != NLR::NodeType::CONSTANT &&
                nodes[i]->getNodeType() != NLR::NodeType::INPUT) {
                outputIndex = i;
//                 std::cerr << "[OnnxToTorchParser] Output index not explicitly set, using last computational node: "
//                           << outputIndex << std::endl << std::flush;
                break;
            }
        }
    }

//     std::cerr << "[OnnxToTorchParser] Final outputIndex before creating TorchModel: " << outputIndex << std::endl << std::flush;
//     std::cerr << "[OnnxToTorchParser] Total nodes created: " << nodes.size() << std::endl << std::flush;

    return std::make_shared<NLR::TorchModel>(
        nodes,
        inputIndices,
        outputIndex,
        dependencies
    );
}

namespace AttributeUtils {

float getFloatAttribute(const onnx::NodeProto &node, const String &name, float defaultValue)
{
    for (const auto &attr : node.attribute()) {
        if (attr.name() == name.ascii()) {
            return attr.f();
        }
    }
    return defaultValue;
}

int getIntAttribute(const onnx::NodeProto &node, const String &name, int defaultValue)
{
    for (const auto &attr : node.attribute()) {
        if (attr.name() == name.ascii()) {
            return attr.i();
        }
    }
    return defaultValue;
}

Vector<int> getIntsAttribute(onnx::NodeProto &node, const String &name, const Vector<int> &defaultValue)
{
    for (const auto &attr : node.attribute()) {
        if (attr.name() == name.ascii()) {
            Vector<int> result;
            for (int i = 0; i < attr.ints_size(); i++) {
                result.append(attr.ints(i));
            }
            return result;
        }
    }
    return defaultValue;
}

String getStringAttribute(onnx::NodeProto &node, const String &name, const String &defaultValue)
{
    for (const auto &attr : node.attribute()) {
        if (attr.name() == name.ascii()) {
            return String(attr.s());
        }
    }
    return defaultValue;
}

Map<String, torch::IValue> extractAttributes(onnx::NodeProto &node)
{
    Map<String, torch::IValue> kwargs;
    
    for (const auto &attr : node.attribute()) {
        String attr_name = attr.name();
        
        if (attr.type() == onnx::AttributeProto::INT) {
            kwargs[attr_name] = torch::IValue(static_cast<int64_t>(attr.i()));
        } else if (attr.type() == onnx::AttributeProto::FLOAT) {
            kwargs[attr_name] = torch::IValue(static_cast<double>(attr.f()));
        } else if (attr.type() == onnx::AttributeProto::STRING) {
            kwargs[attr_name] = torch::IValue(attr.s());
        } else if (attr.type() == onnx::AttributeProto::INTS) {
            std::vector<int64_t> ints;
            for (int i = 0; i < attr.ints_size(); i++) {
                ints.push_back(attr.ints(i));
            }
            kwargs[attr_name] = torch::IValue(ints);
        } else if (attr.type() == onnx::AttributeProto::FLOATS) {
            std::vector<double> floats;
            for (int i = 0; i < attr.floats_size(); i++) {
                floats.push_back(attr.floats(i));
            }
            kwargs[attr_name] = torch::IValue(floats);
        } else if (attr.type() == onnx::AttributeProto::TENSOR) {
            // Convert tensor attribute to torch::Tensor
            // This is simplified - real implementation needs full tensor conversion
            kwargs[attr_name] = torch::IValue(torch::zeros({1})); // Placeholder
        }
        // Add more attribute types as needed
    }
    
    return kwargs;
}
}

namespace Operations {

torch::Tensor Constant::forward()
{
    return value;
}

torch::Tensor ReshapeImpl::forward(const torch::Tensor& input, const torch::Tensor& shape_tensor) {
    std::vector<int64_t> shape = GraphUtils::instantiateReshapeTemplate(input, shape_tensor);
    return input.reshape(shape);
}

} // namespace Operations

namespace GraphUtils {

Vector<String> computeTopologicalOrder(const Map<String, onnx::NodeProto>& name_to_node,
                                      const Map<String, onnx::ValueInfoProto>& name_to_input,
                                      const Map<String, onnx::TensorProto>& name_to_initializer)
{
    Vector<String> order;
    
    // First, add all inputs and initializers (sources)
    for (auto it = name_to_input.begin(); it != name_to_input.end(); ++it) {
        order.append(it->first);
    }
    
    for (auto it = name_to_initializer.begin(); it != name_to_initializer.end(); ++it) {
        order.append(it->first);
    }
    
    // Then add all node outputs in the order they appear in the graph
    // This mimics the Python approach where nodes are processed in order
    for (auto it = name_to_node.begin(); it != name_to_node.end(); ++it) {
        const onnx::NodeProto& node = it->second;
        for (int i = 0; i < node.output_size(); ++i) {
            String output = node.output(i);
            // Only add if not already in order (avoid duplicates)
            bool already_exists = false;
            for (unsigned j = 0; j < order.size(); ++j) {
                if (order[j] == output) {
                    already_exists = true;
                    break;
                }
            }
            if (!already_exists) {
                order.append(output);
            }
        }
    }

    return order;
}

Map<String, Set<String>> computeActivationDependencies(const onnx::GraphProto& graph) {
    /*
     * Compute activation dependencies, mapping each tensor to its dependents.
     * This mimics the Python implementation's compute_activation_dependencies function.
     */
    Map<String, Set<String>> needed_by;
    
    for (const auto& node : graph.node()) {
        String out_op_id = node.output(0);
        for (int i = 0; i < node.input_size(); ++i) {
            String in_op_id = node.input(i);
            needed_by[in_op_id].insert(out_op_id);
        }
        // TODO: Handle Loop nodes if needed
    }
    
    return needed_by;
}

std::vector<int64_t> instantiateReshapeTemplate(const torch::Tensor& input, const torch::Tensor& shape_tensor) {
    std::vector<int64_t> oldShape(input.sizes().begin(), input.sizes().end());
    std::vector<int64_t> newShapeTemplate;
    
    // Convert shape tensor to vector - handle different tensor shapes
    torch::Tensor flattened_shape = shape_tensor.flatten();
    for (int64_t i = 0; i < flattened_shape.numel(); ++i) {
        newShapeTemplate.push_back(flattened_shape[i].item<int64_t>());
    }
    
    std::vector<int64_t> newShape;
    int inferredIndex = -1;
    int64_t knownProduct = 1;
    
    for (size_t i = 0; i < newShapeTemplate.size(); ++i) {
        int64_t dim = newShapeTemplate[i];
        if (dim == 0) {
            // Copy from input shape
            dim = (i < oldShape.size()) ? oldShape[i] : 1;
        }
        if (dim == -1) {
            inferredIndex = i;
            newShape.push_back(1); // Placeholder
        } else {
            newShape.push_back(dim);
            knownProduct *= dim;
        }
    }
    
    if (inferredIndex != -1) {
        int64_t total = input.numel();
        int64_t inferred = total / knownProduct;
        newShape[inferredIndex] = inferred;
    }
    
    return newShape;
}

} // namespace GraphUtils

namespace ConstantProcessor {

torch::Tensor processInitializer(const onnx::TensorProto& tensor) {
    // std::cerr << "      [ConstantProcessor] Processing initializer: " << tensor.name() << std::endl;
    // std::cerr << "      [ConstantProcessor] Data type: " << tensor.data_type() << std::endl;
    // std::cerr << "      [ConstantProcessor] Tensor has " << tensor.dims_size() << " dimensions" << std::endl;

    // Determine tensor shape
    std::vector<int64_t> shape;
    for (int i = 0; i < tensor.dims_size(); ++i) {
        shape.push_back(tensor.dims(i));
//         std::cerr << "      [ConstantProcessor] Dimension " << i << ": " << tensor.dims(i) << std::endl << std::flush;
    }

    // Convert based on data type
    switch (tensor.data_type()) {
        case onnx::TensorProto_DataType_FLOAT: {
//             std::cerr << "      [ConstantProcessor] Processing FLOAT tensor" << std::endl << std::flush;

            // Check if data is in raw_data format (more common in ONNX files)
            // std::cerr << "      [ConstantProcessor] raw_data size: " << tensor.raw_data().size() << " bytes" << std::endl;
            if (!tensor.raw_data().empty()) {
                // std::cerr << "      [ConstantProcessor] Using raw_data with " << tensor.raw_data().size() << " bytes" << std::endl;
                const std::string& raw_data = tensor.raw_data();
                size_t num_elements = raw_data.size() / sizeof(float);
                std::vector<float> data(num_elements);
                std::memcpy(data.data(), raw_data.data(), raw_data.size());
                torch::Tensor result = torch::tensor(data, torch::kFloat32).reshape(shape);
//                 std::cerr << "      [ConstantProcessor] Created FLOAT tensor with shape: ";
                // for (auto dim : result.sizes()) {
                //     std::cerr << dim << " ";
                // }
                // std::cerr << std::endl << std::flush;
                return result;
            }

            // Fall back to float_data format
            // std::cerr << "      [ConstantProcessor] Using float_data with " << tensor.float_data_size() << " elements" << std::endl;
            if (tensor.float_data_size() == 0) {
                // Check if this is an empty tensor (0 elements total)
                int64_t total_elements = 1;
                for (int i = 0; i < tensor.dims_size(); ++i) {
                    total_elements *= tensor.dims(i);
                }
                
                // If the shape specifies 0 elements, create an empty tensor
                if (total_elements == 0) {
                    torch::Tensor result = torch::empty(shape, torch::kFloat32);
                    return result;
                }
                
                // Otherwise, this is an error
                std::string error_msg = "No data found in tensor (neither raw_data nor float_data). ";
                error_msg += "Shape: [";
                for (int i = 0; i < tensor.dims_size(); ++i) {
                    error_msg += std::to_string(tensor.dims(i));
                    if (i < tensor.dims_size() - 1) error_msg += ", ";
                }
                error_msg += "], Total elements: " + std::to_string(total_elements);
                onnxToTorchTensorConversionError(tensor.name(), error_msg);
            }
            std::vector<float> data;
            for (int i = 0; i < tensor.float_data_size(); ++i) {
                data.push_back(tensor.float_data(i));
            }
            torch::Tensor result = torch::tensor(data, torch::kFloat32).reshape(shape);
//             std::cerr << "      [ConstantProcessor] Created FLOAT tensor with shape: ";
//             for (auto dim : result.sizes()) {
//                 std::cerr << dim << " ";
//             }
//             std::cerr << std::endl << std::flush;
            return result;
        }
        case onnx::TensorProto_DataType_INT64: {
//             std::cerr << "      [ConstantProcessor] Processing INT64 tensor" << std::endl << std::flush;

            // Check if data is in raw_data format
            if (!tensor.raw_data().empty()) {
//                 std::cerr << "      [ConstantProcessor] Using raw_data with " << tensor.raw_data().size() << " bytes" << std::endl << std::flush;
                const std::string& raw_data = tensor.raw_data();
                size_t num_elements = raw_data.size() / sizeof(int64_t);
                std::vector<int64_t> data(num_elements);
                std::memcpy(data.data(), raw_data.data(), raw_data.size());
                return torch::tensor(data, torch::kInt64).reshape(shape);
            }

            // Fall back to int64_data format
            std::vector<int64_t> data;
            for (int i = 0; i < tensor.int64_data_size(); ++i) {
                data.push_back(tensor.int64_data(i));
            }
            return torch::tensor(data, torch::kInt64).reshape(shape);
        }
        case onnx::TensorProto_DataType_INT32: {
//             std::cerr << "      [ConstantProcessor] Processing INT32 tensor" << std::endl << std::flush;

            // Check if data is in raw_data format
            if (!tensor.raw_data().empty()) {
//                 std::cerr << "      [ConstantProcessor] Using raw_data with " << tensor.raw_data().size() << " bytes" << std::endl << std::flush;
                const std::string& raw_data = tensor.raw_data();
                size_t num_elements = raw_data.size() / sizeof(int32_t);
                std::vector<int32_t> data(num_elements);
                std::memcpy(data.data(), raw_data.data(), raw_data.size());
                return torch::tensor(data, torch::kInt32).reshape(shape);
            }

            // Fall back to int32_data format
            std::vector<int32_t> data;
            for (int i = 0; i < tensor.int32_data_size(); ++i) {
                data.push_back(tensor.int32_data(i));
            }
            return torch::tensor(data, torch::kInt32).reshape(shape);
        }
        case onnx::TensorProto_DataType_DOUBLE: {
//             std::cerr << "      [ConstantProcessor] Processing DOUBLE tensor" << std::endl << std::flush;

            // Check if data is in raw_data format
            if (!tensor.raw_data().empty()) {
//                 std::cerr << "      [ConstantProcessor] Using raw_data with " << tensor.raw_data().size() << " bytes" << std::endl << std::flush;
                const std::string& raw_data = tensor.raw_data();
                size_t num_elements = raw_data.size() / sizeof(double);
                std::vector<double> data(num_elements);
                std::memcpy(data.data(), raw_data.data(), raw_data.size());
                return torch::tensor(data, torch::kFloat64).reshape(shape);
            }

            // Fall back to double_data format
            std::vector<double> data;
            for (int i = 0; i < tensor.double_data_size(); ++i) {
                data.push_back(tensor.double_data(i));
            }
            return torch::tensor(data, torch::kFloat64).reshape(shape);
        }
        default:
//             std::cerr << "      [ConstantProcessor] Unsupported data type: " << tensor.data_type() << std::endl << std::flush;
            onnxToTorchUnsupportedDataTypeError(static_cast<onnx::TensorProto_DataType>(tensor.data_type()));
            return torch::Tensor(); // This line will never be reached, but satisfies compiler
    }
}

torch::Tensor processConstantNode(const onnx::NodeProto& node) {
//     std::cerr << "    [ConstantProcessor] Processing Constant node with " << node.attribute_size() << " attributes" << std::endl << std::flush;
    for (const auto& attr : node.attribute()) {
//         std::cerr << "    [ConstantProcessor] Checking attribute: " << attr.name() << " (type: " << attr.type() << ")" << std::endl << std::flush;
        if (attr.name() == "value") {
//             std::cerr << "    [ConstantProcessor] Found value attribute, checking if tensor data is present..." << std::endl << std::flush;
            
            // Check if the tensor data is actually present, regardless of type
            if (attr.has_t()) {
//                 std::cerr << "    [ConstantProcessor] Tensor data is present, processing..." << std::endl << std::flush;
                try {
                    torch::Tensor result = processInitializer(attr.t());
//                     std::cerr << "    [ConstantProcessor] Successfully processed constant tensor" << std::endl << std::flush;
                    return result;
                } catch (const std::exception& e) {
//                     std::cerr << "    [ConstantProcessor] Exception processing tensor: " << e.what() << std::endl << std::flush;
                    throw;
                }
            } else {
//                 std::cerr << "    [ConstantProcessor] No tensor data found in attribute" << std::endl << std::flush;
            }
        }
    }
//     std::cerr << "    [ConstantProcessor] No valid value attribute found" << std::endl << std::flush;
    onnxToTorchInvalidConstantNodeError(node, "No valid value attribute found");
    return torch::Tensor(); // This line will never be reached, but satisfies compiler
}

} // namespace ConstantProcessor

namespace BoundedOperationConverter {

    // Helper function to extract shape information from ONNX
    TensorShape extractShapeFromNode(const onnx::NodeProto& node, 
                                   const Map<String, onnx::ValueInfoProto>& name_to_input,
                                   const Map<String, onnx::TensorProto>& name_to_initializer,
                                   const String& tensorName) {
        (void)node; // Suppress unused parameter warning
        // Try to get shape from input info
        if (name_to_input.exists(tensorName)) {
            const auto& inputInfo = name_to_input[tensorName];
            if (inputInfo.type().tensor_type().has_shape()) {
                const auto& shape = inputInfo.type().tensor_type().shape();
                TensorShape result;
                for (int i = 0; i < shape.dim_size(); ++i) {
                    if (shape.dim(i).has_dim_value()) {
                        result.append(shape.dim(i).dim_value());
                    }
                }
                return result;
            }
        }
        
        // Try to get shape from initializer
        if (name_to_initializer.exists(tensorName)) {
            const auto& initializer = name_to_initializer[tensorName];
            TensorShape result;
            for (int i = 0; i < initializer.dims_size(); ++i) {
                result.append(initializer.dims(i));
            }
            return result;
        }
        
        return TensorShape();
    }
    
    // Helper function to compute tensor size from shape
    unsigned computeTensorSize(const TensorShape& shape) {
        if (shape.empty()) return 0;
        
        unsigned size = 1;
        for (unsigned dim : shape) {
            size *= dim;
        }
        return size;
    }

    // Compute broadcasted output shape following NumPy/ONNX broadcasting rules.
    // Shapes are right-aligned; for each dim, they must be equal or one of them is 1.
    // Missing leading dims are treated as 1.
    static TensorShape computeBroadcastShape(const String& operation, const TensorShape& a, const TensorShape& b) {
        if (a.empty()) return b;
        if (b.empty()) return a;

        const unsigned ra = a.size();
        const unsigned rb = b.size();
        const unsigned r = (ra >= rb) ? ra : rb;

        std::vector<unsigned> out_rev;
        out_rev.reserve(r);

        for (unsigned i = 0; i < r; ++i) {
            unsigned da = (i < ra) ? a[ra - 1 - i] : 1;
            unsigned db = (i < rb) ? b[rb - 1 - i] : 1;

            if (da != db && da != 1 && db != 1) {
                onnxToTorchInvalidBroadcastError(operation, a, b);
                return TensorShape(); // unreachable; keeps compiler happy
            }
            out_rev.push_back((da >= db) ? da : db);
        }

        TensorShape out;
        for (auto it = out_rev.rbegin(); it != out_rev.rend(); ++it) {
            out.append(*it);
        }
        return out;
    }

    std::shared_ptr<NLR::BoundedTorchNode> convertGemm(const onnx::NodeProto& node, 
                                                     const Map<String, torch::Tensor>& constants,
                                                     const Map<String, onnx::ValueInfoProto>& name_to_input,
                                                     const Map<String, onnx::TensorProto>& name_to_initializer) {
        (void)name_to_input; // Suppress unused parameter warning
        (void)name_to_initializer; // Suppress unused parameter warning
        // Extract weights and bias from constants
        if (node.input_size() < 2) {
            onnxToTorchUnexpectedNumberOfInputs(node, node.input_size(), 2, 3);
            return nullptr;
        }
        
        String weightName = node.input(1);
        String biasName = (node.input_size() > 2) ? node.input(2) : "";
        
        // Try to find weight tensor
        torch::Tensor weights;
        bool foundWeights = false;
        
        if (constants.exists(weightName)) {
            weights = constants[weightName];
            foundWeights = true;
        } else {
            Vector<String> possibleNames = {weightName, "weight", "W", "weights"};
            for (const auto& name : possibleNames) {
                if (constants.exists(name)) {
                    weights = constants[name];
                    foundWeights = true;
                    break;
                }
            }
        }
        
        if (!foundWeights) {
            onnxToTorchInvalidWeightBiasError("Gemm", "Weight tensor not found in constants");
            return nullptr;
        }
        
        // Handle bias tensor
        torch::Tensor bias;
        if (biasName.length() > 0 && constants.exists(biasName)) {
            bias = constants[biasName];
        } else {
            bias = torch::zeros({weights.size(0)});
        }
        
        // Extract attributes
        float alpha = AttributeUtils::getFloatAttribute(node, "alpha", 1.0f);
        float beta = AttributeUtils::getFloatAttribute(node, "beta", 1.0f);
        int transB = AttributeUtils::getIntAttribute(node, "transB", 0);
        
        // ONNX Gemm preprocessing
        if (transB == 0) {
            weights = weights.transpose(-2, -1);
        }

        if (beta != 1.0f) {
            bias = beta * bias;
        }

        // Ensure weights and bias have proper properties
        // IMPORTANT: Network weights should NOT have requires_grad=True for Alpha-CROWN!
        // Only alpha parameters (ReLU relaxation slopes) should be optimized, NOT network weights.
        // Setting requires_grad=True on weights causes autograd graphs to build up across iterations,
        // leading to "backward through the graph a second time" errors.
        // 1. contiguous() - ensures memory layout supports efficient operations
        // 2. to(torch::kFloat32) - standardizes dtype for consistency
        // 3. detach() - creates a new tensor not connected to previous computation graph
        // 4. requires_grad_(false) - network weights are CONSTANTS, not optimization variables
        weights = weights.contiguous().to(torch::kFloat32).detach().requires_grad_(false);
        bias = bias.contiguous().to(torch::kFloat32).detach().requires_grad_(false);

        // Create linear module
        auto linear_module = torch::nn::Linear(weights.size(1), weights.size(0));
        linear_module->weight = weights;
        linear_module->bias = bias;
        
        // Create bounded linear node - sizes will be set automatically in constructor
        auto boundedNode = std::make_shared<NLR::BoundedLinearNode>(linear_module, alpha);

//         std::cout << "[OnnxToTorch::convertGemm] Created Gemm node with sizes: input="
//                   << boundedNode->getInputSize() << ", output=" << boundedNode->getOutputSize() << std::endl;

        return boundedNode;
    }
    
    std::shared_ptr<NLR::BoundedTorchNode> convertMatMul(const onnx::NodeProto& node,
                                                        const Map<String, torch::Tensor>& constants,
                                                        const Map<String, onnx::ValueInfoProto>& name_to_input,
                                                        const Map<String, onnx::TensorProto>& name_to_initializer) {
        (void)name_to_input; // Suppress unused parameter warning
        (void)name_to_initializer; // Suppress unused parameter warning

        // MatMul requires exactly 2 inputs
        if (node.input_size() != 2) {
            onnxToTorchUnexpectedNumberOfInputs(node, node.input_size(), 2, 2);
            return nullptr;
        }

        String input1Name = node.input(0);
        String input2Name = node.input(1);

        // Check if second input is a constant (weight matrix)
        torch::Tensor weights;
        bool foundWeights = false;

        if (constants.exists(input2Name)) {
            weights = constants[input2Name];
            foundWeights = true;
        }

        if (!foundWeights) {
            onnxToTorchInvalidWeightBiasError("MatMul", "Weight tensor (second input) not found in constants");
            return nullptr;
        }

        // MatMul is Y = X @ W, where X is (batch, in_features) and W is (in_features, out_features)
        // PyTorch Linear expects (in_features, out_features) transposed, i.e., weight shape is (out_features, in_features)
        // So we need to transpose the ONNX weight matrix
        torch::Tensor transposed_weights = weights.transpose(-2, -1);

        // Ensure weights have proper properties for gradient-based optimization
        // This is critical for Alpha-CROWN to work correctly when parsing from ONNX files
        // Network weights are constants for Alpha-CROWN (only alpha parameters are optimized)
        transposed_weights = transposed_weights.contiguous().to(torch::kFloat32).detach().requires_grad_(false);

        // Create linear module without bias (MatMul doesn't include bias)
        int64_t in_features = weights.size(0);
        int64_t out_features = weights.size(1);

        auto linear_module = torch::nn::Linear(torch::nn::LinearOptions(in_features, out_features).bias(false));
        linear_module->weight = transposed_weights;

        // Create bounded linear node
        auto boundedNode = std::make_shared<NLR::BoundedLinearNode>(linear_module, 1.0f);

//         std::cout << "[OnnxToTorch::convertMatMul] Created MatMul node with sizes: input="
//                   << boundedNode->getInputSize() << ", output=" << boundedNode->getOutputSize() << std::endl;

        return boundedNode;
    }

    std::shared_ptr<NLR::BoundedTorchNode> convertAdd(const onnx::NodeProto& node,
                                                     const Map<String, torch::Tensor>& constants,
                                                     const Map<String, onnx::ValueInfoProto>& name_to_input,
                                                     const Map<String, onnx::TensorProto>& name_to_initializer,
                                                     const Vector<std::shared_ptr<NLR::BoundedTorchNode>>& existingNodes,
                                                     const Map<String, unsigned>& nameToIndex) {
        (void)existingNodes; // Suppress unused parameter warning
        (void)nameToIndex; // Suppress unused parameter warning

        // Add requires exactly 2 inputs
        if (node.input_size() != 2) {
            onnxToTorchUnexpectedNumberOfInputs(node, node.input_size(), 2, 2);
            return nullptr;
        }

        String input1Name = node.input(0);
        String input2Name = node.input(1);

//         std::cout << "[OnnxToTorch::convertAdd] Processing Add node with inputs: "
//                   << input1Name << " + " << input2Name << std::endl;

        // Try to infer sizes from inputs
        TensorShape shape1 = extractShapeFromNode(node, name_to_input, name_to_initializer, input1Name);
        TensorShape shape2 = extractShapeFromNode(node, name_to_input, name_to_initializer, input2Name);

        unsigned size1 = computeTensorSize(shape1);
        unsigned size2 = computeTensorSize(shape2);

        // Best-effort broadcasted output size.
        // If shape info is partial/missing, fall back to previous behavior.
        unsigned broadcastOutputSize = 0;
        if (!shape1.empty() || !shape2.empty()) {
            TensorShape outShape = computeBroadcastShape("Add", shape1, shape2);
            broadcastOutputSize = computeTensorSize(outShape);
        }
        if (broadcastOutputSize == 0) {
            broadcastOutputSize = (size1 >= size2) ? size1 : size2;
        }

        // Check which input is a constant
        torch::Tensor constantValue;
        bool isFirstInputConstant = (constants.exists(input1Name) || name_to_initializer.exists(input1Name));
        bool isSecondInputConstant = (constants.exists(input2Name) || name_to_initializer.exists(input2Name));

        // Create BoundedAddNode
        auto boundedNode = std::make_shared<NLR::BoundedAddNode>();

        if (isFirstInputConstant && !isSecondInputConstant) {
            // First input is constant, second is variable
            if (constants.exists(input1Name)) {
                constantValue = constants[input1Name];
            } else if (name_to_initializer.exists(input1Name)) {
                constantValue = ConstantProcessor::processInitializer(name_to_initializer[input1Name]);
            }
            boundedNode->setConstantValue(constantValue);

            // Size based on the variable input
            unsigned outputSize = broadcastOutputSize;
            boundedNode->setInputSize(size2);
            boundedNode->setOutputSize(outputSize);

//             std::cout << "[OnnxToTorch::convertAdd] Created Add node with constant first operand, size="
//                       << outputSize << std::endl;

        } else if (isSecondInputConstant && !isFirstInputConstant) {
            // Second input is constant, first is variable (most common case)
            if (constants.exists(input2Name)) {
                constantValue = constants[input2Name];
            } else if (name_to_initializer.exists(input2Name)) {
                constantValue = ConstantProcessor::processInitializer(name_to_initializer[input2Name]);
            }
            boundedNode->setConstantValue(constantValue);

            // Size based on the variable input
            unsigned outputSize = broadcastOutputSize;
            boundedNode->setInputSize(size1);
            boundedNode->setOutputSize(outputSize);

//             std::cout << "[OnnxToTorch::convertAdd] Created Add node with constant second operand, size="
//                       << outputSize << std::endl;

        } else if (!isFirstInputConstant && !isSecondInputConstant) {
            // Both inputs are variables
            unsigned outputSize = broadcastOutputSize;
            boundedNode->setInputSize(size1);  // First input size
            boundedNode->setOutputSize(outputSize);

//             std::cout << "[OnnxToTorch::convertAdd] Created Add node with two variable inputs: input1="
//                       << size1 << ", input2=" << size2 << ", output=" << outputSize << std::endl;

        } else {
            // Both inputs are constants - this should be folded at export time
//             std::cout << "[OnnxToTorch::convertAdd] Warning: Add with two constant inputs - should be constant-folded" << std::endl;

            // Compute the result and create a constant node instead
            torch::Tensor const1, const2;
            if (constants.exists(input1Name)) {
                const1 = constants[input1Name];
            } else if (name_to_initializer.exists(input1Name)) {
                const1 = ConstantProcessor::processInitializer(name_to_initializer[input1Name]);
            }
            if (constants.exists(input2Name)) {
                const2 = constants[input2Name];
            } else if (name_to_initializer.exists(input2Name)) {
                const2 = ConstantProcessor::processInitializer(name_to_initializer[input2Name]);
            }

            torch::Tensor result = const1 + const2;
            return std::make_shared<NLR::BoundedConstantNode>(result, "add_folded");
        }

        return boundedNode;
    }

    std::shared_ptr<NLR::BoundedTorchNode> convertRelu(const onnx::NodeProto& node,
                                                     const Map<String, onnx::ValueInfoProto>& name_to_input,
                                                     const Map<String, onnx::TensorProto>& name_to_initializer) {
        // Try to infer input size from the input tensor
        unsigned inputSize = 0;
        if (node.input_size() > 0) {
            String inputName = node.input(0);
//             std::cout << "[DEBUG] ReLU input tensor name: " << inputName << std::endl;
            TensorShape inputShape = extractShapeFromNode(node, name_to_input, name_to_initializer, inputName);
//             std::cout << "[DEBUG] ReLU input shape: ";
            // for (unsigned dim : inputShape) {
            //     std::cout << dim << " ";
            // }
            // std::cout << std::endl;
            inputSize = computeTensorSize(inputShape);
//             std::cout << "[DEBUG] ReLU computed input size: " << inputSize << std::endl;
        }
        
        auto relu_module = torch::nn::ReLU(torch::nn::ReLUOptions().inplace(false));
        auto boundedNode = std::make_shared<NLR::BoundedReLUNode>(relu_module);
        
        // Set sizes if we can infer them
        if (inputSize > 0) {
            boundedNode->setInputSize(inputSize);
            boundedNode->setOutputSize(inputSize); // ReLU preserves input size
//             std::cout << "[OnnxToTorch::convertRelu] Set sizes for ReLU node: input=output="
//                       << inputSize << std::endl;
        }

        return boundedNode;
    }
    
    std::shared_ptr<NLR::BoundedTorchNode> convertIdentity(const onnx::NodeProto& node,
                                                         const Map<String, onnx::ValueInfoProto>& name_to_input,
                                                         const Map<String, onnx::TensorProto>& name_to_initializer) {
        // Try to infer input size from the input tensor
        unsigned inputSize = 0;
        if (node.input_size() > 0) {
            String inputName = node.input(0);
            TensorShape inputShape = extractShapeFromNode(node, name_to_input, name_to_initializer, inputName);
            inputSize = computeTensorSize(inputShape);
        }
        
        auto identity_module = torch::nn::Identity();
        auto boundedNode = std::make_shared<NLR::BoundedIdentityNode>(identity_module);
        
        // Set sizes if we can infer them
        if (inputSize > 0) {
            boundedNode->setInputSize(inputSize);
            boundedNode->setOutputSize(inputSize); // Identity preserves input size
//             std::cout << "[OnnxToTorch::convertIdentity] Set sizes for Identity node: input=output="
//                       << inputSize << std::endl;
        }

        return boundedNode;
    }
    
    std::shared_ptr<NLR::BoundedTorchNode> convertReshape(const onnx::NodeProto& node,
                                                        const Map<String, onnx::ValueInfoProto>& name_to_input,
                                                        const Map<String, onnx::TensorProto>& name_to_initializer) {
        // Try to infer sizes from input and output shapes
        unsigned inputSize = 0;
        unsigned outputSize = 0;

        if (node.input_size() > 0) {
            String inputName = node.input(0);
            TensorShape inputShape = extractShapeFromNode(node, name_to_input, name_to_initializer, inputName);
            inputSize = computeTensorSize(inputShape);
        }

        if (node.output_size() > 0) {
            String outputName = node.output(0);
            TensorShape outputShape = extractShapeFromNode(node, name_to_input, name_to_initializer, outputName);
            outputSize = computeTensorSize(outputShape);
        }

        torch::Tensor default_shape = torch::tensor({-1});
        auto reshape_module = Operations::ReshapeWrapper(default_shape);
        auto boundedNode = std::make_shared<NLR::BoundedReshapeNode>(reshape_module);

        // Set sizes if we can infer them
        if (inputSize > 0) {
            boundedNode->setInputSize(inputSize);
        }
        if (outputSize > 0) {
            boundedNode->setOutputSize(outputSize);
        }

        if (inputSize > 0 || outputSize > 0) {
//             std::cout << "[OnnxToTorch::convertReshape] Set sizes for Reshape node: input="
//                       << inputSize << ", output=" << outputSize << std::endl;
        }

        return boundedNode;
    }

    std::shared_ptr<NLR::BoundedTorchNode> convertFlatten(const onnx::NodeProto& node,
                                                        const Map<String, onnx::ValueInfoProto>& name_to_input,
                                                        const Map<String, onnx::TensorProto>& name_to_initializer) {
        // Extract axis attribute (default = 1)
        int axis = AttributeUtils::getIntAttribute(node, "axis", 1);

        // Try to infer sizes from input shape
        unsigned inputSize = 0;
        unsigned outputSize = 0;
        TensorShape inputShape;

        if (node.input_size() > 0) {
            String inputName = node.input(0);
            inputShape = extractShapeFromNode(node, name_to_input, name_to_initializer, inputName);
            inputSize = computeTensorSize(inputShape);

            // For flatten, input and output sizes are the same (total number of elements preserved)
            outputSize = inputSize;

        }

        // Create flatten module with the specified axis
        auto flatten_module = Operations::FlattenWrapper(axis);
        auto boundedNode = std::make_shared<NLR::BoundedFlattenNode>(flatten_module);

        // Set sizes if we can infer them
        if (inputSize > 0) {
            boundedNode->setInputSize(inputSize);
            boundedNode->setOutputSize(outputSize);

            // Store the input shape for backward propagation
            // Convert TensorShape (Vector<unsigned>) to std::vector<int64_t>
            std::vector<int64_t> shapeVec;
            for (unsigned dim : inputShape) {
                shapeVec.push_back(static_cast<int64_t>(dim));
            }
            boundedNode->setInputShape(shapeVec);

        }

        return boundedNode;
    }

    std::shared_ptr<NLR::BoundedTorchNode> convertSigmoid(const onnx::NodeProto& node,
                                                     const Map<String, onnx::ValueInfoProto>& name_to_input,
                                                     const Map<String, onnx::TensorProto>& name_to_initializer) {
        // Try to infer input size from the input tensor
        unsigned inputSize = 0;
        if (node.input_size() > 0) {
            String inputName = node.input(0);
            TensorShape inputShape = extractShapeFromNode(node, name_to_input, name_to_initializer, inputName);
            inputSize = computeTensorSize(inputShape);
        }
        
        auto sigmoid_module = torch::nn::Sigmoid();
        auto boundedNode = std::make_shared<NLR::BoundedSigmoidNode>(sigmoid_module);
        
        // Set sizes if we can infer them
        if (inputSize > 0) {
            boundedNode->setInputSize(inputSize);
            boundedNode->setOutputSize(inputSize); // Sigmoid preserves input size
        }

        return boundedNode;
    }

    std::shared_ptr<NLR::BoundedTorchNode> convertSub(const onnx::NodeProto& node,
                                                     const Map<String, torch::Tensor>& constants,
                                                     const Map<String, onnx::ValueInfoProto>& name_to_input,
                                                     const Map<String, onnx::TensorProto>& name_to_initializer) {
        // Sub requires exactly 2 inputs
        if (node.input_size() != 2) {
            onnxToTorchUnexpectedNumberOfInputs(node, node.input_size(), 2, 2);
            return nullptr;
        }

        // Try to infer sizes from inputs
        String input1Name = node.input(0);
        String input2Name = node.input(1);

//         std::cout << "[OnnxToTorch::convertSub] Processing Sub node with inputs: "
//                   << input1Name << " - " << input2Name << std::endl;

        TensorShape shape1 = extractShapeFromNode(node, name_to_input, name_to_initializer, input1Name);
        TensorShape shape2 = extractShapeFromNode(node, name_to_input, name_to_initializer, input2Name);

        // Compute sizes
        unsigned size1 = computeTensorSize(shape1);
        unsigned size2 = computeTensorSize(shape2);

        // Check if either input is a constant
        bool isFirstInputConstant = (constants.exists(input1Name) || name_to_initializer.exists(input1Name));
        bool isSecondInputConstant = (constants.exists(input2Name) || name_to_initializer.exists(input2Name));

        // Create BoundedSubNode
        auto boundedNode = std::make_shared<NLR::BoundedSubNode>();

        if (isFirstInputConstant && !isSecondInputConstant) {
            // constant - x case
            torch::Tensor constantValue;
            if (constants.exists(input1Name)) {
                constantValue = constants[input1Name];
            } else if (name_to_initializer.exists(input1Name)) {
                constantValue = ConstantProcessor::processInitializer(name_to_initializer[input1Name]);
            }
            boundedNode->setConstantValue(constantValue, false); // false = constant is first operand

            // Size based on the variable input
            unsigned outputSize = size2;
            boundedNode->setInputSize(size2);
            boundedNode->setOutputSize(outputSize);

//             std::cout << "[OnnxToTorch::convertSub] Created Sub node with constant first operand, size="
//                       << outputSize << std::endl;

        } else if (isSecondInputConstant && !isFirstInputConstant) {
            // x - constant case (most common)
            torch::Tensor constantValue;
            if (constants.exists(input2Name)) {
                constantValue = constants[input2Name];
            } else if (name_to_initializer.exists(input2Name)) {
                constantValue = ConstantProcessor::processInitializer(name_to_initializer[input2Name]);
            }
            boundedNode->setConstantValue(constantValue, true); // true = constant is second operand

            unsigned outputSize = size1;
            boundedNode->setInputSize(size1);
            boundedNode->setOutputSize(outputSize);

//             std::cout << "[OnnxToTorch::convertSub] Created Sub node with constant second operand, size="
//                       << outputSize << std::endl;

        } else if (!isFirstInputConstant && !isSecondInputConstant) {
            // Both inputs are variables
            unsigned outputSize = (size1 >= size2) ? size1 : size2;
            boundedNode->setInputSize(size1);  // First input size
            boundedNode->setOutputSize(outputSize);

//             std::cout << "[OnnxToTorch::convertSub] Created Sub node with two variable inputs: input1="
//                       << size1 << ", input2=" << size2 << ", output=" << outputSize << std::endl;

        } else {
            // Both inputs are constants - this should be folded at export time
//             std::cout << "[OnnxToTorch::convertSub] Warning: Sub with two constant inputs - should be constant-folded" << std::endl;

            // Compute the result and create a constant node instead
            torch::Tensor const1, const2;
            if (constants.exists(input1Name)) {
                const1 = constants[input1Name];
            } else if (name_to_initializer.exists(input1Name)) {
                const1 = ConstantProcessor::processInitializer(name_to_initializer[input1Name]);
            }
            if (constants.exists(input2Name)) {
                const2 = constants[input2Name];
            } else if (name_to_initializer.exists(input2Name)) {
                const2 = ConstantProcessor::processInitializer(name_to_initializer[input2Name]);
            }

            torch::Tensor result = const1 - const2;
            return std::make_shared<NLR::BoundedConstantNode>(result, "sub_folded");
        }

        return boundedNode;
    }

    std::shared_ptr<NLR::BoundedTorchNode> convertBatchNormalization(const onnx::NodeProto& node,
                                                     const Map<String, torch::Tensor>& constants,
                                                     const Map<String, onnx::ValueInfoProto>& name_to_input,
                                                     const Map<String, onnx::TensorProto>& name_to_initializer) {
        (void)name_to_input;
        (void)name_to_initializer;

        // ONNX BatchNormalization inputs: X, scale, B, mean, var
        if (node.input_size() != 5) {
            onnxToTorchUnexpectedNumberOfInputs(node, node.input_size(), 5, 5);
            return nullptr;
        }

        float eps = AttributeUtils::getFloatAttribute(node, "epsilon", 1e-5f);

        String scaleName = node.input(1);
        String BName = node.input(2);
        String meanName = node.input(3);
        String varName = node.input(4);

        auto getTensorConst = [&](const String& name, const char* what) -> torch::Tensor {
            if (constants.exists(name)) {
                return constants[name];
            }
            if (name_to_initializer.exists(name)) {
                return ConstantProcessor::processInitializer(name_to_initializer[name]);
            }
            onnxToTorchInvalidWeightBiasError("BatchNormalization", Stringf("%s tensor not found in constants/initializers", what).ascii());
            return torch::Tensor();
        };

        torch::Tensor scale = getTensorConst(scaleName, "scale");
        torch::Tensor B = getTensorConst(BName, "B");
        torch::Tensor mean = getTensorConst(meanName, "mean");
        torch::Tensor var = getTensorConst(varName, "var");

        if (!scale.defined() || !B.defined() || !mean.defined() || !var.defined()) {
            return nullptr;
        }

        // Ensure float32 contiguous
        scale = scale.contiguous().to(torch::kFloat32);
        B = B.contiguous().to(torch::kFloat32);
        mean = mean.contiguous().to(torch::kFloat32);
        var = var.contiguous().to(torch::kFloat32);

        auto boundedNode = std::make_shared<NLR::BoundedBatchNormNode>(scale, B, mean, var, eps);

        // If shape is known, set input/output sizes (dimension preserving)
        if (node.input_size() > 0) {
            String inputName = node.input(0);
            TensorShape inputShape = extractShapeFromNode(node, name_to_input, name_to_initializer, inputName);
            unsigned inputSize = computeTensorSize(inputShape);
            if (inputSize > 0) {
                boundedNode->setInputSize(inputSize);
                boundedNode->setOutputSize(inputSize);
            }
        }

        return boundedNode;
    }

    std::shared_ptr<NLR::BoundedTorchNode> convertConstant(const torch::Tensor& value) {
        auto boundedNode = std::make_shared<NLR::BoundedConstantNode>(value, "");
        
        // Set sizes from constant value
        if (value.defined()) {
            unsigned size = value.numel();
            boundedNode->setInputSize(0); // Constants have no input
            boundedNode->setOutputSize(size);
//             std::cout << "[OnnxToTorch::convertConstant] Set sizes for Constant node: output="
//                       << size << std::endl;
        }

        return boundedNode;
    }

    std::shared_ptr<NLR::BoundedTorchNode> convertConv(const onnx::NodeProto& node,
                                                     const Map<String, torch::Tensor>& constants,
                                                     const Map<String, onnx::ValueInfoProto>& name_to_input,
                                                     const Map<String, onnx::TensorProto>& name_to_initializer,
                                                     const Map<String, Vector<int>>& shape_metadata) {
        (void)name_to_input; // Suppress unused parameter warning
        (void)name_to_initializer; // Suppress unused parameter warning
        
        // Extract input shape from metadata if available
        Vector<int> input_shape;
        if (node.input_size() > 0) {
            String inputName = node.input(0);
            if (shape_metadata.exists(inputName)) {
                input_shape = shape_metadata[inputName];
            }
        }

        // Conv node should have at least 2 inputs: X and W (weight)
        // Optional third input is B (bias)
        if (node.input_size() < 2) {
            onnxToTorchUnexpectedNumberOfInputs(node, node.input_size(), 2, 3);
            return nullptr;
        }

        String weightName = node.input(1);
        String biasName = (node.input_size() > 2) ? node.input(2) : "";

        // Get weight tensor from constants
        torch::Tensor weights;
        bool foundWeights = false;

        // std::cerr << "[convertConv] Looking for weight tensor: " << weightName.ascii() << std::endl;
        // std::cerr << "[convertConv] Constants map size: " << constants.size() << std::endl;

        // Debug: print first few available constants (commented out)
        // int count = 0;
        // for (auto it = constants.begin(); it != constants.end() && count < 5; ++it, ++count) {
        //     std::cerr << "[convertConv] Available constant: " << it->first.ascii() << std::endl;
        // }

        if (constants.exists(weightName)) {
            weights = constants[weightName];
            foundWeights = true;
            // std::cerr << "[convertConv] Found weight tensor with shape: " << weights.sizes() << std::endl;
        } else {
            // Try alternative names if exact name not found
            Vector<String> possibleNames = {weightName, "weight", "W", "weights"};
            for (const auto& name : possibleNames) {
                if (constants.exists(name)) {
                    weights = constants[name];
                    foundWeights = true;
                    // std::cerr << "[convertConv] Found weight tensor with alternative name: " << name.ascii()
                    //          << ", shape: " << weights.sizes() << std::endl;
                    break;
                }
            }
        }

        if (!foundWeights) {
            onnxToTorchInvalidWeightBiasError("Conv", "Weight tensor not found in constants");
            return nullptr;
        }

        // Handle bias tensor
        torch::Tensor bias;
        bool has_bias = false;
        if (biasName.length() > 0 && constants.exists(biasName)) {
            bias = constants[biasName];
            has_bias = true;
        }

        // Extract attributes from ONNX node
        // Default values based on ONNX Conv operator specification
        auto kernel_shape = AttributeUtils::getIntsAttribute(const_cast<onnx::NodeProto&>(node), "kernel_shape", {});
        int group = AttributeUtils::getIntAttribute(node, "group", 1);
        
        // Determine if this is Conv1D or Conv2D based on weight dimensions
        bool is_conv1d = (weights.dim() == 3);
        bool is_conv2d = (weights.dim() == 4);
        
        if (!is_conv1d && !is_conv2d) {
            onnxToTorchInvalidWeightBiasError("Conv",
                Stringf("Expected 3D (Conv1d) or 4D (Conv2d) weight tensor, got %ldD", weights.dim()).ascii());
            return nullptr;
        }
        
        // Set default values based on convolution dimensionality
        auto strides = is_conv1d ? 
            AttributeUtils::getIntsAttribute(const_cast<onnx::NodeProto&>(node), "strides", {1}) :
            AttributeUtils::getIntsAttribute(const_cast<onnx::NodeProto&>(node), "strides", {1, 1});
        auto pads = is_conv1d ?
            AttributeUtils::getIntsAttribute(const_cast<onnx::NodeProto&>(node), "pads", {0, 0}) :
            AttributeUtils::getIntsAttribute(const_cast<onnx::NodeProto&>(node), "pads", {0, 0, 0, 0});
        auto dilations = is_conv1d ?
            AttributeUtils::getIntsAttribute(const_cast<onnx::NodeProto&>(node), "dilations", {1}) :
            AttributeUtils::getIntsAttribute(const_cast<onnx::NodeProto&>(node), "dilations", {1, 1});

        // Extract dimensions from weight tensor
        int out_channels = weights.size(0);
        int in_channels_per_group = weights.size(1);
        int in_channels = in_channels_per_group * group;
        
        if (is_conv1d) {
            // Conv1D: weight shape [M, C/group, kL]
            int kernel_length = weights.size(2);
            
            // Validate kernel_shape if provided
            if (!kernel_shape.empty()) {
                if (kernel_shape.size() != 1 || kernel_shape[0] != kernel_length) {
                    onnxToTorchAttributeProcessingError(node, "kernel_shape",
                        "Mismatch with weight tensor dimensions for Conv1d");
                }
            }
        } else {
            // Conv2D: weight shape [M, C/group, kH, kW]
            int kernel_height = weights.size(2);
            int kernel_width = weights.size(3);
            
            // Validate kernel_shape if provided
            if (!kernel_shape.empty()) {
                if (kernel_shape.size() != 2 ||
                    kernel_shape[0] != kernel_height ||
                    kernel_shape[1] != kernel_width) {
                    onnxToTorchAttributeProcessingError(node, "kernel_shape",
                        "Mismatch with weight tensor dimensions for Conv2d");
                }
            }
        }

        // Convert ONNX padding format to PyTorch format
        std::vector<int64_t> padding;
        if (is_conv1d) {
            // Conv1D: ONNX pads format is [begin, end]
            if (pads.size() == 2) {
                // Check if padding is symmetric
                if (pads[0] == pads[1]) {
                    padding = {pads[0]};
                } else {
                    // Asymmetric padding - use begin padding
                    padding = {pads[0]};
                }
            } else if (pads.empty()) {
                padding = {0};
            } else {
                onnxToTorchAttributeProcessingError(node, "pads",
                    Stringf("Invalid padding size for Conv1d: %lu", pads.size()).ascii());
            }
        } else {
            // Conv2D: ONNX pads format is [top, left, bottom, right]
            if (pads.size() == 4) {
                // Check if padding is symmetric
                if (pads[0] == pads[2] && pads[1] == pads[3]) {
                    padding = {pads[0], pads[1]};
                } else {
                    // Asymmetric padding will need special handling
                    // For now, we'll use the top and left padding values
                    padding = {pads[0], pads[1]};
                }
            } else if (pads.size() == 2) {
                padding = {pads[0], pads[1]};
            } else if (pads.empty()) {
                padding = {0, 0};
            } else {
                onnxToTorchAttributeProcessingError(node, "pads",
                    Stringf("Invalid padding size for Conv2d: %lu", pads.size()).ascii());
            }
        }

        // Convert to int64_t vectors for PyTorch
        std::vector<int64_t> stride_vec(strides.begin(), strides.end());
        std::vector<int64_t> dilation_vec(dilations.begin(), dilations.end());

        // Network weights are constants for Alpha-CROWN (only alpha parameters are optimized)
        weights = weights.contiguous().to(torch::kFloat32).detach().requires_grad_(false);
        if (has_bias) {
            bias = bias.contiguous().to(torch::kFloat32).detach().requires_grad_(false);
        }

        // Create PyTorch Conv module and bounded node
        std::shared_ptr<NLR::BoundedConvNode> boundedNode;
        
        if (is_conv1d) {
            // Create Conv1d module
            int kernel_length = weights.size(2);
            torch::nn::Conv1dOptions conv_options(in_channels, out_channels, kernel_length);
            conv_options.stride(stride_vec);
            conv_options.padding(padding);
            conv_options.dilation(dilation_vec);
            conv_options.groups(group);
            conv_options.bias(has_bias);

            auto conv_module = torch::nn::Conv1d(conv_options);

            // Set weights and bias
            conv_module->weight = weights;
            if (has_bias) {
                conv_module->bias = bias;
            }

            // Create bounded convolution node (Conv1d doesn't use MATRIX mode)
            boundedNode = std::make_shared<NLR::BoundedConvNode>(conv_module,
                                                                   NLR::ConvMode::PATCHES);
        } else {
            // Create Conv2d module
            int kernel_height = weights.size(2);
            int kernel_width = weights.size(3);
            torch::nn::Conv2dOptions conv_options(in_channels, out_channels,
                                                  {kernel_height, kernel_width});
            conv_options.stride(stride_vec);
            conv_options.padding(padding);
            conv_options.dilation(dilation_vec);
            conv_options.groups(group);
            conv_options.bias(has_bias);

            auto conv_module = torch::nn::Conv2d(conv_options);

            // Set weights and bias
            conv_module->weight = weights;
            if (has_bias) {
                conv_module->bias = bias;
            }

            // Create bounded convolution node with default MATRIX mode
            boundedNode = std::make_shared<NLR::BoundedConvNode>(conv_module,
                                                                   NLR::ConvMode::MATRIX);
        }

        // Set sizes based on convolution parameters and ONNX shape metadata
        if (!input_shape.empty()) {
            // Use ONNX shape metadata to compute exact input/output sizes
            unsigned input_size = 1;
            for (unsigned i = 1; i < input_shape.size(); ++i) { // Skip batch dimension
                if (input_shape[i] > 0) {
                    input_size *= input_shape[i];
                }
            }
            
            // Compute output size based on convolution formula
            unsigned output_size = 0;
            if (is_conv1d && input_shape.size() >= 3) {
                // Conv1D: [N, C, L]
                int L = input_shape[2];
                if (L > 0) {
                    int kernel_length = weights.size(2);
                    int out_l = (L + 2 * padding[0] - dilation_vec[0] * (kernel_length - 1) - 1) / stride_vec[0] + 1;
                    output_size = out_channels * out_l;
                }
            } else if (!is_conv1d && input_shape.size() >= 4) {
                // Conv2D: [N, C, H, W]
                int H = input_shape[2];
                int W = input_shape[3];
                if (H > 0 && W > 0) {
                    int kernel_height = weights.size(2);
                    int kernel_width = weights.size(3);
                    int out_h = (H + 2 * padding[0] - dilation_vec[0] * (kernel_height - 1) - 1) / stride_vec[0] + 1;
                    int out_w = (W + 2 * padding[1] - dilation_vec[1] * (kernel_width - 1) - 1) / stride_vec[1] + 1;
                    output_size = out_channels * out_h * out_w;
                }
            }
            
            if (input_size > 0 && output_size > 0) {
                boundedNode->setInputSize(input_size);
                boundedNode->setOutputSize(output_size);
                
                // Also set the full input/output shape vectors for IBP
                std::vector<int> input_shape_vec;
                std::vector<int> output_shape_vec;
                
                for (int dim : input_shape) {
                    // Replace dynamic dimensions (marked as -1 or 0) with 1
                    if (dim > 0) {
                        input_shape_vec.push_back(dim);
                    } else {
                        input_shape_vec.push_back(1); // Use 1 for batch dimension
                    }
                }
                
                // Compute output shape
                if (is_conv1d && input_shape.size() >= 3) {
                    int N = (input_shape[0] > 0) ? input_shape[0] : 1;
                    int L = input_shape[2];
                    int kernel_length = weights.size(2);
                    int out_l = (L + 2 * padding[0] - dilation_vec[0] * (kernel_length - 1) - 1) / stride_vec[0] + 1;
                    output_shape_vec = {N, out_channels, out_l};
                } else if (!is_conv1d && input_shape.size() >= 4) {
                    int N = (input_shape[0] > 0) ? input_shape[0] : 1;
                    int H = input_shape[2];
                    int W = input_shape[3];
                    int kernel_height = weights.size(2);
                    int kernel_width = weights.size(3);
                    int out_h = (H + 2 * padding[0] - dilation_vec[0] * (kernel_height - 1) - 1) / stride_vec[0] + 1;
                    int out_w = (W + 2 * padding[1] - dilation_vec[1] * (kernel_width - 1) - 1) / stride_vec[1] + 1;
                    output_shape_vec = {N, out_channels, out_h, out_w};
                }
                
                if (!input_shape_vec.empty() && !output_shape_vec.empty()) {
                    boundedNode->setInputShape(input_shape_vec);
                    boundedNode->setOutputShape(output_shape_vec);
                }
            }
        }

        return boundedNode;
    }

    std::shared_ptr<NLR::BoundedTorchNode> convertConvTranspose(const onnx::NodeProto& node,
                                                     const Map<String, torch::Tensor>& constants,
                                                     const Map<String, onnx::ValueInfoProto>& name_to_input,
                                                     const Map<String, onnx::TensorProto>& name_to_initializer) {
        (void)name_to_input; // Suppress unused parameter warning
        (void)name_to_initializer; // Suppress unused parameter warning

        // ConvTranspose node should have at least 2 inputs: X and W (weight)
        // Optional third input is B (bias)
        if (node.input_size() < 2) {
            onnxToTorchUnexpectedNumberOfInputs(node, node.input_size(), 2, 3);
            return nullptr;
        }

        String weightName = node.input(1);
        String biasName = (node.input_size() > 2) ? node.input(2) : "";

        // Get weight tensor from constants
        torch::Tensor weights;
        bool foundWeights = false;

        if (constants.exists(weightName)) {
            weights = constants[weightName];
            foundWeights = true;
        } else {
            // Try alternative names if exact name not found
            Vector<String> possibleNames = {weightName, "weight", "W", "weights"};
            for (const auto& name : possibleNames) {
                if (constants.exists(name)) {
                    weights = constants[name];
                    foundWeights = true;
                    break;
                }
            }
        }

        if (!foundWeights) {
            onnxToTorchInvalidWeightBiasError("ConvTranspose", "Weight tensor not found in constants");
            return nullptr;
        }

        // Handle bias tensor
        torch::Tensor bias;
        bool has_bias = false;
        if (biasName.length() > 0 && constants.exists(biasName)) {
            bias = constants[biasName];
            has_bias = true;
        }

        // Extract attributes from ONNX node
        // Default values based on ONNX ConvTranspose operator specification
        auto kernel_shape = AttributeUtils::getIntsAttribute(const_cast<onnx::NodeProto&>(node), "kernel_shape", {});
        auto strides = AttributeUtils::getIntsAttribute(const_cast<onnx::NodeProto&>(node), "strides", {1, 1});
        auto pads = AttributeUtils::getIntsAttribute(const_cast<onnx::NodeProto&>(node), "pads", {0, 0, 0, 0});
        auto dilations = AttributeUtils::getIntsAttribute(const_cast<onnx::NodeProto&>(node), "dilations", {1, 1});
        auto output_padding = AttributeUtils::getIntsAttribute(const_cast<onnx::NodeProto&>(node), "output_padding", {0, 0});
        int group = AttributeUtils::getIntAttribute(node, "group", 1);

        // Validate weight tensor dimensions
        // ONNX ConvTranspose weight format is [C, M/group, kH, kW] for 2D
        // (Note: This is reversed from Conv which is [M, C/group, kH, kW])
        if (weights.dim() != 4) {
            onnxToTorchInvalidWeightBiasError("ConvTranspose",
                Stringf("Expected 4D weight tensor, got %ldD", weights.dim()).ascii());
            return nullptr;
        }

        // Extract dimensions from weight tensor
        // For ConvTranspose: weight is [in_channels, out_channels/group, kH, kW]
        int in_channels = weights.size(0);
        int out_channels_per_group = weights.size(1);
        int out_channels = out_channels_per_group * group;
        int kernel_height = weights.size(2);
        int kernel_width = weights.size(3);

        // Validate kernel_shape if provided
        if (!kernel_shape.empty()) {
            if (kernel_shape.size() != 2 ||
                kernel_shape[0] != kernel_height ||
                kernel_shape[1] != kernel_width) {
                onnxToTorchAttributeProcessingError(node, "kernel_shape",
                    "Mismatch with weight tensor dimensions");
            }
        }

        // Convert ONNX padding format to PyTorch format
        // ONNX: [top, left, bottom, right] or [begin_1, begin_2, end_1, end_2]
        // PyTorch ConvTranspose2d: single value or (height, width)
        std::vector<int64_t> padding;
        if (pads.size() == 4) {
            // Check if padding is symmetric
            if (pads[0] == pads[2] && pads[1] == pads[3]) {
                padding = {pads[0], pads[1]};
            } else {
                // Asymmetric padding will need special handling
                // For now, we'll use the top and left padding values
                padding = {pads[0], pads[1]};
            }
        } else if (pads.size() == 2) {
            padding = {pads[0], pads[1]};
        } else if (pads.empty()) {
            padding = {0, 0};
        } else {
            onnxToTorchAttributeProcessingError(node, "pads",
                Stringf("Invalid padding size: %lu", pads.size()).ascii());
        }

        // Convert to int64_t vectors for PyTorch
        std::vector<int64_t> stride_vec(strides.begin(), strides.end());
        std::vector<int64_t> dilation_vec(dilations.begin(), dilations.end());
        std::vector<int64_t> output_padding_vec(output_padding.begin(), output_padding.end());

        // Apply assertions from auto_LiRPA implementation
        if (output_padding_vec.size() != 2 || output_padding_vec[0] != 0 || output_padding_vec[1] != 0) {
            output_padding_vec = {0, 0};
        }
        if (dilation_vec[0] != 1 || dilation_vec[1] != 1) {
            onnxToTorchAttributeProcessingError(node, "dilations",
                "ConvTranspose only supports dilation [1, 1]");
            return nullptr;
        }
        if (stride_vec[0] != stride_vec[1]) {
            onnxToTorchAttributeProcessingError(node, "strides",
                "ConvTranspose requires symmetric stride (stride[0] == stride[1])");
            return nullptr;
        }
        if (group != 1) {
            onnxToTorchAttributeProcessingError(node, "group",
                "ConvTranspose only supports group = 1");
            return nullptr;
        }

        // Network weights are constants for Alpha-CROWN (only alpha parameters are optimized)
        weights = weights.contiguous().to(torch::kFloat32).detach().requires_grad_(false);
        if (has_bias) {
            bias = bias.contiguous().to(torch::kFloat32).detach().requires_grad_(false);
        }

        // Create PyTorch ConvTranspose2d module
        torch::nn::ConvTranspose2dOptions convt_options(in_channels, out_channels,
                                                        {kernel_height, kernel_width});
        convt_options.stride(stride_vec);
        convt_options.padding(padding);
        convt_options.dilation(dilation_vec);
        convt_options.output_padding(output_padding_vec);
        convt_options.groups(group);
        convt_options.bias(has_bias);

        auto convtranspose_module = torch::nn::ConvTranspose2d(convt_options);

        // Set weights and bias
        convtranspose_module->weight = weights;
        if (has_bias) {
            convtranspose_module->bias = bias;
        }

        // Create bounded convolution transpose node with default MATRIX mode
        auto boundedNode = std::make_shared<NLR::BoundedConvTransposeNode>(convtranspose_module,
                                                                           NLR::ConvMode::MATRIX);

        return boundedNode;
    }

    std::shared_ptr<NLR::BoundedTorchNode> convertConcat(const onnx::NodeProto& node,
                                                          const Map<String, onnx::ValueInfoProto>& name_to_input,
                                                          const Map<String, onnx::TensorProto>& name_to_initializer,
                                                          const Map<String, Vector<int>>& shape_metadata) {
        // Extract axis attribute (default = 1 in ONNX)
        int axis = AttributeUtils::getIntAttribute(node, "axis", 1);
        
        // Get number of inputs
        unsigned numInputs = node.input_size();
        if (numInputs < 2) {
            onnxToTorchUnexpectedNumberOfInputs(node, numInputs, 2, 100);
            return nullptr;
        }
        
        // Create Concat node
        auto boundedNode = std::make_shared<NLR::BoundedConcatNode>(axis, numInputs);
        
        // Collect sizes along the concat axis (matching auto_LiRPA's input_size)
        // In auto_LiRPA: self.input_size = [item.shape[self.axis] for item in x]
        std::vector<unsigned> input_sizes_along_axis;
        unsigned output_size_total = 0;
        
        for (unsigned i = 0; i < numInputs; ++i) {
            String inputName = node.input(i);
            
            // Try to get shape from shape_metadata first (for intermediate tensors)
            TensorShape inputShape;
            if (shape_metadata.exists(inputName)) {
                const Vector<int>& shape_vec = shape_metadata[inputName];
                for (unsigned j = 0; j < shape_vec.size(); ++j) {
                    inputShape.append(static_cast<unsigned>(shape_vec[j]));
                }
            } else {
                inputShape = extractShapeFromNode(node, name_to_input, name_to_initializer, inputName);
            }
            
            if (!inputShape.empty() && axis >= 0 && axis < (int)inputShape.size()) {
                // Get size along concat axis only
                unsigned size_along_axis = inputShape[axis];
                input_sizes_along_axis.push_back(size_along_axis);
                output_size_total += size_along_axis;
            } else {
            }
        }
        
        // Store the sizes along concat axis for use in backward pass
        if (!input_sizes_along_axis.empty()) {
            boundedNode->setInputSizes(input_sizes_along_axis);
            
            // For setInputSize/setOutputSize, we can use the first input's total size
            // and the computed total output size along concat axis
            String firstInputName = node.input(0);
            
            // Try to get shape from shape_metadata first (for intermediate tensors)
            TensorShape firstShape;
            if (shape_metadata.exists(firstInputName)) {
                const Vector<int>& shape_vec = shape_metadata[firstInputName];
                for (unsigned j = 0; j < shape_vec.size(); ++j) {
                    firstShape.append(static_cast<unsigned>(shape_vec[j]));
                }
            } else {
                firstShape = extractShapeFromNode(node, name_to_input, name_to_initializer, firstInputName);
            }
            
            if (!firstShape.empty()) {
                unsigned firstTensorSize = computeTensorSize(firstShape);
                unsigned concatAxisSize = firstShape[axis];
                unsigned computedOutputSize = firstTensorSize / concatAxisSize * output_size_total;
                boundedNode->setInputSize(firstTensorSize);
                boundedNode->setOutputSize(computedOutputSize);
            }
        }
        return boundedNode;
    }

    std::shared_ptr<NLR::BoundedTorchNode> convertSlice(const onnx::NodeProto& node,
                                                         const Map<String, torch::Tensor>& constantsMap,
                                                         const Map<String, onnx::ValueInfoProto>& name_to_input,
                                                         const Map<String, onnx::TensorProto>& name_to_initializer,
                                                         const Map<String, Vector<int>>& shape_metadata) {
        // ONNX Slice operation can have parameters in two formats:
        // 1. Attributes (ONNX < 13): starts, ends, axes, steps as node attributes
        // 2. Inputs (ONNX >= 13): data, starts, ends, axes, steps as separate input tensors

        int64_t start = 0;
        int64_t end = INT64_MAX;
        int64_t axes = 0;
        int64_t steps = 1;

        // Helper lambda to extract scalar value from tensor or constant
        auto extractScalarFromTensor = [&](const torch::Tensor& tensor) -> int64_t {
            if (tensor.numel() == 0) {
                return 0;
            }
            // Handle both 0D and 1D tensors with single element
            torch::Tensor flat = tensor.flatten();
            return flat[0].item<int64_t>();
        };

        auto extractScalarFromName = [&](const String& name) -> std::pair<bool, int64_t> {
            // Try constantsMap first
            if (constantsMap.exists(name)) {
                return {true, extractScalarFromTensor(constantsMap[name])};
            }
            // Try initializers
            if (name_to_initializer.exists(name)) {
                const auto& tensor_proto = name_to_initializer[name];
                torch::Tensor tensor = ConstantProcessor::processInitializer(tensor_proto);
                return {true, extractScalarFromTensor(tensor)};
            }
            return {false, 0};
        };

        // Check for old-style attributes first (ONNX < 13)
        bool has_attributes = false;
        for (const auto& attr : node.attribute()) {
            if (attr.name() == "starts") {
                has_attributes = true;
                if (attr.ints_size() > 0) {
                    start = attr.ints(0);
                }
            } else if (attr.name() == "ends") {
                if (attr.ints_size() > 0) {
                    end = attr.ints(0);
                }
            } else if (attr.name() == "axes") {
                if (attr.ints_size() > 0) {
                    axes = attr.ints(0);
                }
            } else if (attr.name() == "steps") {
                if (attr.ints_size() > 0) {
                    steps = attr.ints(0);
                }
            }
        }

        // If no attributes, extract from inputs (ONNX >= 13)
        // Inputs: data, starts, ends, axes (optional), steps (optional)
        if (!has_attributes && node.input_size() >= 3) {
            // Input 1: starts
            if (node.input_size() > 1) {
                String startsName = node.input(1);
                auto [found, val] = extractScalarFromName(startsName);
                if (found) {
                    start = val;
                }
            }

            // Input 2: ends
            if (node.input_size() > 2) {
                String endsName = node.input(2);
                auto [found, val] = extractScalarFromName(endsName);
                if (found) {
                    end = val;
                }
            }

            // Input 3: axes (optional)
            if (node.input_size() > 3 && !node.input(3).empty()) {
                String axesName = node.input(3);
                auto [found, val] = extractScalarFromName(axesName);
                if (found) {
                    axes = val;
                }
            }

            // Input 4: steps (optional)
            if (node.input_size() > 4 && !node.input(4).empty()) {
                String stepsName = node.input(4);
                auto [found, val] = extractScalarFromName(stepsName);
                if (found) {
                    steps = val;
                }
            }
        }

        // Validate parameters
        if (steps != 1 && steps != -1) {
            onnxToTorchAttributeProcessingError(node, "steps",
                Stringf("Slice only supports steps of 1 or -1, got %lld", (long long)steps));
            return nullptr;
        }

        if (axes == 0) {
            // Slicing along batch dimension - not supported
            // However, check if this might be a misinterpretation
            // (some models use axes=0 for the channel dimension without batch)
        }

        // Create the BoundedSliceNode
        auto boundedNode = std::make_shared<NLR::BoundedSliceNode>(start, end, axes, steps);

        // Compute input and output sizes
        String dataInputName = node.input(0);
        TensorShape inputShape;

        // Try to get shape from shape_metadata first
        if (shape_metadata.exists(dataInputName)) {
            const Vector<int>& shape_vec = shape_metadata[dataInputName];
            for (unsigned j = 0; j < shape_vec.size(); ++j) {
                inputShape.append(static_cast<unsigned>(shape_vec[j]));
            }
        } else {
            inputShape = extractShapeFromNode(node, name_to_input, name_to_initializer, dataInputName);
        }

        if (!inputShape.empty()) {
            unsigned inputSize = computeTensorSize(inputShape);
            boundedNode->setInputSize(inputSize);

            // Store input shape for backward pass
            std::vector<int64_t> input_shape_vec;
            for (unsigned j = 0; j < inputShape.size(); ++j) {
                input_shape_vec.push_back(static_cast<int64_t>(inputShape[j]));
            }
            boundedNode->setInputShape(input_shape_vec);

            // Compute output size by simulating the slice
            int64_t axis_idx = axes;
            if (axis_idx < 0) {
                axis_idx += static_cast<int64_t>(inputShape.size());
            }

            if (axis_idx >= 0 && axis_idx < static_cast<int64_t>(inputShape.size())) {
                int64_t axis_size = static_cast<int64_t>(inputShape[axis_idx]);

                // Fix up start and end
                int64_t fixed_start = start;
                int64_t fixed_end = end;

                if (fixed_start < 0) {
                    fixed_start += axis_size;
                }
                if (fixed_end < 0) {
                    if (fixed_end == -9223372036854775807LL) {  // ONNX -inf
                        fixed_end = 0;
                    } else {
                        fixed_end += axis_size;
                    }
                }
                if (steps == -1) {
                    std::swap(fixed_start, fixed_end);
                    fixed_end = fixed_end + 1;
                }
                fixed_end = std::min(fixed_end, axis_size);
                fixed_start = std::max(fixed_start, static_cast<int64_t>(0));

                int64_t slice_length = fixed_end - fixed_start;
                if (slice_length < 0) slice_length = 0;

                // Compute output size
                unsigned outputSize = inputSize / inputShape[axis_idx] * slice_length;
                boundedNode->setOutputSize(outputSize);
            } else {
                // Fallback: output size same as input
                boundedNode->setOutputSize(inputSize);
            }
        }

        return boundedNode;
    }

} // namespace BoundedOperationConverter