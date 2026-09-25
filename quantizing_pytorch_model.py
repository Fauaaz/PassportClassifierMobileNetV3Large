import torch
import onnx
import onnxscript
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime.quantization.shape_inference import quant_pre_process


model = torch.load('fine_tuned_model.pth', map_location=torch.device('cpu'), weights_only=False)
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy_input,
    "fine_tuned_model_fp32.onnx",
    export_params=True,
    opset_version=18,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
)


input_fp32 = "fine_tuned_model_fp32.onnx"
input_preprocessed = "fine_tuned_model_preprocessed.onnx"
output_int8 = "fine_tuned_model_int8.onnx"

# 1. Fix shape mismatches inside the ONNX model graph
quant_pre_process(
    input_model_path=input_fp32,
    output_model_path=input_preprocessed,
    skip_symbolic_shape_inference=False,
)

# 2. Quantize the pre-processed model
quantize_dynamic(
    model_input=input_preprocessed,
    model_output=output_int8,
    weight_type=QuantType.QUInt8,
)