import tensorrt as trt

modelpath = "./model.onnx"


logger = trt.Logger(trt.Logger.WARNING)

builder = trt.Builder(logger)

network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

parser = trt.OnnxParser(network, logger)

success = parser.parse_from_file(model_path)
for idx in range(parser.num_errors):
    print(parser.get_error(idx))

if not success:
    pass # Error handling code here

config = builder.create_builder_config()

config.max_workspace_size = 1 << 20 # 1 MiB

serialized_engine = builder.build_serialized_network(network, config)

with open("sample.engine", "wb") as f:
    f.write(serialized_engine)

runtime = trt.Runtime(logger)

engine = runtime.deserialize_cuda_engine(serialized_engine)

with open("sample.engine", "rb") as f:
    serialized_engine = f.read()

context = engine.create_execution_context()

input_idx = engine[input_name]
output_idx = engine[output_name]

buffers = [None] * 2 # Assuming 1 input and 1 output
buffers[input_idx] = input_ptr
buffers[output_idx] = output_ptr

context.execute_async_v2(buffers, stream_ptr)