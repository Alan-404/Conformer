import pycuda.driver as cuda
import pycuda.autoinit as autoinit
import tensorrt as trt
import os

import fire

from typing import Tuple
 
import logging
logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.DEBUG)
 
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
 
class TRTEngineBuilder:
    def __init__(self, onnx_file_path: str, min_shapes: Tuple[int], opt_shapes: Tuple[int], max_shapes: Tuple[int], fp16: bool = False, int8: bool=False, gpu_fallback: bool=False, verbose: bool=True):
        """
            TensorRT Inference Engine
        """
        trt.init_libnvinfer_plugins(None, "")
        self.TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
        self.onnx_file_path = onnx_file_path
        self.min_shapes = min_shapes
        self.opt_shapes = opt_shapes
        self.max_shapes = max_shapes
        self.fp16 = fp16
        self.int8 = int8
        self.gpu_fallback = gpu_fallback
        self.verbose = verbose
    def build_engine(self, save_path: str):

        builder = trt.Builder(self.TRT_LOGGER)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        config = builder.create_builder_config()
        parser = trt.OnnxParser(network, self.TRT_LOGGER)
 
        # Load the Onnx model and parse it in order to populate the TensorRT network.
        with open(self.onnx_file_path, "rb") as model:
            parser.parse(model.read(), path='onnx/')
 
        # Define the optimization profile
        profile = builder.create_optimization_profile()
        profile.set_shape("input", self.min_shapes, self.opt_shapes, self.max_shapes)
        # Set config
        if self.fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        if self.int8:
            config.set_flag(trt.BuilderFlag.INT8)
        if self.gpu_fallback:
            config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, (2 << 30) * 16)    
        config.add_optimization_profile(profile)
        if self.verbose:
            logging.info("Building engine...")
        serialized_network = builder.build_serialized_network(network, config)
        with trt.Runtime(self.TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(serialized_network)
        if save_path is not None:
            if self.verbose:
                logging.info("Saving engine...")   
            with open(save_path, "wb") as f:
                f.write(engine.serialize())
        return engine


def build_engine(
        onnx_path: str,
        engine_path: str,
        min_time: int,
        opt_time: int,
        max_time: int,
        fp16: bool = True,
        int8: bool = False,
        gpu_fallback: bool = False,
        verbose: bool = True
    ):
    assert os.path.exists(engine_path) and os.path.exists(onnx_path)
    builder = TRTEngineBuilder(
        onnx_file_path=onnx_path,
        min_shapes=(1, 80, min_time),
        opt_shapes=(1, 80, opt_time),
        max_shapes=(1, 80, max_time),
        fp16=fp16,
        int8=int8,
        gpu_fallback=gpu_fallback,
        verbose=verbose
    )

    builder.build_engine(save_path=engine_path)

if __name__ == '__main__':
    fire.Fire(build_engine)