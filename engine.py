import tensorrt as trt
import os
import torch
 
import logging
logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.DEBUG)
 
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class TRTInferenceEngine:
    def __init__(self, engine_file_path, device='cpu', verbose=True):
        """
            TensorRT Inference Engine for Realtime Object Detection
        """
        trt.init_libnvinfer_plugins(None, "")
        self.TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
        self.engine_file_path = engine_file_path
        self.device = device
        self.verbose = verbose
        self.engine = self._load_engine(self.engine_file_path)
        self.context = self.engine.create_execution_context()
        if self.verbose:
            self._verbose_binding_shape()
    def _verbose_binding_shape(self):
        for binding in self.engine:
            binding_idx = self.engine.get_binding_index(binding)
            binding_shape = self.engine.get_binding_shape(binding_idx)
            logging.info('{}, shape: {}'.format(binding, binding_shape))
 
        
    def _load_engine(self, engine_file_path):
        assert os.path.exists(engine_file_path)
        logging.info("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(self.TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    def _overwrite_batch_size(self, shape: tuple, batch_size):
        assert len(shape) == 4
        typecasted_shape = list(shape)
        typecasted_shape[0] = batch_size
        return tuple(typecasted_shape) 
    def __call__(self, batched_images): # [batch_size, channels, height, width]
        batch = batched_images.shape[0]
 
        input_buffers, output_buffers = [], []
        outputs = []
        for binding in self.engine:
            binding_idx = self.engine.get_binding_index(binding)
            if self.engine.binding_is_input(binding):
                input_binding_shape = self.engine.get_binding_shape(binding_idx)
                self.context.set_binding_shape(binding_idx, self._overwrite_batch_size(input_binding_shape, batch))
                # print(self.engine.get_binding_dtype(binding))
                # batched_images.type(torch.float32)
                batched_images.to(self.device)
                input_buffers.append(int(batched_images.data_ptr()))
            else:    
                # print(self.engine.get_binding_dtype(binding))
                output = torch.zeros(size=tuple(self.context.get_binding_shape(binding_idx)), dtype=torch.float32, device=self.device)
                outputs.append(output)
                output_buffers.append(int(output.data_ptr()))
 
        bindings = input_buffers + output_buffers
        # stream = cuda.Stream()
 
        self.context.execute_v2(bindings=bindings)
        # stream.synchronize()
        return outputs
    def __del__(self):
        del self.TRT_LOGGER
        del self.engine
        del self.context