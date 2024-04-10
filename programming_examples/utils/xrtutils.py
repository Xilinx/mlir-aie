# from npu.runtime import pyxrt as xrt
import npu.runtime as xrt
import numpy as np


class AIE_Application:
    
    def __init__(self, xclbin_path, insts_path, kernel_name="PP_FD_PRE"):
        self.device = None
        self.kernel = None
        self.buffers = [None] * 8
        self.device = xrt.device(0)   

        # Find kernel by name in the xclbin
        self.xclbin = xrt.xclbin(xclbin_path)
        kernels = self.xclbin.get_kernels()
        try:
            xkernel = [k for k in kernels if kernel_name == k.get_name()][0]
        except KeyError:
            raise AIE_Application_Error("No such kernel: " + kernel_name)
        self.device.register_xclbin(self.xclbin)
        self.context = xrt.hw_context(self.device, self.xclbin.get_uuid())
        self.kernel = xrt.kernel(self.context, xkernel.get_name())

        ## Set up instruction stream
        insts = read_insts(insts_path)
        self.n_insts = len(insts)
        self.insts_buffer = AIE_Buffer(self, 0, insts.dtype, insts.shape, 
                                       xrt.bo.cacheable)
        self.insts_buffer.write(insts)

    def register_buffer(self, group_id, *args, **kwargs):
        self.buffers[group_id] = AIE_Buffer(self, group_id, *args, **kwargs)
    
    def run(self):
        self.insts_buffer.sync_to_device()
        h = self.call()
        h.wait()
    
    def call(self):
        h = self.kernel(self.insts_buffer.bo, self.n_insts*4,
                        *[b.bo for b in self.buffers if b is not None])
        return h
    
    def __del__(self):
        del self.kernel
        del self.device


class AIE_Buffer:

    def __init__(self, application, group_id, dtype, shape, flags=xrt.bo.host_only):
        self.application = application
        self.dtype = dtype
        self.shape = shape
        self.len_bytes = np.prod(shape) * np.dtype(dtype).itemsize
        self.bo = xrt.bo(application.device, self.len_bytes, flags, 
                         application.kernel.group_id(group_id))

    def read(self):
        self.sync_from_device()
        return self.bo.read(self.len_bytes, 0).view(self.dtype).reshape(self.shape)
    
    def write(self, v,offset=0):
        self.bo.write(v.view(np.uint8), offset)
        self.sync_to_device()
    
    def sync_to_device(self):
        return self.bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    def sync_from_device(self):
        return self.bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    
    def __del__(self):
        del self.bo
        self.bo = None


class AIE_Application_Error(Exception):
    pass


insts_cache = {}
def read_insts(insts_path):
    global insts_cache
    if insts_path in insts_cache:
        # Speed up things if we re-configure the array a lot: Don't re-parse 
        # the insts.txt each time
        return insts_cache[insts_path]
    with open(insts_path, 'r') as f:
        insts_text = f.readlines()
        insts_text = [l for l in insts_text if l != '']
        insts_v = np.array([int(c, 16) for c in insts_text], 
                           dtype=np.uint32)
        insts_cache[insts_path] = insts_v
    return insts_v