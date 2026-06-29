# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses
from collections.abc import Callable
from contextlib import ExitStack
from typing import Any
from unittest.mock import patch

import torch
import ctypes
import os,subprocess
from cuda.bindings import driver as cuda
from cuda.bindings import nvrtc
#from cuda.bindings import runtime as cudart
import traceback
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
)
import vllm.envs as envs
from vllm.compilation.counter import compilation_counter
from vllm.compilation.monitor import validate_cudagraph_capturing_enabled
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed.device_communicators.pynccl_allocator import set_graph_pool_id
from vllm.forward_context import BatchDescriptor, get_forward_context
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.torch_utils import weak_ref_tensors
#import vllm.activations_extractor.cond_kernels as cond_kernels

logger = init_logger(__name__)
misha_switch =0
#do_node_enabled_toggling = 1
do_node_enabled_toggling = 0
add_identical_node = 0
add_cond_node = 0
graph_instantiated = 0

@dataclasses.dataclass
class CUDAGraphEntry:
    batch_descriptor: BatchDescriptor
    cudagraph: torch.cuda.CUDAGraph | None = None
    output: Any | None = None

    # for cudagraph debugging, track the input addresses
    # during capture, and check if they are the same during replay
    input_addresses: list[int] | None = None
    # Set once interventions have been applied to this entry's instantiated
    # graph exec. Guards against redundant work on every replay.
    interventions_applied: bool = False

@dataclasses.dataclass
class CUDAGraphOptions:
    debug_log_enable: bool = True
    gc_disable: bool = False
    weak_ref_output: bool = True


class CUDAGraphWrapper:
    """Wraps a runnable to add CUDA graph capturing and replaying ability. And
    provide attribute access to the underlying `runnable` via `__getattr__`.

    The workflow of this wrapper in the cudagraph dispatching is as follows:
    1. At initialization, a runtime mode is assigned to the wrapper (FULL or
    PIECEWISE).
    2. At runtime, the wrapper receives a runtime_mode and a
    batch_descriptor(key) from the forward context and blindly trust them
    for cudagraph dispatching.
    3. If runtime_mode is NONE or runtime_mode does not match the mode of the
    wrapper, just call the runnable directly.
    4. Otherwise, i.e., the runtime_mode matches the mode of the wrapper,
    the wrapper will perform cudagraph capture(if key does not exist, create
    a new entry and cache it) or replay (if key exists in the cache).

    Note: CUDAGraphWrapper does not store persistent buffers or copy any
    runtime inputs into that buffers for replay. We assume implementing them
    is done outside of the wrapper. That is because we do not make any
    assumption on the dynamic shape (batch size) of the runtime inputs, as a
    trade-off for staying orthogonal to compilation logic. Nevertheless,
    tracing and checking the input addresses to be consistent during replay is
    guaranteed when VLLM_LOGGING_LEVEL == "DEBUG".
    """

    def __init__(
        self,
        runnable: Callable,
        vllm_config: VllmConfig,
        runtime_mode: CUDAGraphMode,
        cudagraph_options: CUDAGraphOptions | None = None,
    ):
        self.runnable = runnable
        self.vllm_config = vllm_config
        self.runtime_mode = runtime_mode
        self.compilation_config = vllm_config.compilation_config

        self.first_run_finished = False
        self.is_debugging_mode = envs.VLLM_LOGGING_LEVEL == "DEBUG"

        # assert runtime_mode is not NONE(no cudagraph), otherwise, we don't
        # need to initialize a CUDAGraphWrapper.
        assert self.runtime_mode != CUDAGraphMode.NONE
        # TODO: in the future, if we want to use multiple
        # streams, it might not be safe to share a global pool.
        # only investigate this when we use multiple streams
        self.graph_pool = current_platform.get_global_graph_pool()

        if cudagraph_options is None:
            cudagraph_options = CUDAGraphOptions()
        self.cudagraph_options = cudagraph_options
        # the entries for different batch descriptors that we need to capture
        # cudagraphs for.
        self.concrete_cudagraph_entries: dict[BatchDescriptor, CUDAGraphEntry] = {}
        self.toggled_cudagraph : dict[BatchDescriptor, bool] = {}

    def __getattr__(self, key: str):
        # allow accessing the attributes of the runnable.
        if hasattr(self.runnable, key):
            return getattr(self.runnable, key)
        raise AttributeError(
            f"Attribute {key} not exists in the runnable of "
            f"cudagraph wrapper: {self.runnable}"
        )

    def cudagraph_add_conditional_node(self,kernel_name:str,input_conditional_buffer:torch.Tensor):
        forward_context = get_forward_context()
        batch_descriptor = forward_context.batch_descriptor
        cudagraph_runtime_mode = forward_context.cudagraph_runtime_mode

        print("Misha Debug entered cudagraph_add_conditional_node!!!")
        if (
            cudagraph_runtime_mode == CUDAGraphMode.NONE
            or cudagraph_runtime_mode != self.runtime_mode
        ):
            # CUDAGraphMode.NONE could mean the profile run, a warmup run, or
            # running without cudagraphs.
            # We do not trigger capture/replay if the runtime mode is not
            # matches. This enables properly dispatching to the correct
            # CUDAGraphWrapper when nesting multiple instances with different
            # runtime modes.
            return

        if batch_descriptor not in self.concrete_cudagraph_entries:
            # create a new entry for this batch descriptor
            self.concrete_cudagraph_entries[batch_descriptor] = CUDAGraphEntry(
                batch_descriptor=batch_descriptor
            )

        entry = self.concrete_cudagraph_entries[batch_descriptor]

        if entry.cudagraph is None:
            print("Misha Debug: cudagraph entry doesn't exist yet! - exiting")
        else:
            raw_graph = entry.cudagraph.raw_cuda_graph()
            cu_graph_template = cuda.CUgraph(raw_graph)

            err, device = cuda.cuDeviceGet(0)
            err, context = cuda.cuCtxGetCurrent()

            err,nodes, num_nodes = cuda.cuGraphGetNodes(cu_graph_template )# first call: get count
            err, nodes,num_nodes_2 = cuda.cuGraphGetNodes(cu_graph_template, num_nodes)
            for node in nodes:
                err,params = cuda.cuGraphKernelNodeGetParams(node)
                if params != None:
                    try:
                        err, name = cuda.cuFuncGetName(params.func)
                        name_str = name.decode('utf-8') if isinstance(name, bytes) else str(name)
                        if kernel_name in name_str:
                            #err, cond_handle = cuda.cuGraphConditionalHandleCreate(cu_graph_template, context, 0, 0)
                            kernelArgs = params.kernelParams

                            err, cond_handle = cuda.cuGraphConditionalHandleCreate(cu_graph_template ,context, 0, 0)
                            """
                            # --- 2. Configure the Conditional Node Parameters ---
                            cond_node_params = cuda.CUgraphNodeParams()
                            cond_node_params.type = cuda.CUgraphNodeType.CU_GRAPH_NODE_TYPE_CONDITIONAL
                            cond_node_params.conditional.handle = cond_handle
                            cond_node_params.conditional.type = cuda.CUgraphConditionalNodeType.CU_GRAPH_COND_TYPE_IF
                            cond_node_params.conditional.size = 1  # 1 because it's a simple IF statement (one body graph)
                            cond_node_params.conditional.ctx = context
                            # We must provide an array in memory for the driver to return the newly created subgraphs into.
                            value_type = ctypes.c_int32

                            # 5. Change the value by accessing index 0
                            #new_ptr.value = cond_handle
                            arg1_storage = ctypes.c_uint64.from_address(params.kernelParams + 8).value
                            print(f"arg1 storage ptr: {arg1_storage:#x}")
                            ctypes.c_uint64.from_address(arg1_storage).value = int(cond_handle)
                            
                            try:
                                # If it's a ctypes wrapper:
                                ptr_value = cond_handle.value
                            except AttributeError:
                                # If it's cuda-python or similar that casts to int:
                                ptr_value = int(cond_handle)
                            
                            print(f"Misha Debug- CUDA Handle Pointer Value: {ptr_value:#x}")
                            # 2. The address of the Python object in Host (CPU) memory
                            # (This is just where Python stores the wrapper, the GPU cannot read this)
                            print(f"Misha Debug- Python Object Host Address: {id(cond_handle):#x}")

                            handle_val = ctypes.cast(cond_handle, ctypes.c_void_p).value
                            print(f"Misha debug: Handle Value (Hex): {handle_val:#x}")
                            """ 
                            # Misha - New code addition, adding compiled cuda code on the fly
                            # for the conditional node cudaGraphSetConditional API call
                            # Pack args for: set_cond_kernel(cudaGraphConditionalHandle handle, const int* n_valid)
                            # Both are 64-bit values when passed to the kernel
                            err, module = cuda.cuModuleLoad(b"set_cond.cubin")
                            # Add this check right here:
                            if err != 0: # or cu_wrapper.CUresult.CUDA_SUCCESS depending on your bindings
                                print(f"Misha Debug - line 207- FATAL: cuModuleLoad failed with error code: {err}")
                            err, func = cuda.cuModuleGetFunction(module, b"set_cond_kernel")
                            if err != 0:
                                print(f"Misha Debug - line 210- FATAL: cuModuleGetFunction failed: {err}")

                            handle_arg  = ctypes.c_uint64(int(cond_handle))
                            input_buffer_arg = ctypes.c_uint64(input_conditional_buffer.data_ptr())  # SteeringVectorAdder.n_valid_buf

                            # kernelParams is void** — array of pointers to each argument value
                            kern_args = (ctypes.c_void_p * 2)(
                                ctypes.addressof(handle_arg),
                                ctypes.addressof(input_buffer_arg),
                            )

                            set_cond_params = cuda.CUDA_KERNEL_NODE_PARAMS()
                            set_cond_params.func           = func
                            set_cond_params.gridDimX       = 1
                            set_cond_params.gridDimY       = 1
                            set_cond_params.gridDimZ       = 1
                            set_cond_params.blockDimX      = 1
                            set_cond_params.blockDimY      = 1
                            set_cond_params.blockDimZ      = 1
                            set_cond_params.sharedMemBytes = 0
                            set_cond_params.kernelParams   = ctypes.addressof(kern_args)

                            # Add NVRTC node after the found node (which already ran EOL detection)
                            err, set_cond_node = cuda.cuGraphAddKernelNode(cu_graph_template, [node], 1, set_cond_params)
                            print(f"Misha Debug - line 245 - after first cuGraphAddKernelNode err:{err}")
                            # Add conditional node depending on the NVRTC node
                            cond_node_params = cuda.CUgraphNodeParams()
                            cond_node_params.type                  = cuda.CUgraphNodeType.CU_GRAPH_NODE_TYPE_CONDITIONAL
                            cond_node_params.conditional.handle    = cond_handle
                            cond_node_params.conditional.type      = cuda.CUgraphConditionalNodeType.CU_GRAPH_COND_TYPE_IF
                            cond_node_params.conditional.size      = 1
                            cond_node_params.conditional.ctx       = context
                            err, cond_node = cuda.cuGraphAddNode(cu_graph_template, [set_cond_node], None, 1, cond_node_params)
                            print(f"Misha Debug - line 254 - after second cuGraphAddKernelNode err:{err}")
                            err,params = cuda.cuGraphKernelNodeGetParams(node )
                            err, added_node = cuda.cuGraphAddKernelNode(cu_graph_template, [cond_node], 1,params) 
                            # Misha - End of new code

                            #err = cuda.cuGraphKernelNodeSetParams(node, params)
                                    
                            # --- 3. Add the Conditional Node to the Main Graph ---
                            # We insert it after Node A by setting [node_a] as its dependency
                            #err, cond_node = cuda.cuGraphAddNode(cu_graph_template, [node], None,1, cond_node_params)
                    except AttributeError:
                        print("    Kernel Name: [Requires CUDA 12.0+ or cuFuncGetName not found]")


    def cudagraph_intervention_func(self,kernel_name:str,action:bool,layer:int)->None:
        # Should only be called when forward_context and batch_descriptor are defined 
        # i.e. Just before running the model.
        #print("Misha Debug - entered cudagraph_intervention_func")
        global misha_switch
        global graph_instantiated
        layer_counter = 0
        forward_context = get_forward_context()
        batch_descriptor = forward_context.batch_descriptor
        cudagraph_runtime_mode = forward_context.cudagraph_runtime_mode

        if (
            cudagraph_runtime_mode == CUDAGraphMode.NONE
            or cudagraph_runtime_mode != self.runtime_mode
        ):
            # CUDAGraphMode.NONE could mean the profile run, a warmup run, or
            # running without cudagraphs.
            # We do not trigger capture/replay if the runtime mode is not
            # matches. This enables properly dispatching to the correct
            # CUDAGraphWrapper when nesting multiple instances with different
            # runtime modes.
            return

        if batch_descriptor not in self.concrete_cudagraph_entries:
            # create a new entry for this batch descriptor
            self.concrete_cudagraph_entries[batch_descriptor] = CUDAGraphEntry(
                batch_descriptor=batch_descriptor
            )
        

        entry = self.concrete_cudagraph_entries[batch_descriptor]
        #if self.toggled_cudagraph[batch_descriptor] == False:
        #    return

        if entry.cudagraph is None:
            print("Misha Debug: cudagraph entry doesn't exist yet! - exiting")
        else:
            #print("Misha Debug - cuda_graph.py - line 117 - misha_func called successsfully")
            # Here we can put logic of node toggling
            if graph_instantiated == 0:
                return
            raw_graph = entry.cudagraph.raw_cuda_graph()
            cu_graph_template = cuda.CUgraph(raw_graph)
            try:
                raw_graph_exec = entry.cudagraph.raw_cuda_graph_exec()            
            except:
                return
            cu_graph = cuda.CUgraphExec(raw_graph_exec)
            err,nodes, num_nodes = cuda.cuGraphGetNodes(cu_graph_template )# first call: get count
            err, nodes,num_nodes_2 = cuda.cuGraphGetNodes(cu_graph_template, num_nodes)
            #if misha_switch == 0: # For debug only - to not print too much
            #    return
            #else:
            #    misha_switch = 0
            for node in nodes:
                err,params = cuda.cuGraphKernelNodeGetParams(node)
                if params != None:
                    try:
                        err, name = cuda.cuFuncGetName(params.func)
                        name_str = name.decode('utf-8') if isinstance(name, bytes) else str(name)
                        #if get_tensor_model_parallel_rank() == 0: 
                        #    print(f"    Kernel Name: {name_str}")
                        if kernel_name in name_str and (layer_counter == layer or layer == -1):
                            #print(f"Misha Debug - found my custom node")
                            #print(f"Misha Debug - kernel_name:{name_str}")
                            kernelArgs = params.kernelParams
                            #i=0
                            #while i < 100:
                                #ptr_at_i = ctypes.c_void_p.from_address(kernelArgs + i * 8).value
                                #if ptr_at_i is None or ptr_at_i ==0:
                                #    break
                                #print(f"arg_num:{i} address:{ptr_at_i:#x}")
                                #i += 1
                                #print(f"Misha Debug - cuda_graphs - argument insptection:arg_num:{i} address:{kernelArgs[i]}")
                                #i+=1
                            err = cuda.cuGraphNodeSetEnabled(cu_graph,node,action)
                            if err !=0 :
                                print(f"Misha Debug- Node disable failed with:err={err}")
                        if "att" in name_str:
                            layer_counter +=1
                    except AttributeError:
                        print("    Kernel Name: [Requires CUDA 12.0+ or cuFuncGetName not found]")
            #if get_tensor_model_parallel_rank() == 0: 
            #    print(f"Misha Debug - calculated:{layer_counter} layers")
            
    def unwrap(self) -> Callable:
        # in case we need to access the original runnable.
        return self.runnable

    def __call__(self, *args, **kwargs):
        global do_node_enabled_toggling
        global add_identical_node
        global add_cond_node
        global misha_switch
        global graph_instantiated
        forward_context = get_forward_context()
        batch_descriptor = forward_context.batch_descriptor
        cudagraph_runtime_mode = forward_context.cudagraph_runtime_mode

        if (
            cudagraph_runtime_mode == CUDAGraphMode.NONE
            or cudagraph_runtime_mode != self.runtime_mode
        ):
            # CUDAGraphMode.NONE could mean the profile run, a warmup run, or
            # running without cudagraphs.
            # We do not trigger capture/replay if the runtime mode is not
            # matches. This enables properly dispatching to the correct
            # CUDAGraphWrapper when nesting multiple instances with different
            # runtime modes.
            return self.runnable(*args, **kwargs)

        if batch_descriptor not in self.concrete_cudagraph_entries:
            # create a new entry for this batch descriptor
            self.concrete_cudagraph_entries[batch_descriptor] = CUDAGraphEntry(
                batch_descriptor=batch_descriptor
            )
            #self.toggled_cudagraph[batch_descriptor] = True

        entry = self.concrete_cudagraph_entries[batch_descriptor]

        if entry.cudagraph is None:
            if self.cudagraph_options.debug_log_enable:
                # Since we capture cudagraph for many different shapes and
                # capturing is fast, we don't need to log it for every
                # shape. E.g. we only log it for the first subgraph in
                # piecewise mode.
                logger.debug(
                    "Capturing a cudagraph on (%s,%s)",
                    self.runtime_mode.name,
                    entry.batch_descriptor,
                )
            # validate that cudagraph capturing is legal at this point.
            validate_cudagraph_capturing_enabled()

            input_addresses = [
                x.data_ptr() for x in args if isinstance(x, torch.Tensor)
            ]
            entry.input_addresses = input_addresses
            cudagraph = torch.cuda.CUDAGraph(keep_graph=True)

            with ExitStack() as stack:
                if self.cudagraph_options.gc_disable:
                    # during every model forward for piecewise cudagraph
                    # mode, we will capture many pieces of cudagraphs
                    # (roughly one per layer). running gc again and again
                    # across layers will make the cudagraph capture very slow.
                    # therefore, we only run gc for the first graph,
                    # and disable gc for the rest of the graphs.
                    stack.enter_context(patch("gc.collect", lambda: None))
                    stack.enter_context(patch("torch.cuda.empty_cache", lambda: None))

                if self.graph_pool is not None:
                    set_graph_pool_id(self.graph_pool)
                else:
                    set_graph_pool_id(current_platform.graph_pool_handle())
                # mind-exploding: carefully manage the reference and memory.
                with torch.cuda.graph(cudagraph, pool=self.graph_pool):
                    # `output` is managed by pytorch's cudagraph pool
                    output = self.runnable(*args, **kwargs)
                    if self.cudagraph_options.weak_ref_output:
                        # by converting it to weak ref,
                        # the original `output` will immediately be released
                        # to save memory. It is only safe to do this for
                        # the last graph in piecewise cuadgraph mode, because
                        # the output of the last graph will not be used by
                        # any other cuda graph.
                        output = weak_ref_tensors(output)

            # here we always use weak ref for the output
            # to save memory
            entry.output = weak_ref_tensors(output)
            entry.cudagraph = cudagraph

            compilation_counter.num_cudagraph_captured += 1

            #self.cudagraph_add_conditional_node("set_conditional_kernel")
            # In this code I just add an Identical node imm. after add_scaled
            if add_identical_node == 1: 
                raw_graph = entry.cudagraph.raw_cuda_graph()            
                cu_graph = cuda.CUgraph(raw_graph)
                err,nodes, num_nodes = cuda.cuGraphGetNodes(cu_graph)   # first call: get count
                err, nodes,num_nodes_2 = cuda.cuGraphGetNodes(cu_graph, num_nodes)
                previous_node = None 
                added_node = None
                cond_node = None

                err, device = cuda.cuDeviceGet(0)
                err, context = cuda.cuCtxGetCurrent()
                for current_node in nodes:
                    #print(f"Node handle: {node}") 
                    err,params = cuda.cuGraphKernelNodeGetParams(current_node )
                    if params != None:
                        #print(f"Misha Debug vllm/compilation/cuda_graph.py:205, params.func:{params.func}")
                        try:
                                err, name = cuda.cuFuncGetName(params.func)
                                # The name usually comes back as a bytes object
                                name_str = name.decode('utf-8') if isinstance(name, bytes) else str(name)
                                #print(f"    Kernel Name: {name_str}")
                                target_add_scaled = "add_scaled_summed_vectors_kernel"
                                target_all_reduce ="AllReduce"
                                #if target_all_reduce in name_str:
                                #    print(f"Misha Debug - found all_reduce node")
                                if target_add_scaled in name_str:
                                    #node is the node we started from - is the dependancy
                                    err, added_node = cuda.cuGraphAddKernelNode(cu_graph, [current_node], 1,params) 
                                    #if err != 0:
                                    #    print(f"Misha Debug - line 223 - cuGraphAddKernelNode - err:{err}")
                                    previous_node = current_node
                                elif previous_node != None and added_node != None:
                                    err = cuda.cuGraphRemoveDependencies(cu_graph, [previous_node], [current_node], None,1) 
                                    #if err != 0:
                                    #    print(f"Misha Debug - line 228 - cuGraphRemoveDependencies- err:{err}")
                                    err = cuda.cuGraphAddDependencies(cu_graph, [added_node], [current_node], None,1)
                                    #if err != 0:
                                    #    print(f"Misha Debug - line 231 -cuGraphAddDependencies- err:{err}")
                                    previous_node = current_node 
                                    added_node = None
                                else:
                                    previous_node = None
                        except AttributeError:
                            print("    Kernel Name: [Requires CUDA 12.0+ or cuFuncGetName not found]")
            if add_cond_node == 1 :
                raw_graph = entry.cudagraph.raw_cuda_graph()            
                cu_graph = cuda.CUgraph(raw_graph)
                err,nodes, num_nodes = cuda.cuGraphGetNodes(cu_graph)   # first call: get count
                err, nodes,num_nodes_2 = cuda.cuGraphGetNodes(cu_graph, num_nodes)
                previous_node = None 
                added_node = None
                cond_node = None

                err, device = cuda.cuDeviceGet(0)
                err, context = cuda.cuCtxGetCurrent()
                for current_node in nodes:
                    err,params = cuda.cuGraphKernelNodeGetParams(current_node )
                    if params != None:
                        try:
                                err, name = cuda.cuFuncGetName(params.func)
                                name_str = name.decode('utf-8') if isinstance(name, bytes) else str(name)
                                target_add_scaled = "add_scaled_summed_vectors_kernel"
                                target_all_reduce ="AllReduce"
                                if target_add_scaled in name_str:

                                    
                                    err, node_a = cuda.cuGraphAddEmptyNode(cu_graph, [current_node], 1)

                                    # --- 1. Create the Conditional Handle ---
                                    # This handle acts as the boolean variable that the GPU checks to decide the flow.
                                    # It defaults to 0 (False). Node A should theoretically update this handle.
                                   
                                    err, cond_handle = cuda.cuGraphConditionalHandleCreate(cu_graph,context, 0, 0)

                                    # --- 2. Configure the Conditional Node Parameters ---
                                    cond_node_params = cuda.CUgraphNodeParams()
                                    cond_node_params.type = cuda.CUgraphNodeType.CU_GRAPH_NODE_TYPE_CONDITIONAL
                                    cond_node_params.conditional.handle = cond_handle
                                    cond_node_params.conditional.type = cuda.CUgraphConditionalNodeType.CU_GRAPH_COND_TYPE_IF
                                    cond_node_params.conditional.size = 1  # 1 because it's a simple IF statement (one body graph)
                                    cond_node_params.conditional.ctx = context
                                    # We must provide an array in memory for the driver to return the newly created subgraphs into.
                                    
                                    # --- 3. Add the Conditional Node to the Main Graph ---
                                    # We insert it after Node A by setting [node_a] as its dependency
                                    err, cond_node = cuda.cuGraphAddNode(cu_graph, [node_a], None,1, cond_node_params)
                                    # --- 4. Populate the Auto-Created Subgraph (The IF Body) ---
                                    # Extract the handle to the subgraph the driver just created for us
                                    #body_graph = cuda.CUgraph(cond_node_params.conditional.phGraph_out[0])
                                    body_graph = cond_node_params.conditional.phGraph_out[0]
                                    # Add nodes to the IF body subgraph. 
                                    # Notice this node has NO dependencies yet because it's the first node *inside* the subgraph.
                                    err, added_node = cuda.cuGraphAddKernelNode(body_graph, [], 0, params)
                                    
                                    previous_node = current_node 
                                elif previous_node != None and cond_node != None:
                                    err = cuda.cuGraphRemoveDependencies(cu_graph, [previous_node], [current_node], None,1)
                                    err = cuda.cuGraphAddDependencies(cu_graph, [cond_node], [current_node], None,1)
                                    previous_node = None
                                    cond_node = None 
                                    
                                    # Clean up
                        except AttributeError:
                            print("    Line 316: Kernel Name: [Requires CUDA 12.0+ or cuFuncGetName not found]")
            """
            err, from_nodes, to_nodes,edge_data,num_edges = cuda.cuGraphGetEdges(cu_graph)
            print(f"Misha Debug - line 208 - num_edges:{num_edges}")
            if num_edges > 0 and (from_nodes == None or to_nodes == None):
                err, from_nodes, to_nodes,edge_data,num_edges = cuda.cuGraphGetEdges(cu_graph, num_edges)
                for f, t in zip(from_nodes, to_nodes):
                    print(f"Edge: {f} -> {t}")
            else:
                for f, t in zip(from_nodes, to_nodes):
                    print(f"Edge: {f} -> {t}")
            """
            # important: we need to return the output, rather than
            # the weak ref of the output, so that pytorch can correctly
            # manage the memory during cuda graph capture
            return output

        if self.is_debugging_mode:
            # check if the input addresses are the same
            new_input_addresses = [
                x.data_ptr() for x in args if isinstance(x, torch.Tensor)
            ]
            assert new_input_addresses == entry.input_addresses, (
                f"Input addresses for cudagraphs are different "
                f"during replay. Expected {entry.input_addresses}, "
                f"got {new_input_addresses}"
            )

        entry.cudagraph.replay()
        graph_instantiated =1
        #if get_tensor_model_parallel_rank() == 0: 
        #    traceback.print_stack()


        if do_node_enabled_toggling == 1 and not entry.interventions_applied:
            raw_graph = entry.cudagraph.raw_cuda_graph()
            cu_graph_template = cuda.CUgraph(raw_graph)

            raw_graph_exec = entry.cudagraph.raw_cuda_graph_exec()            
            cu_graph = cuda.CUgraphExec(raw_graph_exec)
            
            err,nodes, num_nodes = cuda.cuGraphGetNodes(cu_graph_template )# first call: get count
            #node_arr = (cuda.CUgraphNode * num_nodes)()
            err, nodes,num_nodes_2 = cuda.cuGraphGetNodes(cu_graph_template, num_nodes)
            for node in nodes:
                err,params = cuda.cuGraphKernelNodeGetParams(node)
                if params != None:
                    try:
                        err, name = cuda.cuFuncGetName(params.func)
                        name_str = name.decode('utf-8') if isinstance(name, bytes) else str(name)
                        #print(f"    Kernel Name: {name_str}")
                        target_kernel_old = "add_scaled_summed_vectors_kernel"
                        target_kernel_1 = "subtract_projection_kernel"
                        target_kernel_2 = "dot_product_kernel"
                        if target_kernel_1 in name_str or target_kernel_2 in name_str:
                            #print(f"Misha Debug - found my custom node")
                            if misha_switch == 0:
                                err = cuda.cuGraphNodeSetEnabled(cu_graph,node,0)
                            else:
                                err = cuda.cuGraphNodeSetEnabled(cu_graph,node,1)
                            #if err == 0:
                            #    print("Misha Debug - Disabling custom node succeeded")
                            #else:
                            #    print(f"Misha Debug line 373 - disabling kernel code:{err}")
                    except AttributeError:
                        print("    Kernel Name: [Requires CUDA 12.0+ or cuFuncGetName not found]")
            entry.interventions_applied = True
            #if misha_switch == 1:
            #    misha_switch = 0  
            #else:
            #    misha_switch =1




        return entry.output
