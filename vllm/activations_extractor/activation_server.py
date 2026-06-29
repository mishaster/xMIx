from multiprocessing import shared_memory
import numpy as np
import torch
import torch.distributed
import torch.nn as nn
import time
import socket
import sys
import os

sys.path.append(os.path.abspath("/home/michael/Language-Model-SAEs/src/lm_saes"))

from lm_saes.sae import SparseAutoEncoder

class ActivationServer() :
    def __init__(
        self,
        shm_buffer:torch.Tensor,
        pinned_activations_buffer:torch.Tensor,
        dtype
    ):
        self.counter =0
        self.shm_tensor = shm_buffer	
        # This socket IS your synchronization object
        self.buffer_copied_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.buffer_copied_socket.bind(("127.0.0.1", 0))
        self.app_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.app_socket.bind(("127.0.0.1", 0))
        self.pinned_staging = pinned_activations_buffer
        #sae_path = "/home/michael/Llama-Scope/Llama3_1-8B-Base-LXR-8x/Llama3_1-8B-Base-L17R-8x"
        #with torch.device("cpu"):
        #    self.sae = SparseAutoEncoder.from_pretrained(sae_path)
        #self.sae = self.sae.to("cpu")
        #self.sae = self.sae.to(dtype)        # converts weights to same dtype as model (float16)
        return None

    def get_socket_buffer_port(self):
        #print(f"Activation Server socket is:{self.buffer_copied_socket.getsockname()[1]}")
        return self.buffer_copied_socket.getsockname()[1] 

    def get_socket_app_port(self):
        return self.app_socket.getsockname()[1]
    def wait_for_activations(self):
        #print("Activation serving Server - Start Working...")
        #self.sae = self.sae.to("cpu")
        #original_decoder_norm = self.sae.decoder_norm

        # ...and create a new function that calls the original but forces the result to CPU
        def cpu_safe_decoder_norm():
            # Call the original method (which returns the rogue GPU tensor)
            gpu_result = original_decoder_norm()
            # Move it to CPU before returning
            return gpu_result.to("cpu")

        # Replace the method on the object with our safe version
        #self.sae.decoder_norm = cpu_safe_decoder_norm
        while True:
            # 1. This blocks the thread. The OS puts it to sleep.
            # It will strictly wait here.
            #print(f"Activation Server thread waiting on {self.buffer_copied_socket.getsockname()[0]},{self.buffer_copied_socket.getsockname()[1] }",flush = True)
            num_tokens= int(self.buffer_copied_socket.recv(32).decode()) 
            for i in range(32):
                #pinned_staging_of_size = self.pinned_staging[i,:num_tokens]
                self.shm_tensor[i,:num_tokens].copy_(self.pinned_staging[i,:num_tokens])

            # Misha - Remove comment
            #print(f"Input Mean: {layer_17_hidden.mean()}, Input Max: {layer_17_hidden.max()}")
            #print(f"DEBUG: Input Tensor Device: {layer_17_hidden.device}")
            # Check a specific weight inside the SAE that caused the crash
            #print(f"DEBUG: SAE Weight Device: {self.sae.decoder_norm.weight.device if hasattr(self.sae.decoder_norm, 'weight') else 'Unknown'}")
            # Check the module itself
            #print(f"DEBUG: SAE Module Device: {next(self.sae.parameters()).device}")


            #layer_17_hidden = self.shm_tensor[16,:num_tokens]
            #with torch.device("cpu"):
            #    features = self.sae.encode(layer_17_hidden)
            #top_feats_per_token = torch.topk(features, k=10, dim=-1)
            #print(f"A. server: top feats per token are:{top_feats_per_token}")


            # Misha - end

            #features = self.sae.encode(layer_17_hidden) 
            #self.counter +=1
            #print(f"Misha Debug - Activations_server- Received token activations num:{self.counter}")
            #print(f"Misha debug - ACT. SERVER received: num_tokens:{num_tokens}")
            # 2. If we reach this line, we were just woken up!
            #print("Activation Notification Received! Reading shared memory now...",flush=True)
            # Notify consumer of data arrival


