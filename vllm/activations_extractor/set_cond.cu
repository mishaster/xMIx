// set_cond.cu
#include <cuda_runtime.h>

extern "C" __global__ void set_cond_kernel(
    cudaGraphConditionalHandle handle,
    const int* n_valid
) {
    //cudaGraphSetConditional(handle, (unsigned int)(*n_valid > 0));
    cudaGraphSetConditional(handle, (unsigned int)(*n_valid <= 0));
}
extern "C" __global__ void set_cond_kernel_thresh(
    cudaGraphConditionalHandle handle,
    const float* value,
    float threshold,
    bool greater_than
) {
    bool condition = greater_than ? (*value > threshold) : (*value <= threshold);
    cudaGraphSetConditional(handle, (unsigned int)condition);
}
