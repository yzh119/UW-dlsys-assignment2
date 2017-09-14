#include "./c_runtime_api.h"
#include <cassert>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>

/* all your GPU kernel code, e.g. matrix_softmax_cross_entropy_kernel */

__global__ void array_set_kernel(int size, float *arr, float val) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < size) {
    	arr[x] = val;
    }
}

__global__ void broadcast_to_kernel(int size, int size_from_1, const float *input, float *output) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x < size) {
		output[x] = input[x % size_from_1];
	}
}

__global__ void reduce_sum_axis_to_zero_kernel(int size_0, int size_from_1, const float *input, float *output) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x < size_from_1) {
		output[x] = 0;
		for (int y = 0; y < size_0; ++y) {
			output[x] += input[y * size_from_1 + x];
		}
	}
}

__global__ void matrix_elementwise_add_kernel(int size, const float *matA, const float *matB, float *output) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x < size) {
		output[x] = matA[x] + matB[x];
	}
}

__global__ void matrix_elementwise_add_by_const_kernel(int size, const float *matA, float val, float *output) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x < size) {
		output[x] = matA[x] + val;
	}
}

__global__ void matrix_elementwise_multiply_kernel(int size, const float *matA, const float *matB, float *output) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x < size) {
		output[x] = matA[x] * matB[x];
	}
}

__global__ void matrix_multiply_by_const_kernel(int size, const float *matA, float val, float *output) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x < size) {
		output[x] = matA[x] * val;
	}
}

__global__ void relu_kernel(int size, const float *input, float *output) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x < size) {
		output[x] = (input[x] > 0) ? input[x]: 0;
	}
}

__global__ void relu_gradient_kernel(int size, const float *input, const float *in_grad, float *output) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x < size) {
		output[x] = (input[x] > 0) ? in_grad[x]: 0;
	}
}

__global__ void softmax_kernel(int nrow, int ncol, const float *input, float *output) {
	int y = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x +
        threadIdx.x;
   	input += y * ncol;
   	output += y * ncol;
   	float max_val = *input;
   	for (int x = 1; x < ncol; ++x) {
   		max_val = max(max_val, input[x]);
   	}

   	float sum = 0;
   	for (int x = 0; x < ncol; ++x) {
   		float now = exp(input[x] - max_val);
   		output[x] = now;
   		sum += now;
   	}

   	for (int x = 0; x < ncol; ++x) {
   		output[x] /= sum;
   	}
}


// y = inputs[0], y_ = inputs[1]
// np.mean(-np.sum(y_ * np.log(softmax(y)), axis=1), keepdims=True)
__global__ void matrix_softmax_cross_entropy_kernel(int nrow, int ncol,
        const float *input_a,
        const float *input_b,
        float *output) {
    // Dynamic shared memory, size provided at kernel launch.
    extern __shared__ float loss_per_row[];
    // Two dimensional thread blocks.
    int y = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x +
        threadIdx.x;
    if (y >= nrow) {
        return;
    }
    input_a += y * ncol;
    input_b += y * ncol;
    float maxval = *input_a;
    // Find max for a row.
    for (int x = 1; x < ncol; ++x) {
        maxval = max(maxval, input_a[x]);
    }
    // Deduct by max for a row, and raise to exp.
    float sum = 0;
    for (int x = 0; x < ncol; ++x) {
        sum += exp(input_a[x] - maxval);
    }
    // Compute per-row loss.
    float loss = 0;
    for (int x = 0; x < ncol; ++x) {
        loss -= input_b[x] * log(exp(input_a[x] - maxval) / sum);
    }
    loss_per_row[y] = loss;
    __syncthreads();
    // Compute reduce_mean across rows.
    float mean_loss = 0;
    // Use a single thread to reduce mean across rows.
    if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
        for (int i = 0; i < nrow; ++i) {
            mean_loss += loss_per_row[i];
        }
        mean_loss /= nrow;
        output[0] = mean_loss;
    }
}

int DLGpuArraySet(DLArrayHandle arr, float value) { 
	int size = 1;
    for (int i = 0; i < arr->ndim; ++i)
    	size *= arr->shape[i];
    float *p_arr = (float *)arr->data;
    dim3 threads;
    threads.x = 1024;
    int n_blocks = (size + 1023) / 1024;
    array_set_kernel<<<n_blocks, threads>>>(size, p_arr, value);
    return 0;
}

int DLGpuBroadcastTo(const DLArrayHandle input, DLArrayHandle output) {
    assert(input->ndim == output->ndim - 1);
    for (int i = 0; i < input->ndim; ++i)
    	assert(input->shape[i] == output->shape[i + 1]);
    int size_from_1 = 1, size = 1;
    for (int i = 0; i < output->ndim; ++i) {
    	if (i > 0) 
    		size_from_1 *= output->shape[i];
    	size *= output->shape[i];
    }
   	const float *p_input = (const float	*)input->data;
   	float *p_output = (float *)output->data;
   	dim3 threads;
   	threads.x = 1024;
   	int n_blocks = (size + 1023) / 1024;
   	broadcast_to_kernel<<<n_blocks, threads>>>(size, size_from_1, p_input, p_output);
    return 0;
}

int DLGpuReduceSumAxisZero(const DLArrayHandle input, DLArrayHandle output) {
    assert(input->ndim == output->ndim + 1);
    for (int i = 1; i < input->ndim; ++i)
    	assert(input->shape[i] == output->shape[i - 1]);
    int size_0 = input->shape[0], size_from_1 = 1;
    for (int i = 1; i < input->ndim; ++i)
    	size_from_1 *= input->shape[i];
   	const float *p_input = (const float *)input->data;
   	float *p_output = (float *)output->data;
   	dim3 threads;
   	threads.x = 1024;
   	int n_blocks = (size_from_1 + 1023) / 1024;
   	reduce_sum_axis_to_zero_kernel<<<n_blocks, threads>>>(size_0, size_from_1, p_input, p_output);
    return 0;
}

int DLGpuMatrixElementwiseAdd(const DLArrayHandle matA,
        const DLArrayHandle matB, DLArrayHandle output) {
    assert(matA->ndim == matB->ndim);
    assert(matA->ndim == output->ndim);
    int size = 1;
    for (int i = 0; i < matA->ndim; ++i) {
    	assert(matA->shape[i] == output->shape[i]);
    	assert(matA->shape[i] == matB->shape[i]);
    	size *= matA->shape[i];
    }
    const float *p_matA = (const float *)matA->data;
    const float	*p_matB = (const float *)matB->data;
    float *p_output = (float *)output->data;
    dim3 threads;
    threads.x = 1024;
    int n_blocks = (size + 1023) / 1024;
    matrix_elementwise_add_kernel<<<n_blocks, threads>>>(size, p_matA, p_matB, p_output);
    return 0;
}

int DLGpuMatrixElementwiseAddByConst(const DLArrayHandle input, float val,
        DLArrayHandle output) {
    assert(input->ndim == output->ndim);
    int size = 1;
    for (int i = 0; i < input->ndim; ++i) {
    	assert(input->shape[i] == output->shape[i]);
    	size *= input->shape[i];
    }
    const float *p_input = (const float *)input->data;
    float *p_output = (float *)output->data;
    dim3 threads;
    threads.x = 1024;
    int n_blocks = (size + 1023) / 1024;
    matrix_elementwise_add_by_const_kernel<<<n_blocks, threads>>>(size, p_input, val, p_output);
    return 0;
}

int DLGpuMatrixElementwiseMultiply(const DLArrayHandle matA,
        const DLArrayHandle matB,
        DLArrayHandle output) {
    assert(matA->ndim == matB->ndim);
    assert(matA->ndim == output->ndim);
    int size = 1;
    for (int i = 0; i < matA->ndim; ++i) {
    	assert(matA->shape[i] == output->shape[i]);
    	assert(matA->shape[i] == matB->shape[i]);
    	size *= matA->shape[i];
    }
    const float *p_matA = (const float *)matA->data;
    const float	*p_matB = (const float *)matB->data;
    float *p_output = (float *)output->data;
    dim3 threads;
    threads.x = 1024;
    int n_blocks = (size + 1023) / 1024;
    matrix_elementwise_multiply_kernel<<<n_blocks, threads>>>(size, p_matA, p_matB, p_output);
    return 0;
}

int DLGpuMatrixMultiplyByConst(const DLArrayHandle input, float val,
        DLArrayHandle output) {
    assert(input->ndim == output->ndim);
    int size = 1;
    for (int i = 0; i < input->ndim; ++i) {
    	assert(input->shape[i] == output->shape[i]);
    	size *= input->shape[i];
    }
    const float *p_input = (const float *)input->data;
    float *p_output = (float *)output->data;
    dim3 threads;
    threads.x = 1024;
    int n_blocks = (size + 1023) / 1024;
    matrix_multiply_by_const_kernel<<<n_blocks, threads>>>(size, p_input, val, p_output);
    return 0;
}

int DLGpuMatrixMultiply(const DLArrayHandle matA, bool transposeA,
        const DLArrayHandle matB, bool transposeB,
        DLArrayHandle matC) {
	// Column major!
	assert(matA->ndim == 2);
	assert(matB->ndim == 2);
	assert(matC->ndim == 2);
	int m = matA->shape[transposeA], k = matA->shape[!transposeA], n = matB->shape[!transposeB];
    static cublasHandle_t handle;
    cublasCreate(&handle);
    cublasOperation_t transa = transposeA? CUBLAS_OP_T: CUBLAS_OP_N;
    cublasOperation_t transb = transposeB? CUBLAS_OP_T: CUBLAS_OP_N;
    float alpha = 1., beta = 0.;
    const float *A = (const float *) matA->data;
    const float *B = (const float *) matB->data;
    float *C = (float *) matC->data;
    cublasSgemm(handle,
    	transa, transb,
    	m, n, k,
    	&alpha,
    	A, matA->shape[0],
    	B, matB->shape[0],
    	&beta,
    	C, matC->shape[0]
    	);
    cublasDestroy(handle);
    return 0;
}

int DLGpuRelu(const DLArrayHandle input, DLArrayHandle output) {
    assert(input->ndim == output->ndim);
    int size = 1;
    for (int i = 0; i < input->ndim; ++i) {
    	assert(input->shape[i] == output->shape[i]);
    	size *= input->shape[i];
    }
    const float *p_input = (const float *)input->data;
    float *p_output = (float *)output->data;
    dim3 threads;
    threads.x = 1024;
    int n_blocks = (size + 1023) / 1024;
    relu_kernel<<<n_blocks, threads>>>(size, p_input, p_output);
    return 0;
}

int DLGpuReluGradient(const DLArrayHandle input, const DLArrayHandle in_grad,
        DLArrayHandle output) {
    assert(input->ndim == output->ndim);
    int size = 1;
    for (int i = 0; i < input->ndim; ++i) {
    	assert(input->shape[i] == output->shape[i]);
    	size *= input->shape[i];
    }
    const float *p_input = (const float *)input->data;
    const float *p_in_grad = (const float *)in_grad->data;
    float *p_output = (float *)output->data;
    dim3 threads;
    threads.x = 1024;
    int n_blocks = (size + 1023) / 1024;
    relu_gradient_kernel<<<n_blocks, threads>>>(size, p_input, p_in_grad, p_output);
    return 0;
}

int DLGpuSoftmax(const DLArrayHandle input, DLArrayHandle output) {
    assert(input->ndim == 2);
    assert(output->ndim == 2);
    int nrow = input->shape[0];
    assert(nrow <= 1024 * 4);
    int ncol = input->shape[1];
    const float *p_input = (const float *)input->data;
    float *p_output = (float *)output->data;
    dim3 threads;
    if (nrow <= 1024) {
    	threads.x = nrow;
    } else {
    	threads.x = 1024;
    	threads.y = (nrow + 1023) * 1024;
    }
    softmax_kernel<<<1, threads>>>(nrow, ncol, p_input, p_output);
    return 0;
}

int DLGpuSoftmaxCrossEntropy(const DLArrayHandle input_a,
        const DLArrayHandle input_b,
        DLArrayHandle output) {
    assert(input_a->ndim == 2);
    assert(input_b->ndim == 2);
    assert(output->ndim == 1);
    assert(input_a->shape[0] == input_b->shape[0] &&
            input_a->shape[1] == input_b->shape[1]);
    int nrow = input_a->shape[0];
    // Maximum x- or y-dimension of a block = 1024
    // But we need 'nrow' shared memory, and max shared memory is 48KB.
    // Conservatively allow max 16KB shared memory.
    assert(nrow <= 1024 * 4);
    int ncol = input_a->shape[1];
    const float *input_data_a = (const float *)input_a->data;
    const float *input_data_b = (const float *)input_b->data;
    float *output_data = (float *)output->data;
    dim3 threads;
    if (nrow <= 1024) {
        threads.x = nrow; 
    } else {
        threads.x = 1024;
        threads.y = (nrow + 1023) / 1024;
    }
    // 1 block, each block with 'threads' number of threads with 'nrow' shared
    // memory size
    matrix_softmax_cross_entropy_kernel<<<1, threads, nrow * sizeof(float)>>>(
            nrow, ncol, input_data_a, input_data_b, output_data);
    return 0;
}
