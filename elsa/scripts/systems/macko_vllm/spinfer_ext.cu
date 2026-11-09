// torch binding for the SpInfer (EuroSys'25) TCA-BME SpMM kernel, so vLLM can call it inside its
// CUDA graphs. The kernel source is compiled into this extension (shared cudart with torch).
//   torch.ops.spinfer.spmm(x[N,K] fp16, vals, tog, tom, bitmap, maxnnz, M, split_k) -> y[N,M]
// N must be 8/16/32/64/128 or a multiple of 128 (caller pads). y = x @ W^T with W the compressed
// [M,K] weight: SpInfer computes C[M,N] (col-major) = A[M,K] (row-major) x B[K,N] (col-major); a
// row-major x[N,K] is B and a row-major y[N,M] is C, so no transposes are needed.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "SpMM_API.cu"

static torch::Tensor spmm(torch::Tensor x, torch::Tensor vals, torch::Tensor tog, torch::Tensor tom,
                          torch::Tensor bitmap, torch::Tensor maxnnz, int64_t M, int64_t split_k) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kHalf && x.dim() == 2 && x.is_contiguous(),
                "x must be contiguous fp16 [N,K] on cuda");
    const int N = x.size(0), K = x.size(1);
    TORCH_CHECK(N == 8 || N == 16 || N == 32 || N == 64 || N % 128 == 0, "unsupported N ", N);
    TORCH_CHECK(M % 64 == 0 && K % 64 == 0, "M,K must be multiples of 64");
    const at::cuda::OptionalCUDAGuard guard(x.device());
    auto y = torch::empty({N, M}, x.options());
    torch::Tensor ws;
    if (split_k > 1) ws = torch::empty({split_k * M * N}, x.options());
    cudaStream_t st = at::cuda::getCurrentCUDAStream();
    const half* cv = reinterpret_cast<const half*>(vals.data_ptr<at::Half>());
    cudaError_t e = SpMM_SplitK_API_bitmap_v3(
        st, cv /* dense A: unused by the kernel */, cv, tog.data_ptr<int>(), tom.data_ptr<int>(),
        reinterpret_cast<const uint64_t*>(bitmap.data_ptr<int64_t>()), maxnnz.data_ptr<int>(),
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()), reinterpret_cast<half*>(y.data_ptr<at::Half>()),
        (int)M, N, K, split_k > 1 ? reinterpret_cast<half*>(ws.data_ptr<at::Half>()) : nullptr, (int)split_k);
    TORCH_CHECK(e == cudaSuccess, "SpInfer SpMM failed: ", cudaGetErrorString(e));
    return y;
}

TORCH_LIBRARY(spinfer, m) {
    m.def("spmm(Tensor x, Tensor vals, Tensor tog, Tensor tom, Tensor bitmap, Tensor maxnnz, int M, int split_k) -> Tensor");
}
TORCH_LIBRARY_IMPL(spinfer, CUDA, m) { m.impl("spmm", &spmm); }
