#ifndef __TUM3D__CUDA_UTIL_H__
#define __TUM3D__CUDA_UTIL_H__


#include <cstdlib>
#include <cstdio>

#include <cuda_runtime.h>


#define cudaSafeCall(err) __cudaSafeCall(err, __FILE__, __LINE__)
#define cudaCheckMsg(msg) __cudaCheckMsg(msg, __FILE__, __LINE__)


#ifdef _DEBUG
#define CHECK_ERROR(err) (cudaSuccess != err || cudaSuccess != (err = cudaDeviceSynchronize()))
#else
#define CHECK_ERROR(err) (cudaSuccess != err)
#endif

inline void __cudaSafeCall(cudaError err, const char* file, const int line)
{
    if(CHECK_ERROR(err)) {
        fprintf(stderr, "%s(%i) : cudaSafeCall() Runtime API error : %s.\n", file, line, cudaGetErrorString(err));
#ifdef _DEBUG
        __debugbreak();
#endif
    }
}

inline void __cudaCheckMsg(const char* errorMessage, const char* file, const int line)
{
    cudaError_t err = cudaGetLastError();
    if(CHECK_ERROR(err)) {
        fprintf(stderr, "%s(%i) : cudaCheckMsg() CUDA error : %s : (%d) %s.\n", file, line, errorMessage, (int)err, cudaGetErrorString(err));
#ifdef _DEBUG
        __debugbreak();
#endif
    }
}

#undef CHECK_ERROR


#endif
