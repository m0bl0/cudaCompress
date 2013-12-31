#ifndef __TUM3D__COMPRESS_HEIGHTFIELD_H__
#define __TUM3D__COMPRESS_HEIGHTFIELD_H__


#include "global.h"

#include <vector>

#include <cuda_runtime.h>

#include <cudaCompress/Instance.h>
#include <cudaCompress/util/CudaTimer.h>

#include "GPUResources.h"


struct CompressHeightfieldResources
{
    CompressHeightfieldResources()
        : pBitsUpload(nullptr), pBitsDownload(nullptr)
        , syncEventUpload(0), syncEventDownload(0) {}

    static GPUResources::Config getRequiredResources(uint sizeX, uint sizeY);

    bool create(const GPUResources::Config& config);
    void destroy();

    uint* pBitsUpload;
    uint* pBitsDownload;

    cudaEvent_t syncEventUpload;
    cudaEvent_t syncEventDownload;

    cudaCompress::util::CudaTimerResources timerEncode;
    cudaCompress::util::CudaTimerResources timerDecode;
};


// Compress one level of a heightfield image:
// - perform integer (reversible) DWT
// - truncate and store least significant bit of lowpass coefficients
// - encode highpass coefficients into bitstream
// - return lowpass coefficients (minus last bit)
// The input is assumed to be roughly zero-centered, and shouldn't use more than ~10 bits.
// Decompress works analogously.
// NOTE: Decompress modifies the lowpass image! (by appending the bit that was removed during compression)
bool compressHeightfieldOneLevel(GPUResources& shared, CompressHeightfieldResources& resources, const short* dpImage, uint sizeX, uint sizeY, short* dpLowpass, short lowpassShift, std::vector<uint>& highpassBitStream);
void decompressHeightfieldOneLevel(GPUResources& shared, CompressHeightfieldResources& resources, short* dpImage, uint sizeX, uint sizeY, short* dpLowpass, short lowpassShift, const std::vector<uint>& highpassBitStream);
void decompressHeightfieldOneLevel(GPUResources& shared, CompressHeightfieldResources& resources, short* dpImage, uint sizeX, uint sizeY, short* dpLowpass, short lowpassShift, const uint* pHighpassBits, uint highpassBitCount);

void getLowpassHeightfield(GPUResources& shared, CompressHeightfieldResources& resources, const short* dpImage, uint sizeX, uint sizeY, short* dpLowpass, short lowpassShift);


#endif
