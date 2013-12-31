#ifndef __TUM3D__COMPRESS_VOLUME_H__
#define __TUM3D__COMPRESS_VOLUME_H__


#include "global.h"

#include <vector>

#include <cuda_runtime.h>

#include <cudaCompress/Instance.h>
#include <cudaCompress/util/CudaTimer.h>

#include "GPUResources.h"


struct CompressVolumeResources
{
    CompressVolumeResources()
        : pUpload(nullptr), syncEventUpload(0) {}

    static GPUResources::Config getRequiredResources(uint sizeX, uint sizeY, uint sizeZ, uint channelCount, uint log2HuffmanDistinctSymbolCountMax = 0);

    bool create(const GPUResources::Config& config);
    void destroy();

    byte* pUpload;
    cudaEvent_t syncEventUpload;

    cudaCompress::util::CudaTimerResources timerEncode;
    cudaCompress::util::CudaTimerResources timerDecode;
};


// helper struct for multi-channel compression functions
struct VolumeChannel
{
    float*      dpImage;
    const uint* pBits;
    uint        bitCount;
    float       quantizationStepLevel0;
};


// Compress one level of a scalar volume (lossless):
// - perform integer (reversible) DWT
// - encode highpass coefficients into bitstream
// - return lowpass coefficients
// The input is assumed to be roughly zero-centered.
// Decompress works analogously.
bool compressVolumeLosslessOneLevel(GPUResources& shared, CompressVolumeResources& resources, const short* dpImage, uint sizeX, uint sizeY, uint sizeZ, short* dpLowpass, std::vector<uint>& highpassBitStream);
void decompressVolumeLosslessOneLevel(GPUResources& shared, CompressVolumeResources& resources, short* dpImage, uint sizeX, uint sizeY, uint sizeZ, const short* dpLowpass, const std::vector<uint>& highpassBitStream);

// Convenience functions for multi-level lossless compression
bool compressVolumeLossless(GPUResources& shared, CompressVolumeResources& resources, const short* dpImage, uint sizeX, uint sizeY, uint sizeZ, uint numLevels, std::vector<uint>& bits);
void decompressVolumeLossless(GPUResources& shared, CompressVolumeResources& resources, short* dpImage, uint sizeX, uint sizeY, uint sizeZ, uint numLevels, const std::vector<uint>& bits);


// Compress a volume (lossy):
// - perform numLevels DWT
// - quantize coefficients and encode into bitstream
// The input is assumed to be roughly zero-centered.
// Decompress works analogously.
bool compressVolumeFloat(GPUResources& shared, CompressVolumeResources& resources, const float* dpImage, uint sizeX, uint sizeY, uint sizeZ, uint numLevels, std::vector<uint>& bitStream, float quantizationStepLevel0, bool doRLEOnlyOnLvl0 = false);
void decompressVolumeFloat(GPUResources& shared, CompressVolumeResources& resources, float* dpImage, uint sizeX, uint sizeY, uint sizeZ, uint numLevels, const std::vector<uint>& bitStream, float quantizationStepLevel0, bool doRLEOnlyOnLvl0 = false);
void decompressVolumeFloat(GPUResources& shared, CompressVolumeResources& resources, float* dpImage, uint sizeX, uint sizeY, uint sizeZ, uint numLevels, const uint* pBits, uint bitCount, float quantizationStepLevel0, bool doRLEOnlyOnLvl0 = false);
void decompressVolumeFloatMultiChannel(GPUResources& shared, CompressVolumeResources& resources, const VolumeChannel* pChannels, uint channelCount, uint sizeX, uint sizeY, uint sizeZ, uint numLevels, bool doRLEOnlyOnLvl0 = false);


// Compress a volume (lossy):
// - quantize first
// - perform numLevels integers DWT
// - encode coefficients into bitstream
// This ensures a maximum error <= quantStep / 2
bool compressVolumeFloatQuantFirst(GPUResources& shared, CompressVolumeResources& resources, const float* dpImage, uint sizeX, uint sizeY, uint sizeZ, uint numLevels, std::vector<uint>& bitStream, float quantizationStep, bool doRLEOnlyOnLvl0 = false);
bool decompressVolumeFloatQuantFirst(GPUResources& shared, CompressVolumeResources& resources, float* dpImage, uint sizeX, uint sizeY, uint sizeZ, uint numLevels, const std::vector<uint>& bitStream, float quantizationStep, bool doRLEOnlyOnLvl0 = false);
bool decompressVolumeFloatQuantFirst(GPUResources& shared, CompressVolumeResources& resources, float* dpImage, uint sizeX, uint sizeY, uint sizeZ, uint numLevels, const uint* pBits, uint bitCount, float quantizationStep, bool doRLEOnlyOnLvl0 = false);
bool decompressVolumeFloatQuantFirstMultiChannel(GPUResources& shared, CompressVolumeResources& resources, const VolumeChannel* pChannels, uint channelCount, uint sizeX, uint sizeY, uint sizeZ, uint numLevels, bool doRLEOnlyOnLvl0 = false);


#endif
