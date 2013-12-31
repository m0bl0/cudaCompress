#ifndef __TUM3D__COMPRESS_IMAGE_H__
#define __TUM3D__COMPRESS_IMAGE_H__


#include "global.h"

#include <vector>

#include <cuda_runtime.h>

#include <cudaCompress/Instance.h>
#include <cudaCompress/util/CudaTimer.h>

#include "GPUResources.h"


struct CompressImageResources
{
    CompressImageResources()
        : pUpload(nullptr), syncEventUpload(0) {}

    static GPUResources::Config getRequiredResources(uint sizeX, uint sizeY, uint channelCount, uint imageCount = 1);

    bool create(const GPUResources::Config& config);
    void destroy();

    byte* pUpload;
    cudaEvent_t syncEventUpload;

    cudaCompress::util::CudaTimerResources timerEncode;
    cudaCompress::util::CudaTimerResources timerDecode;
};


// Compress/decompress one level of an image:
// - perform DWT
// - quantize and encode difference between lowpass coefficients and reference lowpass (i.e. reconstructed image from next level)
// - quantize and encode highpass coefficients into bitstream
// - compress performs roundtrip, i.e. dpImage is updated to the reconstructed image
// Processes the first channelCountToProcess channels, out of channelCountTotal channels
// (useful e.g. to process only RGB out of an RGBA image).
// Restrictions: 1 <= channelCountTotal <= 4 and channelCountToProcess <= channelCountTotal.
// Does not perform color space conversion!
bool compressImageOneLevel(GPUResources& shared, CompressImageResources& resources, byte* dpImage, uint sizeX, uint sizeY, uint channelCountTotal, uint channelCountToProcess, const byte* dpLowpassReference, std::vector<uint>& highpassBitStream, float quantizationStep);
void decompressImageOneLevel(GPUResources& shared, CompressImageResources& resources, byte* dpImage, uint sizeX, uint sizeY, uint channelCountTotal, uint channelCountToProcess, const byte* dpLowpassReference, const std::vector<uint>& highpassBitStream, float quantizationStep);
void decompressImageOneLevel(GPUResources& shared, CompressImageResources& resources, byte* dpImage, uint sizeX, uint sizeY, uint channelCountTotal, uint channelCountToProcess, const byte* dpLowpassReference, const uint* pHighpassBits, uint highpassBitCount, float quantizationStep);

// Process multiple images at once:
struct ImageV
{
    ImageV() : dpImage(nullptr), dpLowpassReference(nullptr), quantizationStep(0.0f) {}

    byte* dpImage;
    byte* dpLowpassReference;
    std::vector<uint> highpassBitStream;
    float quantizationStep;
};
struct Image
{
    Image() : dpImage(nullptr), dpLowpassReference(nullptr), pHighpassBits(nullptr), highpassBitCount(0), quantizationStep(0.0f) {}

    byte* dpImage;
    byte* dpLowpassReference;
    uint* pHighpassBits;
    uint highpassBitCount;
    float quantizationStep;
};
bool compressImagesOneLevel(GPUResources& shared, CompressImageResources& resources, std::vector<ImageV>& images, uint sizeX, uint sizeY, uint channelCountTotal, uint channelCountToProcess);
void decompressImagesOneLevel(GPUResources& shared, CompressImageResources& resources, const std::vector<ImageV>& images, uint sizeX, uint sizeY, uint channelCountTotal, uint channelCountToProcess);
void decompressImagesOneLevel(GPUResources& shared, CompressImageResources& resources, const std::vector<Image>& images, uint sizeX, uint sizeY, uint channelCountTotal, uint channelCountToProcess);

// Helper: Get lowpass band of DWT
void getLowpassImage(GPUResources& shared, CompressImageResources& resources, const byte* dpImage, uint sizeX, uint sizeY, uint channelCountTotal, uint channelCountToProcess, byte* dpLowpass);


#endif
