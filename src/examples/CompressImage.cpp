#include "CompressImage.h"

#include <cuda_runtime.h>

#include <cudaCompress/BitStream.h>
#include <cudaCompress/Encode.h>
#include <cudaCompress/util/DWT.h>
#include <cudaCompress/util/Quantize.h>
using namespace cudaCompress;

#include "cudaUtil.h"


static const uint g_blockCountPerChannel = 4;


GPUResources::Config CompressImageResources::getRequiredResources(uint sizeX, uint sizeY, uint channelCount, uint imageCount)
{
    uint blockCountPerImage = g_blockCountPerChannel * channelCount;
    uint blockCountTotal = blockCountPerImage * imageCount;

    uint elemCount = sizeX * sizeY;
    uint elemCountPerBlock = elemCount / 4;

    // accumulate GPU buffer size
    size_t size = 0;

    // dpDWT
    size += getAlignedSize(elemCount * sizeof(float), 128);
    size += getAlignedSize(elemCount * sizeof(float), 128);

    // dpSymbolStreams
    for(uint i = 0; i < blockCountTotal; i++) {
        size += getAlignedSize(elemCountPerBlock * sizeof(Symbol16));
    }

    // dppSymbolStreams
    size += getAlignedSize(blockCountTotal * sizeof(Symbol16*));

    // build GPUResources config
    GPUResources::Config config;
    config.blockCountMax = blockCountTotal;
    config.elemCountPerBlockMax = elemCountPerBlock;
    config.bufferSize = size;

    return config;
}

bool CompressImageResources::create(const GPUResources::Config& config)
{
    size_t uploadSize = config.blockCountMax * sizeof(Symbol16*);
    cudaSafeCall(cudaMallocHost(&pUpload, uploadSize, cudaHostAllocWriteCombined));

    cudaSafeCall(cudaEventCreateWithFlags(&syncEventUpload, cudaEventDisableTiming));
    // immediately record to signal that buffers are ready to use (ie first cudaEventSynchronize works)
    cudaSafeCall(cudaEventRecord(syncEventUpload));

    return true;
}

void CompressImageResources::destroy()
{
    if(syncEventUpload) {
        cudaSafeCall(cudaEventDestroy(syncEventUpload));
        syncEventUpload = 0;
    }

    cudaSafeCall(cudaFreeHost(pUpload));
    pUpload = nullptr;
}



bool compressImageOneLevel(GPUResources& shared, CompressImageResources& resources, byte* dpImage, uint sizeX, uint sizeY, uint channelCountTotal, uint channelCountToProcess, const byte* dpLowpassReference, std::vector<uint>& highpassBitStream, float quantizationStep)
{
    const uint blockCountTotal = channelCountToProcess * g_blockCountPerChannel;

    uint blockSizeX = sizeX / 2;
    uint blockSizeY = sizeY / 2;

    uint elemCount = sizeX * sizeY;
    uint elemCountPerBlock = blockSizeX * blockSizeY;

    float* dpDWT0 = shared.getBuffer<float>(elemCount);
    float* dpDWT1 = shared.getBuffer<float>(elemCount);
    std::vector<Symbol16*> dpSymbolStreams(blockCountTotal);
    for(uint i = 0; i < blockCountTotal; i++) {
        dpSymbolStreams[i] = shared.getBuffer<Symbol16>(elemCountPerBlock);
    }

    util::CudaScopedTimer timer(resources.timerEncode);

    uint nextSymbolStreamLowpass = 0;
    uint nextSymbolStreamHighpass = channelCountToProcess;
    for(uint channel = 0; channel < channelCountToProcess; channel++) {
        // perform DWT
        timer("DWT");
        util::dwtFloat2DForwardFromByte(dpDWT1, dpDWT0, dpImage + channel, sizeX, sizeY, channelCountTotal);

        // roundtrip-quantize lowpass band, store differences to reference lowpass
        timer("Quantize LP");
        util::quantizeDifferenceToSymbolsRoundtrip2D(dpSymbolStreams[nextSymbolStreamLowpass++], dpDWT1, quantizationStep, dpLowpassReference, channelCountTotal, channel, blockSizeX, blockSizeY, sizeX);

        // roundtrip-quantize highpass bands
        timer("Quantize HP");
        int offsetX = blockSizeX;
        int offsetY = blockSizeY * sizeX;
        util::quantizeToSymbolsRoundtrip(dpSymbolStreams[nextSymbolStreamHighpass++], dpDWT1 + offsetX,           blockSizeX, blockSizeY, 1, quantizationStep, sizeX);
        util::quantizeToSymbolsRoundtrip(dpSymbolStreams[nextSymbolStreamHighpass++], dpDWT1 +           offsetY, blockSizeX, blockSizeY, 1, quantizationStep, sizeX);
        util::quantizeToSymbolsRoundtrip(dpSymbolStreams[nextSymbolStreamHighpass++], dpDWT1 + offsetX + offsetY, blockSizeX, blockSizeY, 1, quantizationStep, sizeX);

        // perform IDWT to complete roundtrip
        timer("IDWT");
        util::dwtFloat2DInverseToByte(dpImage + channel, dpDWT0, dpDWT1, sizeX, sizeY, channelCountTotal);
    }

    timer();

    BitStream bitStream(&highpassBitStream);
    bitStream.reserveBitSize(bitStream.getBitPosition() + sizeX * sizeY * 24);

    // encode lowpass increments + highpass
    timer("Encode");
    bool result = encodeRLHuff(shared.m_pCuCompInstance, bitStream, dpSymbolStreams.data(), blockCountTotal, elemCountPerBlock);

    timer();

    shared.releaseBuffers(blockCountTotal + 2);

    return result;
}

void decompressImageOneLevel(GPUResources& shared, CompressImageResources& resources, byte* dpImage, uint sizeX, uint sizeY, uint channelCountTotal, uint channelCountToProcess, const byte* dpLowpassReference, const std::vector<uint>& highpassBitStream, float quantizationStep)
{
    decompressImageOneLevel(shared, resources, dpImage, sizeX, sizeY, channelCountTotal, channelCountToProcess, dpLowpassReference, highpassBitStream.data(), uint(highpassBitStream.size() * sizeof(uint) * 8), quantizationStep);
}

void decompressImageOneLevel(GPUResources& shared, CompressImageResources& resources, byte* dpImage, uint sizeX, uint sizeY, uint channelCountTotal, uint channelCountToProcess, const byte* dpLowpassReference, const uint* pHighpassBits, uint highpassBitCount, float quantizationStep)
{
    const uint blockCountTotal = channelCountToProcess * g_blockCountPerChannel;

    uint blockSizeX = sizeX / 2;
    uint blockSizeY = sizeY / 2;

    uint elemCount = sizeX * sizeY;
    uint elemCountPerBlock = blockSizeX * blockSizeY;

    float* dpDWT0 = shared.getBuffer<float>(elemCount);
    float* dpDWT1 = shared.getBuffer<float>(elemCount);
    Symbol16* dpSymbolStream = shared.getBuffer<Symbol16>(blockCountTotal * elemCountPerBlock);
    Symbol16** dppSymbolStreams = shared.getBuffer<Symbol16*>(blockCountTotal);

    cudaSafeCall(cudaMemsetAsync(dpSymbolStream, 0, blockCountTotal * elemCountPerBlock * sizeof(Symbol16)));

    std::vector<Symbol16*> dpSymbolStreams(blockCountTotal);
    for(uint i = 0; i < blockCountTotal; i++) {
        dpSymbolStreams[i] = dpSymbolStream + i * elemCountPerBlock;
    }

    cudaSafeCall(cudaEventSynchronize(resources.syncEventUpload));
    memcpy(resources.pUpload, dpSymbolStreams.data(), blockCountTotal * sizeof(Symbol16*));
    cudaSafeCall(cudaMemcpyAsync(dppSymbolStreams, resources.pUpload, blockCountTotal * sizeof(Symbol16*), cudaMemcpyHostToDevice));
    cudaSafeCall(cudaEventRecord(resources.syncEventUpload));

    BitStreamReadOnly bitStream(pHighpassBits, highpassBitCount);

    util::CudaScopedTimer timer(resources.timerDecode);

    // decode lowpass increments + highpass
    timer("Decode");
    decodeRLHuff(shared.m_pCuCompInstance, bitStream, dpSymbolStreams.data(), blockCountTotal, elemCountPerBlock);

    uint nextSymbolStreamLowpass = 0;
    uint nextSymbolStreamHighpass = channelCountToProcess;
    for(uint channel = 0; channel < channelCountToProcess; channel++) {
        // unquantize lowpass band, add reference lowpass
        timer("Unquantize LP");
        util::unquantizeDifferenceFromSymbols2D(dpDWT0, dpSymbolStreams[nextSymbolStreamLowpass++], quantizationStep, dpLowpassReference, channelCountTotal, channel, blockSizeX, blockSizeY, sizeX);

        // unquantize highpass bands and perform IDWT
        timer("Unquantize HP + IDWT");
        util::dwtFloat2DInverseFromSymbolsToByte(dpImage + channel, dpDWT1, dpDWT0, dppSymbolStreams + nextSymbolStreamHighpass, quantizationStep, sizeX, sizeY, channelCountTotal);
        nextSymbolStreamHighpass += 3;
    }

    timer();

    shared.releaseBuffers(4);
}


bool compressImagesOneLevel(GPUResources& shared, CompressImageResources& resources, std::vector<ImageV>& images, uint sizeX, uint sizeY, uint channelCountTotal, uint channelCountToProcess)
{
    const uint blockCountPerImage = channelCountToProcess * g_blockCountPerChannel;
    const uint blockCountTotal = blockCountPerImage * uint(images.size());

    uint blockSizeX = sizeX / 2;
    uint blockSizeY = sizeY / 2;

    uint elemCount = sizeX * sizeY;
    uint elemCountPerBlock = blockSizeX * blockSizeY;

    float* dpDWT0 = shared.getBuffer<float>(elemCount);
    float* dpDWT1 = shared.getBuffer<float>(elemCount);
    std::vector<Symbol16*> dpSymbolStreams(blockCountTotal);
    for(uint i = 0; i < blockCountTotal; i++) {
        dpSymbolStreams[i] = shared.getBuffer<Symbol16>(elemCountPerBlock);
    }

    util::CudaScopedTimer timer(resources.timerEncode);

    for(size_t i = 0; i < images.size(); i++) {
        const ImageV& image = images[i];
        uint nextSymbolStreamLowpass = uint(i) * blockCountPerImage;
        uint nextSymbolStreamHighpass = nextSymbolStreamLowpass + channelCountToProcess;

        for(uint channel = 0; channel < channelCountToProcess; channel++) {
            // perform DWT
            timer("DWT");
            util::dwtFloat2DForwardFromByte(dpDWT1, dpDWT0, images[i].dpImage + channel, sizeX, sizeY, channelCountTotal);

            // roundtrip-quantize lowpass band, store differences to reference lowpass
            timer("Quantize LP");
            util::quantizeDifferenceToSymbolsRoundtrip2D(dpSymbolStreams[nextSymbolStreamLowpass++], dpDWT1, images[i].quantizationStep, images[i].dpLowpassReference, channelCountTotal, channel, blockSizeX, blockSizeY, sizeX);

            // roundtrip-quantize highpass bands
            timer("Quantize HP");
            int offsetX = blockSizeX;
            int offsetY = blockSizeY * sizeX;
            util::quantizeToSymbolsRoundtrip(dpSymbolStreams[nextSymbolStreamHighpass++], dpDWT1 + offsetX,           blockSizeX, blockSizeY, 1, images[i].quantizationStep, sizeX);
            util::quantizeToSymbolsRoundtrip(dpSymbolStreams[nextSymbolStreamHighpass++], dpDWT1 +           offsetY, blockSizeX, blockSizeY, 1, images[i].quantizationStep, sizeX);
            util::quantizeToSymbolsRoundtrip(dpSymbolStreams[nextSymbolStreamHighpass++], dpDWT1 + offsetX + offsetY, blockSizeX, blockSizeY, 1, images[i].quantizationStep, sizeX);

            // perform IDWT to complete roundtrip
            timer("IDWT");
            util::dwtFloat2DInverseToByte(images[i].dpImage + channel, dpDWT0, dpDWT1, sizeX, sizeY, channelCountTotal);
        }
    }

    timer();

    std::vector<BitStream> bitStreams;
    for(size_t i = 0; i < images.size(); i++) {
        ImageV& image = images[i];
        bitStreams.emplace_back(&image.highpassBitStream);
        bitStreams.back().reserveBitSize(bitStreams.back().getBitPosition() + sizeX * sizeY * 24);
    }
    std::vector<BitStream*> pBitStreams;
    for(size_t i = 0; i < images.size(); i++) {
        for(uint b = 0; b < blockCountPerImage; b++) {
            pBitStreams.push_back(&bitStreams[i]);
        }
    }

    // encode lowpass increments + highpass
    timer("Encode");
    bool result = encodeRLHuff(shared.m_pCuCompInstance, pBitStreams.data(), dpSymbolStreams.data(), blockCountTotal, elemCountPerBlock);

    timer();

    shared.releaseBuffers(blockCountTotal + 2);

    return result;
}

void decompressImagesOneLevel(GPUResources& shared, CompressImageResources& resources, const std::vector<ImageV>& images, uint sizeX, uint sizeY, uint channelCountTotal, uint channelCountToProcess)
{
    std::vector<Image> images2(images.size());
    for(size_t i = 0; i < images.size(); i++) {
        images2[i].dpImage            = images[i].dpImage;
        images2[i].dpLowpassReference = images[i].dpLowpassReference;
        images2[i].pHighpassBits      = const_cast<uint*>(images[i].highpassBitStream.data());
        images2[i].highpassBitCount   = uint(images[i].highpassBitStream.size() * sizeof(uint) * 8);
        images2[i].quantizationStep   = images[i].quantizationStep;
    }
    return decompressImagesOneLevel(shared, resources, images2, sizeX, sizeY, channelCountTotal, channelCountToProcess);
}

void decompressImagesOneLevel(GPUResources& shared, CompressImageResources& resources, const std::vector<Image>& images, uint sizeX, uint sizeY, uint channelCountTotal, uint channelCountToProcess)
{
    const uint blockCountPerImage = channelCountToProcess * g_blockCountPerChannel;
    const uint blockCountTotal = blockCountPerImage * uint(images.size());

    uint blockSizeX = sizeX / 2;
    uint blockSizeY = sizeY / 2;

    uint elemCount = sizeX * sizeY;
    uint elemCountPerBlock = blockSizeX * blockSizeY;

    float* dpDWT0 = shared.getBuffer<float>(elemCount);
    float* dpDWT1 = shared.getBuffer<float>(elemCount);
    Symbol16* dpSymbolStream = shared.getBuffer<Symbol16>(blockCountTotal * elemCountPerBlock);
    Symbol16** dppSymbolStreams = shared.getBuffer<Symbol16*>(blockCountTotal);

    cudaSafeCall(cudaMemsetAsync(dpSymbolStream, 0, blockCountTotal * elemCountPerBlock * sizeof(Symbol16)));

    std::vector<Symbol16*> dpSymbolStreams(blockCountTotal);
    for(uint i = 0; i < blockCountTotal; i++) {
        dpSymbolStreams[i] = dpSymbolStream + i * elemCountPerBlock;
    }

    cudaSafeCall(cudaEventSynchronize(resources.syncEventUpload));
    memcpy(resources.pUpload, dpSymbolStreams.data(), blockCountTotal * sizeof(Symbol16*));
    cudaSafeCall(cudaMemcpyAsync(dppSymbolStreams, resources.pUpload, blockCountTotal * sizeof(Symbol16*), cudaMemcpyHostToDevice));
    cudaSafeCall(cudaEventRecord(resources.syncEventUpload));

    std::vector<BitStreamReadOnly> bitStreams;
    for(size_t i = 0; i < images.size(); i++) {
        const Image& image = images[i];
        bitStreams.emplace_back(image.pHighpassBits, image.highpassBitCount);
    }
    std::vector<BitStreamReadOnly*> pBitStreams;
    for(size_t i = 0; i < images.size(); i++) {
        for(uint b = 0; b < blockCountPerImage; b++) {
            pBitStreams.push_back(&bitStreams[i]);
        }
    }

    util::CudaScopedTimer timer(resources.timerDecode);

    // decode lowpass increments + highpass
    timer("Decode");
    decodeRLHuff(shared.m_pCuCompInstance, pBitStreams.data(), dpSymbolStreams.data(), blockCountTotal, elemCountPerBlock);

    for(size_t i = 0; i < images.size(); i++) {
        const Image& image = images[i];
        uint nextSymbolStreamLowpass = uint(i) * blockCountPerImage;
        uint nextSymbolStreamHighpass = nextSymbolStreamLowpass + channelCountToProcess;
        for(uint channel = 0; channel < channelCountToProcess; channel++) {
            // unquantize lowpass band, add reference lowpass
            timer("Unquantize LP");
            util::unquantizeDifferenceFromSymbols2D(dpDWT0, dpSymbolStreams[nextSymbolStreamLowpass++], image.quantizationStep, image.dpLowpassReference, channelCountTotal, channel, blockSizeX, blockSizeY, sizeX);

            // unquantize highpass bands and perform IDWT
            timer("Unquantize HP + IDWT");
            util::dwtFloat2DInverseFromSymbolsToByte(image.dpImage + channel, dpDWT1, dpDWT0, dppSymbolStreams + nextSymbolStreamHighpass, image.quantizationStep, sizeX, sizeY, channelCountTotal);
            nextSymbolStreamHighpass += 3;
        }
    }

    timer();

    shared.releaseBuffers(4);
}


void getLowpassImage(GPUResources& shared, CompressImageResources& resources, const byte* dpImage, uint sizeX, uint sizeY, uint channelCountTotal, uint channelCountToProcess, byte* dpLowpass)
{
    uint blockSizeX = sizeX / 2;
    uint blockSizeY = sizeY / 2;

    uint elemCount = sizeX * sizeY;
    uint elemCountPerBlock = blockSizeX * blockSizeY;

    float* dpDWT0 = shared.getBuffer<float>(elemCount);
    float* dpDWT1 = shared.getBuffer<float>(elemCount);

    for(uint channel = 0; channel < channelCountToProcess; channel++) {
        // perform DWT
        util::dwtFloat2DForwardLowpassOnlyFromByte(dpDWT1, dpDWT0, dpImage + channel, sizeX, sizeY, channelCountTotal);

        // quantize lowpass band
        util::floatToByte2D(dpLowpass, channelCountTotal, channel, dpDWT1, blockSizeX, blockSizeY, blockSizeX);
    }

    shared.releaseBuffers(2);
}
