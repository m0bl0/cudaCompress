#include "CompressHeightfield.h"

#include <cudaCompress/BitStream.h>
#include <cudaCompress/Encode.h>
#include <cudaCompress/util/Bits.h>
#include <cudaCompress/util/DWT.h>
#include <cudaCompress/util/Quantize.h>
using namespace cudaCompress;

#include "cudaUtil.h"


static const uint g_blockCount = 3;


GPUResources::Config CompressHeightfieldResources::getRequiredResources(uint sizeX, uint sizeY)
{
    uint blockSizeX = sizeX / 2;
    uint blockSizeY = sizeY / 2;

    uint elemCount = sizeX * sizeY;
    uint elemCountPerBlock = blockSizeX * blockSizeY;

    uint bitsSize = (elemCountPerBlock + 31) / 32;

    // accumulate GPU buffer size
    size_t size = 0;

    // dpDWT
    size += getAlignedSize(elemCount * sizeof(short), 128);
    size += getAlignedSize(elemCount * sizeof(short), 128);

    // dpSymbolStreams
    size += getAlignedSize(g_blockCount * elemCountPerBlock * sizeof(Symbol16), 128);

    // dpBits
    size += getAlignedSize(bitsSize * sizeof(uint), 128);

    // build GPUResources config
    GPUResources::Config config;
    config.blockCountMax = g_blockCount;
    config.elemCountPerBlockMax = elemCountPerBlock;
    config.bufferSize = size;

    return config;
}

bool CompressHeightfieldResources::create(const GPUResources::Config& config)
{
    uint bitsSize = (config.elemCountPerBlockMax + 31) / 32;
    size_t uploadSize = bitsSize * sizeof(uint);
    cudaSafeCall(cudaMallocHost(&pBitsUpload, uploadSize, cudaHostAllocWriteCombined));
    cudaSafeCall(cudaMallocHost(&pBitsDownload, uploadSize));

    cudaSafeCall(cudaEventCreateWithFlags(&syncEventUpload, cudaEventDisableTiming));
    cudaSafeCall(cudaEventCreateWithFlags(&syncEventDownload, cudaEventDisableTiming));
    // immediately record to signal that buffers are ready to use (ie first cudaEventSynchronize works)
    cudaSafeCall(cudaEventRecord(syncEventUpload));

    return true;
}

void CompressHeightfieldResources::destroy()
{
    if(syncEventDownload) {
        cudaSafeCall(cudaEventDestroy(syncEventDownload));
        syncEventDownload = 0;
    }
    if(syncEventUpload) {
        cudaSafeCall(cudaEventDestroy(syncEventUpload));
        syncEventUpload = 0;
    }

    cudaSafeCall(cudaFreeHost(pBitsDownload));
    pBitsDownload = nullptr;
    cudaSafeCall(cudaFreeHost(pBitsUpload));
    pBitsUpload = nullptr;
}



bool compressHeightfieldOneLevel(GPUResources& shared, CompressHeightfieldResources& resources, const short* dpImage, uint sizeX, uint sizeY, short* dpLowpass, short lowpassShift, std::vector<uint>& highpassBitStream)
{
    uint blockSizeX = sizeX / 2;
    uint blockSizeY = sizeY / 2;

    uint elemCount = sizeX * sizeY;
    uint elemCountPerBlock = blockSizeX * blockSizeY;

    uint bitsSize = (elemCountPerBlock + 31) / 32;

    short* dpDWT0 = shared.getBuffer<short>(elemCount);
    short* dpDWT1 = shared.getBuffer<short>(elemCount);
    std::vector<Symbol16*> dpSymbolStreams(g_blockCount);
    for(uint i = 0; i < g_blockCount; i++) {
        dpSymbolStreams[i] = shared.getBuffer<Symbol16>(elemCountPerBlock);
    }
    uint* dpBits = shared.getBuffer<uint>(bitsSize);

    util::CudaScopedTimer timer(resources.timerEncode);

    // perform DWT
    timer("DWT");
    util::dwtIntForward(dpDWT1, dpDWT0, dpImage, sizeX, sizeY, 1);

    // copy lowpass band
    timer("Copy LP");
    cudaSafeCall(cudaMemcpy2DAsync(dpLowpass, blockSizeX * sizeof(short), dpDWT1, sizeX * sizeof(short), blockSizeX * sizeof(short), blockSizeY, cudaMemcpyDeviceToDevice));

    // truncate and download lsb of lowpass and apply shift
    timer("Process LP");
    util::removeLSB(dpLowpass, dpBits, elemCountPerBlock, lowpassShift);
    cudaSafeCall(cudaMemcpyAsync(resources.pBitsDownload, dpBits, bitsSize * sizeof(uint), cudaMemcpyDeviceToHost));
    cudaSafeCall(cudaEventRecord(resources.syncEventDownload));

    // symbolize highpass bands
    timer("Symbolize HP");
    uint nextSymbolStream = 0;
    for(uint block = 1; block < 1 + g_blockCount; block++) {
        uint x = block % 2;
        uint y = block / 2;
        uint offset = x * blockSizeX + y * blockSizeY * sizeX;

        // make (unsigned!) symbols
        util::symbolize(dpSymbolStreams[nextSymbolStream++], dpDWT1 + offset, blockSizeX, blockSizeY, 1, sizeX);
    }

    BitStream bitStream(&highpassBitStream);

    // store lsb of lowpass
    timer("Store LP LSB");
    cudaSafeCall(cudaEventSynchronize(resources.syncEventDownload));
    bitStream.writeAligned(resources.pBitsDownload, bitsSize);

    // encode
    timer("Encode");
    bitStream.reserveBitSize(bitStream.getBitPosition() + sizeX * sizeY * 16);
    bool result = encodeRLHuff(shared.m_pCuCompInstance, bitStream, &dpSymbolStreams[0], g_blockCount, elemCountPerBlock);

    timer();

    shared.releaseBuffers(g_blockCount + 3);

    return result;
}

void decompressHeightfieldOneLevel(GPUResources& shared, CompressHeightfieldResources& resources, short* dpImage, uint sizeX, uint sizeY, short* dpLowpass, short lowpassShift, const std::vector<uint>& highpassBitStream)
{
    decompressHeightfieldOneLevel(shared, resources, dpImage, sizeX, sizeY, dpLowpass, lowpassShift, highpassBitStream.data(), uint(highpassBitStream.size() * sizeof(uint) * 8));
}

void decompressHeightfieldOneLevel(GPUResources& shared, CompressHeightfieldResources& resources, short* dpImage, uint sizeX, uint sizeY, short* dpLowpass, short lowpassShift, const uint* pHighpassBits, uint highpassBitCount)
{
    uint blockSizeX = sizeX / 2;
    uint blockSizeY = sizeY / 2;

    uint elemCount = sizeX * sizeY;
    uint elemCountPerBlock = blockSizeX * blockSizeY;

    uint bitsSize = (elemCountPerBlock + 31) / 32;

    short* dpDWT0 = shared.getBuffer<short>(elemCount);
    short* dpDWT1 = shared.getBuffer<short>(elemCount);
    Symbol16* dpSymbolStream = shared.getBuffer<Symbol16>(g_blockCount * elemCountPerBlock);

    uint* dpBits = shared.getBuffer<uint>(bitsSize);

    std::vector<Symbol16*> dpSymbolStreams(g_blockCount);
    for(uint i = 0; i < g_blockCount; i++) {
        dpSymbolStreams[i] = dpSymbolStream + i * elemCountPerBlock;
    }

    util::CudaScopedTimer timer(resources.timerDecode);

    cudaSafeCall(cudaMemsetAsync(dpSymbolStream, 0, g_blockCount * elemCountPerBlock * sizeof(Symbol16)));

    BitStreamReadOnly bitStream(pHighpassBits, highpassBitCount);

    // read lowpass lsb from bitstream and upload
    // wait for previous upload to complete before overwriting cpu buffer
    timer("Sync Prev");
    cudaSafeCall(cudaEventSynchronize(resources.syncEventUpload));
    bitStream.readAligned(resources.pBitsUpload, bitsSize);
    cudaSafeCall(cudaMemcpyAsync(dpBits, resources.pBitsUpload, bitsSize * sizeof(uint), cudaMemcpyHostToDevice));
    cudaSafeCall(cudaEventRecord(resources.syncEventUpload));

    // append lsb to lowpass and apply shift
    timer("Append LP LSB");
    util::appendLSB(dpLowpass, dpBits, elemCountPerBlock, lowpassShift);

    // copy lowpass band into larger buffer
    timer("Copy LP");
    cudaSafeCall(cudaMemcpy2DAsync(dpDWT1, sizeX * sizeof(short), dpLowpass, blockSizeX * sizeof(short), blockSizeX * sizeof(short), blockSizeY, cudaMemcpyDeviceToDevice));

    // decode
    timer("Decode");
    decodeRLHuff(shared.m_pCuCompInstance, bitStream, &dpSymbolStreams[0], g_blockCount, elemCountPerBlock);

    // unsymbolize highpass bands
    timer("Unsymbolize HP");
    uint nextSymbolStream = 0;
    for(uint block = 1; block < 1 + g_blockCount; block++) {
        uint x = block % 2;
        uint y = block / 2;
        uint offset = x * blockSizeX + y * blockSizeY * sizeX;

        // get signed values back from unsigned symbols
        util::unsymbolize(dpDWT1 + offset, dpSymbolStreams[nextSymbolStream++], blockSizeX, blockSizeY, 1, sizeX);
    }

    // perform IDWT
    timer("IDWT");
    util::dwtIntInverse(dpImage, dpDWT0, dpDWT1, sizeX, sizeY, 1);

    shared.releaseBuffers(4);
}

void getLowpassHeightfield(GPUResources& shared, CompressHeightfieldResources& resources, const short* dpImage, uint sizeX, uint sizeY, short* dpLowpass, short lowpassShift)
{
    uint blockSizeX = sizeX / 2;
    uint blockSizeY = sizeY / 2;

    uint elemCount = sizeX * sizeY;
    uint elemCountPerBlock = blockSizeX * blockSizeY;

    short* dpDWT = shared.getBuffer<short>(elemCount);

    // perform DWT
    util::dwtIntForwardLowpassOnly(dpLowpass, dpDWT, dpImage, sizeX, sizeY);

    // truncate lsb of lowpass and apply shift
    util::removeLSB(dpLowpass, 0, elemCountPerBlock, lowpassShift);

    shared.releaseBuffer();
}
