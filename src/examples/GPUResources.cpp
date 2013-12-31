#include "GPUResources.h"

#include <algorithm>
#include <cassert>

#include "cudaUtil.h"


GPUResources::Config::Config()
    : cudaDevice(-1), blockCountMax(0), elemCountPerBlockMax(0), codingBlockSize(0), log2HuffmanDistinctSymbolCountMax(0), bufferSize(0)
{
}

void GPUResources::Config::merge(const GPUResources::Config& other)
{
    if(cudaDevice == -1) {
        cudaDevice = other.cudaDevice;
    }

    if(blockCountMax == 0) {
        blockCountMax = other.blockCountMax;
    } else {
        blockCountMax = std::max(blockCountMax, other.blockCountMax);
    }

    if(elemCountPerBlockMax == 0) {
        elemCountPerBlockMax = other.elemCountPerBlockMax;
    } else {
        elemCountPerBlockMax = std::max(elemCountPerBlockMax, other.elemCountPerBlockMax);
    }

    if(codingBlockSize == 0) {
        codingBlockSize = other.codingBlockSize;
    } else {
        codingBlockSize = std::min(codingBlockSize, other.codingBlockSize);
    }

    if(log2HuffmanDistinctSymbolCountMax == 0) {
        log2HuffmanDistinctSymbolCountMax = other.log2HuffmanDistinctSymbolCountMax;
    } else {
        log2HuffmanDistinctSymbolCountMax = std::max(log2HuffmanDistinctSymbolCountMax, other.log2HuffmanDistinctSymbolCountMax);
    }

    if(bufferSize == 0) {
        bufferSize = other.bufferSize;
    } else {
        bufferSize = std::max(bufferSize, other.bufferSize);
    }
}


GPUResources::GPUResources()
    : m_pCuCompInstance(nullptr)
    , m_dpBuffer(nullptr)
    , m_bufferOffset(0)
{
}

GPUResources::~GPUResources()
{
    assert(m_pCuCompInstance == nullptr);
    assert(m_dpBuffer == nullptr);
}


bool GPUResources::create(const Config& config)
{
    m_config = config;

    assert(m_pCuCompInstance == nullptr);
    m_pCuCompInstance = cudaCompress::createInstance(m_config.cudaDevice, m_config.blockCountMax, m_config.elemCountPerBlockMax, m_config.codingBlockSize, m_config.log2HuffmanDistinctSymbolCountMax);
    if(!m_pCuCompInstance) {
        return false;
    }

    //TODO don't use cudaSafeCall, but manually check for out of memory?
    assert(m_dpBuffer == nullptr);
    cudaSafeCall(cudaMalloc(&m_dpBuffer, m_config.bufferSize));

    return true;
}

void GPUResources::destroy()
{
    cudaSafeCall(cudaFree(m_dpBuffer));
    m_dpBuffer = nullptr;

    cudaCompress::destroyInstance(m_pCuCompInstance);
    m_pCuCompInstance = nullptr;
}


byte* GPUResources::getByteBuffer(size_t bytes)
{
    assert(m_bufferOffset + bytes <= m_config.bufferSize);
    if(m_bufferOffset + bytes > m_config.bufferSize) {
        printf("ERROR: GPUResources::getByteBuffer: out of memory!\n");
        return nullptr;
    }

    byte* dpResult = m_dpBuffer + m_bufferOffset;
    m_allocatedSizes.push_back(bytes);
    m_bufferOffset += getAlignedSize(bytes, 128);

    return dpResult;
}

void GPUResources::releaseBuffer()
{
    assert(!m_allocatedSizes.empty());
    if(m_allocatedSizes.empty()) {
        printf("ERROR: GPUResources::releaseBuffer: no more buffers to release\n");
        return;
    }

    size_t lastSize = m_allocatedSizes.back();
    m_allocatedSizes.pop_back();

    m_bufferOffset -= getAlignedSize(lastSize, 128);
    assert(m_bufferOffset % 128 == 0);
}

void GPUResources::releaseBuffers(uint bufferCount)
{
    for(uint i = 0; i < bufferCount; i++) {
        releaseBuffer();
    }
}
