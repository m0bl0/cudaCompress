#include "EncoderTestSuite.h"

#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include <cudaCompress/cudaUtil.h>

#include <cudaCompress/BitStream.h>
#include <cudaCompress/EncodeCommon.h>
#include <cudaCompress/util.h>
#include <cudaCompress/util/DWT.h>
#include <cudaCompress/util/Quantize.h>
#include <cudaCompress/util/YCoCg.h>

#include "tools/Entropy.h"
#include "tools/stb_image.h"

#include "EncodeCPU.h"


namespace cudaCompress {


bool loadImageToGPU(const std::string& filename, uint& sizeX, uint& sizeY, byte* dpImage, uint sizeMax)
{
    std::vector<byte> data;
    uint channelCount = 3;
    if(filename.length() >= 4 && filename.substr(filename.length() - 4) == ".raw") {
        // assume .raw files are 3-channel 2048^2
        uint imageSizeX = 2048, imageSizeY = 2048;
        if(imageSizeX * imageSizeY > sizeMax) {
            return false;
        }
        sizeX = imageSizeX;
        sizeY = imageSizeY;

        data.resize(sizeX * sizeY * channelCount);

        // read image data from raw file
        std::ifstream file(filename, std::ifstream::binary);
        if(!file.good()) {
            return false;
        }
        file.read((char*)data.data(), data.size() * sizeof(byte));
        file.close();
    } else {
        int imageSizeX = 0, imageSizeY = 0;
        int imageChannelCount = 0;
        byte* pData = stbi_load(filename.c_str(), &imageSizeX, &imageSizeY, &imageChannelCount, 3);
        if(imageSizeX * imageSizeY > (int)sizeMax || imageChannelCount != (int)channelCount) {
            stbi_image_free(pData);
            return false;
        }
        sizeX = imageSizeX;
        sizeY = imageSizeY;

        data.resize(sizeX * sizeY * channelCount);
        memcpy(data.data(), pData, sizeX * sizeY * channelCount * sizeof(byte));

        stbi_image_free(pData);
    }

    // upload data
    cudaSafeCall(cudaMemcpy(dpImage, data.data(), sizeX * sizeY * channelCount * sizeof(byte), cudaMemcpyHostToDevice));

    // color space transform
    util::convertRGBtoYCoCg((uchar3*)dpImage, (uchar3*)dpImage, sizeX * sizeY);

    return true;
}


std::vector<std::vector<Symbol16>> getImageSymbolStreams(const byte* dpImage, uint sizeX, uint sizeY, uint dwtLevelFrom, uint dwtLevelTo, float* dpDWTBuffer, float* dpDWTOut, Symbol16* dpSymbols, float quantizationStep)
{
    assert(dwtLevelFrom > 0);
    assert(dwtLevelFrom <= dwtLevelTo);

    uint channelCount = 3;

    std::vector<std::vector<Symbol16>> result;

    for(uint channel = 0; channel < channelCount; channel++) {
        // DWT
        util::dwtFloat2DForwardFromByte(dpDWTOut, dpDWTBuffer, dpImage + channel, sizeX, sizeY, channelCount);
        for(uint level = 1; level <= dwtLevelTo; level++) {
            util::dwtFloat2DForward(dpDWTOut, dpDWTBuffer, dpDWTOut, sizeX >> level, sizeY >> level, 1, sizeX, sizeX);

            if(level >= dwtLevelFrom) {
                uint blockSizeX = sizeX >> level;
                uint blockSizeY = sizeY >> level;

                // quantize and download highpass bands
                int offsetX = blockSizeX;
                int offsetY = blockSizeY * sizeX;

                util::quantizeToSymbols(dpSymbols, dpDWTOut + offsetX,           blockSizeX, blockSizeY, 1, quantizationStep, sizeX);
                result.emplace_back(blockSizeX * blockSizeY);
                cudaSafeCall(cudaMemcpy(result.back().data(), dpSymbols, blockSizeX * blockSizeY * sizeof(Symbol16), cudaMemcpyDeviceToHost));

                util::quantizeToSymbols(dpSymbols, dpDWTOut +           offsetY, blockSizeX, blockSizeY, 1, quantizationStep, sizeX);
                result.emplace_back(blockSizeX * blockSizeY);
                cudaSafeCall(cudaMemcpy(result.back().data(), dpSymbols, blockSizeX * blockSizeY * sizeof(Symbol16), cudaMemcpyDeviceToHost));

                util::quantizeToSymbols(dpSymbols, dpDWTOut + offsetX + offsetY, blockSizeX, blockSizeY, 1, quantizationStep, sizeX);
                result.emplace_back(blockSizeX * blockSizeY);
                cudaSafeCall(cudaMemcpy(result.back().data(), dpSymbols, blockSizeX * blockSizeY * sizeof(Symbol16), cudaMemcpyDeviceToHost));
            }
        }
    }

    return result;
}

Stats runEncoderTestSuite(
    const std::string& filenamePattern,
    uint indexMin, uint indexMax, uint indexStep,
    uint dwtLevelFrom, uint dwtLevelTo,
    float quantStepMin, float quantStepMax, float quantStepFactor,
    bool blocked
)
{
    //uint symbolCount = 1024 * 1024;
    //uint symbolMax = 511;

    uint sizeMax = 7168 * 5376; // enough for the "new test images" suite
    uint channelCount = 3;
    uint symbolMax = 255;

    uint bitsPerSymbol = getRequiredBits(symbolMax);

    bool checkResult = false;
    bool verbose = false;

    std::vector<ECoder> coders;
    coders.push_back(CODER_ARITHMETIC);
    coders.push_back(CODER_GOLOMBRICE);
    coders.push_back(CODER_HUFFMAN);
    coders.push_back(CODER_RUNLENGTH);
    coders.push_back(CODER_RUNLENGTH_GOLOMBRICE);
    coders.push_back(CODER_RUNLENGTH_HUFFMAN);
    //coders.push_back(CODER_RBUC);

    //uint step = 2;
    //std::vector<double> distParams;
    //for(uint i = step; i < 100; i += step) {
    //    distParams.push_back(i * 0.01);
    //}
    std::vector<float> quantSteps;
    for(float q = quantStepMin; q <= quantStepMax; q *= quantStepFactor) {
        quantSteps.push_back(q);
    }



    // alloc GPU buffers
    byte* dpImage = nullptr;
    float* dpDWTBuffer = nullptr;
    float* dpDWTOut = nullptr;
    Symbol16* dpSymbols = nullptr;
    cudaSafeCall(cudaMalloc(&dpImage, sizeMax * channelCount * sizeof(byte)));
    cudaSafeCall(cudaMalloc(&dpDWTBuffer, sizeMax * sizeof(float)));
    cudaSafeCall(cudaMalloc(&dpDWTOut, sizeMax * sizeof(float)));
    cudaSafeCall(cudaMalloc(&dpSymbols, sizeMax/4 * sizeof(Symbol16)));

    Stats stats;
    stats.entries.resize(quantSteps.size());

    uint imageCount = 0;
    for(uint index = indexMin; index <= indexMax; index += indexStep) {
        char buf[1024];
        sprintf_s(buf, filenamePattern.c_str(), index);
        std::string filename(buf);

        uint sizeX = 0;
        uint sizeY = 0;
        if(!loadImageToGPU(filename, sizeX, sizeY, dpImage, sizeMax)) {
            printf("Failed loading image %s!\n", filename.c_str());
            continue;
        }

        imageCount++;

        clock_t start = clock();
        printf("Image %s\n", filename.c_str());

        //for(double param : distParams) {
            //// generate symbols
            //std::vector<Symbol16> symbols(symbolCount);
            //std::geometric_distribution<uint> dist(param);
            //std::mt19937 engine;
            //for(uint s = 0; s < symbolCount; s++) {
            //    uint r = dist(engine);
            //    symbols[s] = Symbol16(r % (symbolMax + 1));
            //}
        for(size_t q = 0; q < quantSteps.size(); q++) {
            float quantStep = quantSteps[q];

            StatsEntry& statsEntry = stats.entries[q];

            std::vector<std::vector<Symbol16>> symbolStreams = getImageSymbolStreams(dpImage, sizeX, sizeY, dwtLevelFrom, dwtLevelTo, dpDWTBuffer, dpDWTOut, dpSymbols, quantStep);

            for(std::vector<Symbol16>& symbols : symbolStreams) {
                // compute entropy
                double entropy = computeEntropy(symbols.data(), symbols.size());
                if(verbose) {
                    //printf("Param: %.2f   Entropy: %.3f\n\n", param, entropy);
                    printf("QuantStep: %.2f   Entropy: %.3f\n\n", quantStep, entropy);
                }

                // update stats
                statsEntry.symbolCount += symbols.size();
                statsEntry.entropySize += entropy * symbols.size();

                // compress with each coder
                uint codingBlockSize = blocked ? 128 : uint(symbols.size());
                for(ECoder coder : coders) {
                    BitStream bitStream;
                    encodeCPU(coder, &bitStream, &symbols, codingBlockSize);
                    uint bitCount = bitStream.getBitSize();
                    if(verbose) {
                        uint bitCountUncomp = (uint)symbols.size() * bitsPerSymbol;
                        float bps = float(bitStream.getBitSize()) / float(symbols.size());
                        printf("%18s size: %8u (%4.2f bps, %4.1fx)\n", getCoderName(coder).c_str(), bitCount, bps, float(bitCountUncomp) / float(bitStream.getBitSize()));
                    }

                    statsEntry.coderSizes[coder] += bitCount;

                    if(checkResult) {
                        bitStream.setBitPosition(0);
                        std::vector<Symbol16> symbolsReconst;
                        decodeCPU(coder, &bitStream, &symbolsReconst, (uint)symbols.size(), codingBlockSize);

                        auto mismatch = std::mismatch(symbols.begin(), symbols.end(), symbolsReconst.begin());
                        if(mismatch.first != symbols.end()) {
                            printf("         !!! MISMATCH at %i !!!\n", std::distance(symbols.begin(), mismatch.first));
                        }
                    }
                }
                if(verbose) {
                    printf("\n\n");
                }
            }
        }

        clock_t end = clock();
        float time = float(end - start) / float(CLOCKS_PER_SEC);
        printf("  done in %.1f s\n\n", time);
    }

    cudaSafeCall(cudaFree(dpSymbols));
    cudaSafeCall(cudaFree(dpDWTOut));
    cudaSafeCall(cudaFree(dpDWTBuffer));
    cudaSafeCall(cudaFree(dpImage));

    return stats;
}


}
