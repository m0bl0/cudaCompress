#include "examples.h"

#include <cmath>
#include <algorithm>
#include <fstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include <cudaCompress/Instance.h>
#include <cudaCompress/Encode.h>
#include <cudaCompress/util/Bits.h>
#include <cudaCompress/util/DWT.h>
#include <cudaCompress/util/Quantize.h>
#include <cudaCompress/util/YCoCg.h>
#include <cudaCompress/Timing.h>
using namespace cudaCompress;

#include "tthread/tinythread.h"

#include "tools/entropy.h"
#include "tools/imgtools.h"
#include "tools/rawfile.h"

#include "cudaUtil.h"

#include "CompressImage.h"
#include "CompressHeightfield.h"
#include "CompressVolume.h"


#pragma warning( push )
#pragma warning( disable : 4996 ) // "fopen may be unsafe"


int compressImageScalable(const std::string& filenameOrig, uint width, uint height, uint levelCount, float quantStep)
{
    uint byteCount = width * height * 3;

    // read file
    std::vector<byte> image(byteCount);
    if(!readByteRaw(filenameOrig, byteCount, image.data())) {
        printf("Failed opening file %s\n", filenameOrig.c_str());
        return -1;
    }

    std::vector<byte*> dpLevelData(levelCount + 1);
    uint byteCountLevel = byteCount;
    for(uint level = 0; level <= levelCount; level++) {
        cudaSafeCall(cudaMalloc(&dpLevelData[level], byteCountLevel));
        byteCountLevel /= 4;
    }

    cudaSafeCall(cudaMemcpy(dpLevelData[0], image.data(), width * height * 3, cudaMemcpyHostToDevice));


    GPUResources::Config config = CompressImageResources::getRequiredResources(width, height, 3);
    GPUResources shared;
    shared.create(config);
    CompressImageResources res;
    res.create(shared.getConfig());


    util::convertRGBtoYCoCg((uchar3*)dpLevelData[0], (uchar3*)dpLevelData[0], width * height);

    // propagate
    for(uint level = 0; level < levelCount; level++) {
        uint widthLevel  = width  >> level;
        uint heightLevel = height >> level;
        getLowpassImage(shared, res, dpLevelData[level], widthLevel, heightLevel, 3, 3, dpLevelData[level+1]);
    }

    // compress and decompress
    size_t compressedSize = 0;
    std::vector<std::vector<uint>> levelBitStream(levelCount);
    for(uint level1 = levelCount; level1 > 0; level1--) {
        uint level = level1 - 1;
        uint widthLevel  = width  >> level;
        uint heightLevel = height >> level;
        compressImageOneLevel(shared, res, dpLevelData[level], widthLevel, heightLevel, 3, 3, dpLevelData[level+1], levelBitStream[level], quantStep);

        compressedSize += levelBitStream[level].size() * sizeof(uint);

        cudaSafeCall(cudaMemset(dpLevelData[level], 0, widthLevel * heightLevel * 3));
        decompressImageOneLevel(shared, res, dpLevelData[level], widthLevel, heightLevel, 3, 3, dpLevelData[level+1], levelBitStream[level], quantStep);
    }
    compressedSize += (width / (1 << levelCount)) * (height / (1 << levelCount)) * 3;

    util::convertYCoCgtoRGB((uchar3*)dpLevelData[0], (uchar3*)dpLevelData[0], width * height);


    res.destroy();
    shared.destroy();


    cudaSafeCall(cudaMemcpy(image.data(), dpLevelData[0], width * height * 3, cudaMemcpyDeviceToHost));

    writeByteRaw("out.raw", width * height * 3, image.data());

    printf("PSNR: %.2f", computePSNRByteRaws(filenameOrig, "out.raw", width * height * 3));
    printf("     Compressed size: %8llu   bpp: %.2f\n", compressedSize, float(compressedSize * 8) / float(width * height));


    for(uint level = 0; level <= levelCount; level++) {
        cudaSafeCall(cudaFree(dpLevelData[level]));
    }

    return 0;
}


std::vector<byte> copyImageTile(const std::vector<byte>& image, uint sizeX, uint sizeY, uint channelCount, uint offsetX, uint offsetY, uint tileSizeX, uint tileSizeY)
{
    std::vector<byte> result(tileSizeX * tileSizeY * channelCount);

    for(uint y = 0; y < tileSizeY; y++) {
        for(uint x = 0; x < tileSizeX; x++) {
            uint indexImage = (offsetX + x) + sizeX * (offsetY + y);
            uint indexTile = x + tileSizeX * y;
            for(uint c = 0; c < channelCount; c++) {
                result[indexTile * channelCount + c] = image[indexImage * channelCount + c];
            }
        }
    }

    return result;
}

void copyTileIntoImage(std::vector<byte>& image, uint sizeX, uint sizeY, uint channelCount, const std::vector<byte>& tile, uint offsetX, uint offsetY, uint tileSizeX, uint tileSizeY)
{
    for(uint y = 0; y < tileSizeY; y++) {
        for(uint x = 0; x < tileSizeX; x++) {
            uint indexImage = (offsetX + x) + sizeX * (offsetY + y);
            uint indexTile = x + tileSizeX * y;
            for(uint c = 0; c < channelCount; c++) {
                image[indexImage * channelCount + c] = tile[indexTile * channelCount + c];
            }
        }
    }
}

std::vector<byte> expandChannels(const std::vector<byte>& image, uint channelCountBefore, uint channelCountAfter)
{
    uint pixelCount = uint(image.size()) / channelCountBefore;
    std::vector<byte> result(pixelCount * channelCountAfter);
    for(uint i = 0; i < pixelCount; i++) {
        for(uint c = 0; c < std::min(channelCountBefore, channelCountAfter); c++) {
            result[channelCountAfter*i+c] = image[channelCountBefore*i+c];
        }
        for(uint c = channelCountBefore; c < channelCountAfter; c++) {
            result[channelCountAfter*i+c] = 0;
        }
    }
    return result;
}

int benchmarkOneLevelImage(const std::string& filenameOrig, uint width, uint height, uint tiles, float quantizationStep, uint iterations, ETimingDetail timingDetail)
{
    printf("Image \"%s\" in %ix%i tiles:\n", filenameOrig.c_str(), tiles, tiles);

    const uint tileCount = tiles * tiles;
    const uint tileWidth  = width  / tiles;
    const uint tileHeight = height / tiles;
    const uint blockWidth  = tileWidth  / 2;
    const uint blockHeight = tileHeight / 2;

    const uint elemCountTotal = width * height;
    const uint elemCountPerTile = tileWidth * tileHeight;
    const uint elemCountPerBlock = blockWidth * blockHeight;
    const uint blockCount = 3;

    // read file
    std::vector<byte> imageData(elemCountTotal*3);
    if(!readByteRaw(filenameOrig, elemCountTotal*3, imageData.data())) {
        printf("Failed opening file %s\n", filenameOrig.c_str());
        return -1;
    }

    // separate into tiles and expand to four channels
    std::vector<std::vector<byte>> tileData(tileCount);
    for(uint tileY = 0; tileY < tiles; tileY++) {
        for(uint tileX = 0; tileX < tiles; tileX++) {
            tileData[tileX + tiles * tileY] = expandChannels(copyImageTile(imageData, width, height, 3, tileX * tileWidth, tileY * tileHeight, tileWidth, tileHeight), 3, 4);
        }
    }


    std::vector<ImageV> images(tileCount);

    // allocate GPU arrays and upload data
    for(uint t = 0; t < tileCount; t++) {
        cudaSafeCall(cudaMalloc(&images[t].dpImage, elemCountPerTile * 4 * sizeof(byte)));
        cudaSafeCall(cudaMemcpy(images[t].dpImage, tileData[t].data(), elemCountPerTile * 4 * sizeof(byte), cudaMemcpyHostToDevice));
        cudaSafeCall(cudaMalloc(&images[t].dpLowpassReference, elemCountPerBlock * 4 * sizeof(byte)));

        images[t].quantizationStep = quantizationStep;

        util::convertRGBtoYCoCg((uchar4*)images[t].dpImage, (uchar4*)images[t].dpImage, elemCountPerTile);
    }


    cudaEvent_t start, end;
    cudaSafeCall(cudaEventCreate(&start));
    cudaSafeCall(cudaEventCreate(&end));
    float time = 0.0f;

    GPUResources::Config config = CompressImageResources::getRequiredResources(tileWidth, tileHeight, 3, tileCount);
    GPUResources shared;
    shared.create(config);
    CompressImageResources res;
    res.create(shared.getConfig());
    setTimingDetail(shared.m_pCuCompInstance, timingDetail);
    if(timingDetail != TIMING_DETAIL_NONE) {
        res.timerEncode.setEnabled(true);
        res.timerDecode.setEnabled(true);
    }

    //cudaProfilerStart();
    cudaSafeCall(cudaEventRecord(start));

    for(uint i = 0; i < iterations; i++) {
        for(uint t = 0; t < tileCount; t++) {
            getLowpassImage(shared, res, images[t].dpImage, tileWidth, tileHeight, 4, 3, images[t].dpLowpassReference); // 1.0 ms for 2k^2
        }
        compressImagesOneLevel(shared, res, images, tileWidth, tileHeight, 4, 3);
    }

    cudaSafeCall(cudaEventRecord(end));
    //cudaProfilerStop();

    cudaSafeCall(cudaEventSynchronize(end));
    cudaSafeCall(cudaEventElapsedTime(&time, start, end));
    float throughput = float(iterations * elemCountPerBlock * blockCount * tileCount) * 1000.0f / (time * 1024.0f * 1024.0f);
    printf("Compress:   %6.2f ms  (%7.2f MPix/s)\n", time / float(iterations), throughput);
    if(timingDetail != TIMING_DETAIL_NONE) {
        printf("Detailed Timings (sum for %i iterations):\n", iterations);
        printTimings(shared.m_pCuCompInstance);
        resetTimings(shared.m_pCuCompInstance);
        printf("\n");
    }

    size_t totalSize = 0;
    for(uint t = 0; t < tileCount; t++) {
        totalSize += images[t].highpassBitStream.size();
    }
    printf("Encoded highpass size: %.2f kB\n\n", float(totalSize * sizeof(uint)) / 1024.0f);
    //encodePrintBitCounts();
    //printf("\n");


    for(uint t = 0; t < tileCount; t++) {
        cudaSafeCall(cudaMemset(images[t].dpImage, 0, elemCountPerTile * 4 * sizeof(byte)));

        cudaSafeCall(cudaHostRegister(images[t].highpassBitStream.data(), images[t].highpassBitStream.size() * sizeof(uint), cudaHostRegisterDefault));
    }

    cudaProfilerStart();
    cudaSafeCall(cudaEventRecord(start));

    for(uint i = 0; i < iterations; i++) {
        decompressImagesOneLevel(shared, res, images, tileWidth, tileHeight, 4, 3);
        //for(uint t = 0; t < tileCount; t++) {
        //    decompressImageOneLevel(shared, res, images[t].dpImage, tileWidth, tileHeight, 4, 3, images[t].dpLowpassReference, bitStreams[t], quantizationStep);
        //}
    }

    cudaSafeCall(cudaEventRecord(end));
    cudaProfilerStop();

    for(uint t = 0; t < tileCount; t++) {
        cudaSafeCall(cudaHostUnregister(images[t].highpassBitStream.data()));
    }

    cudaSafeCall(cudaEventSynchronize(end));
    cudaSafeCall(cudaEventElapsedTime(&time, start, end));
    throughput = float(iterations * elemCountPerBlock * blockCount * tileCount) * 1000.0f / (time * 1024.0f * 1024.0f);
    printf("Decompress: %6.2f ms  (%7.2f MPix/s)\n", time / float(iterations), throughput);
    if(timingDetail != TIMING_DETAIL_NONE) {
        printf("Detailed Timings (sum for %i iterations):\n", iterations);
        std::vector<std::string> names;
        std::vector<float> times;
        res.timerDecode.getAccumulatedTimes(names, times, true);
        for(size_t i = 0; i < names.size(); i++) {
            printf("%s: %.1f ms\n", names[i].c_str(), times[i]);
        }
        printTimings(shared.m_pCuCompInstance);
        resetTimings(shared.m_pCuCompInstance);
        printf("\n");
    }

    for(uint t = 0; t < tileCount; t++) {
        util::convertYCoCgtoRGB((uchar4*)images[t].dpImage, (uchar4*)images[t].dpImage, elemCountPerTile);

        memset(tileData[t].data(), 0, elemCountPerTile * 4 * sizeof(byte));
        cudaSafeCall(cudaMemcpy(tileData[t].data(), images[t].dpImage, elemCountPerTile * 4 * sizeof(byte), cudaMemcpyDeviceToHost));

        uint tileX = t % tiles;
        uint tileY = t / tiles;
        copyTileIntoImage(imageData, width, height, 3, expandChannels(tileData[t], 4, 3), tileX * tileWidth, tileY * tileHeight, tileWidth, tileHeight);
    }
    FILE* file = fopen("out.raw", "wb");
    fwrite(imageData.data(), elemCountTotal * 3, 1, file);
    fclose(file);

    res.destroy();
    shared.destroy();

    for(uint t = 0; t < tileCount; t++) {
        cudaSafeCall(cudaFree(images[t].dpLowpassReference));
        cudaSafeCall(cudaFree(images[t].dpImage));
    }

    //printf("Highpass size: %u\n", bitStream.getRawSizeBytes());
    printf("PSNR: %.2f\n", computePSNRByteRaws(filenameOrig, "out.raw", elemCountTotal * 3));
    printf("\n\n");

    return 0;
}

int benchmarkOneLevelHeightfield(const std::string& filenameOrig, uint width, uint height, uint iterations, ETimingDetail timingDetail)
{
    printf("Heightfield \"%s\":\n", filenameOrig.c_str());

    const uint blockWidth  = width  / 2;
    const uint blockHeight = height / 2;
    const uint elemCountTotal = width * height;
    const uint elemCountPerBlock = blockWidth * blockHeight;
    const uint blockCount = 3;

    // read file
    std::vector<short> field(elemCountTotal);
    if(!readShortRaw(filenameOrig, elemCountTotal, field.data())) {
        printf("Failed opening file %s\n", filenameOrig.c_str());
        return -1;
    }

    // allocate GPU arrays and upload data
    short* dpImage = 0;
    cudaSafeCall(cudaMalloc(&dpImage, elemCountTotal * sizeof(short)));
    cudaSafeCall(cudaMemcpy(dpImage, field.data(), elemCountTotal * sizeof(short), cudaMemcpyHostToDevice));
    short* dpLowpass = 0;
    cudaSafeCall(cudaMalloc(&dpLowpass, elemCountPerBlock * sizeof(short)));

    cudaEvent_t start, end;
    cudaSafeCall(cudaEventCreate(&start));
    cudaSafeCall(cudaEventCreate(&end));
    float time = 0.0f;

    GPUResources::Config config = CompressHeightfieldResources::getRequiredResources(width, height);
    GPUResources shared;
    shared.create(config);
    CompressHeightfieldResources res;
    res.create(shared.getConfig());
    setTimingDetail(shared.m_pCuCompInstance, timingDetail);

    std::vector<uint> bitStream;

    //cudaProfilerStart();
    cudaSafeCall(cudaEventRecord(start));

    for(uint i = 0; i < iterations; i++) {
        compressHeightfieldOneLevel(shared, res, dpImage, width, height, dpLowpass, 0, bitStream);
    }

    cudaSafeCall(cudaEventRecord(end));
    //cudaProfilerStop();

    cudaSafeCall(cudaEventSynchronize(end));
    cudaSafeCall(cudaEventElapsedTime(&time, start, end));
    float throughput = float(iterations * elemCountPerBlock * blockCount) * 1000.0f / (time * 1024.0f * 1024.0f);
    printf("Compress:   %6.2f ms  (%7.2f MPix/s)\n", time / float(iterations), throughput);
    if(timingDetail != TIMING_DETAIL_NONE) {
        printf("Detailed Timings (sum for %i iterations):\n", iterations);
        printTimings(shared.m_pCuCompInstance);
        resetTimings(shared.m_pCuCompInstance);
        printf("\n");
    }

    printf("Encoded highpass size: %.2f kB\n\n", float(bitStream.size() * sizeof(uint)) / 1024.0f);
    //encodePrintBitCounts();
    //printf("\n");

    cudaSafeCall(cudaMemset(dpImage, 0, elemCountTotal * sizeof(short)));

    // decompressHeightfieldOneLevel modifies the lowpass image, so make a backup first
    short* dpLowpassBak = 0;
    cudaSafeCall(cudaMalloc(&dpLowpassBak, elemCountPerBlock * sizeof(short)));
    cudaSafeCall(cudaMemcpy(dpLowpassBak, dpLowpass, elemCountPerBlock * sizeof(short), cudaMemcpyDeviceToDevice));

    cudaSafeCall(cudaHostRegister(bitStream.data(), bitStream.size() * sizeof(uint), cudaHostRegisterDefault));

    cudaProfilerStart();
    cudaSafeCall(cudaEventRecord(start));

    for(uint i = 0; i < iterations; i++) {
        cudaSafeCall(cudaMemcpy(dpLowpass, dpLowpassBak, elemCountPerBlock * sizeof(short), cudaMemcpyDeviceToDevice));
        decompressHeightfieldOneLevel(shared, res, dpImage, width, height, dpLowpass, 0, bitStream);
    }

    cudaSafeCall(cudaEventRecord(end));
    cudaProfilerStop();

    cudaSafeCall(cudaHostUnregister(bitStream.data()));

    cudaSafeCall(cudaFree(dpLowpassBak));

    cudaSafeCall(cudaEventSynchronize(end));
    cudaSafeCall(cudaEventElapsedTime(&time, start, end));
    throughput = float(iterations * elemCountPerBlock * blockCount) * 1000.0f / (time * 1024.0f * 1024.0f);
    printf("Decompress: %6.2f ms  (%7.2f MPix/s)\n", time / float(iterations), throughput);
    if(timingDetail != TIMING_DETAIL_NONE) {
        printf("Detailed Timings (sum for %i iterations):\n", iterations);
        printTimings(shared.m_pCuCompInstance);
        resetTimings(shared.m_pCuCompInstance);
        printf("\n");
    }

    memset(field.data(), 0, elemCountTotal * sizeof(short));
    cudaSafeCall(cudaMemcpy(field.data(), dpImage, elemCountTotal * sizeof(short), cudaMemcpyDeviceToHost));

    writeShortRaw("out.raw", elemCountTotal, field.data());

    res.destroy();
    shared.destroy();

    cudaSafeCall(cudaFree(dpLowpass));
    cudaSafeCall(cudaFree(dpImage));

    //printf("Highpass size: %u\n", bitStream.getRawSizeBytes());
    printf("%s\n", compareShortRaws(filenameOrig, "out.raw", elemCountTotal) ? "EQUAL" : "NOT EQUAL" );
    printf("\n\n");

    return 0;
}


struct Stats
{
    float Range;
    float MaxE;
    float RMSE;
    float PSNR;
    float SNR;
};

int benchmarkVolumeFloat(
    const std::vector<std::string>& filenamesOrig,
    uint width, uint height, uint depth,
    uint numLevels, float quantStep,
    uint iterations, ETimingDetail timingDetail,
    const std::string& filenameOut, const std::string& filenameComp)
{
    const bool doRLEOnlyOnLvl0 = true;

    const uint elemCountTotal = width * height * depth;
    const size_t channelCount = filenamesOrig.size();

    // read files
    std::vector<std::vector<float>> dataBak(filenamesOrig.size());
    std::vector<std::vector<float>> data(filenamesOrig.size());
    for(size_t c = 0; c < channelCount; c++) {
        if(c == 0) printf("Volume ");
        else       printf("       ");
        printf("\"%s\":\n", filenamesOrig[c].c_str());

        data[c].resize(elemCountTotal);
        if(!readFloatRaw(filenamesOrig[c], elemCountTotal, data[c].data())) {
            printf("Failed opening file %s\n", filenamesOrig[c].c_str());
            return -1;
        }
        dataBak[c] = data[c];
    }

    // allocate GPU arrays and upload data
    std::vector<float*> dpImages(channelCount);
    for(size_t c = 0; c < channelCount; c++) {
        cudaSafeCall(cudaMalloc(&dpImages[c], elemCountTotal * sizeof(float)));
        cudaSafeCall(cudaMemcpy(dpImages[c], data[c].data(), elemCountTotal * sizeof(float), cudaMemcpyHostToDevice));
    }

    cudaEvent_t start, end;
    cudaSafeCall(cudaEventCreate(&start));
    cudaSafeCall(cudaEventCreate(&end));
    float time = 0.0f;

    uint huffmanBits = 0;
    GPUResources::Config config = CompressVolumeResources::getRequiredResources(width, height, depth, (uint)channelCount, huffmanBits);
    GPUResources shared;
    shared.create(config);
    CompressVolumeResources res;
    res.create(shared.getConfig());
    setTimingDetail(shared.m_pCuCompInstance, timingDetail);

    std::vector<std::vector<uint>> bitStreams(channelCount);

    //cudaProfilerStart();
    cudaSafeCall(cudaEventRecord(start));

    for(uint i = 0; i < iterations; i++) {
        for(size_t c = 0; c < channelCount; c++) {
            compressVolumeFloat(shared, res, dpImages[c], width, height, depth, numLevels, bitStreams[c], quantStep, doRLEOnlyOnLvl0);
        }
    }

    cudaSafeCall(cudaEventRecord(end));
    //cudaProfilerStop();

    cudaSafeCall(cudaEventSynchronize(end));
    cudaSafeCall(cudaEventElapsedTime(&time, start, end));
    float throughput = float(iterations * elemCountTotal) * 1000.0f / (time * 1024.0f * 1024.0f);
    printf("Compress:   %6.2f ms  (%7.2f MPix/s  %7.2f Mfloat/s)\n", time / float(iterations), throughput, throughput * float(channelCount));
    if(timingDetail != TIMING_DETAIL_NONE) {
        printf("Detailed Timings (sum for %i iterations):\n", iterations);
        printTimings(shared.m_pCuCompInstance);
        resetTimings(shared.m_pCuCompInstance);
        printf("\n");
    }

    //encodePrintBitCounts();
    //printf("\n");

    //printf("Press Enter to continue...\n");
    //getchar();

    if(!filenameComp.empty()) {
        FILE* fileComp = 0;
        fopen_s(&fileComp, filenameComp.c_str(), "wb");
        if(!fileComp) {
            printf("Failed to open file %s for writing.\n", filenameComp.c_str());
        } else {
            fwrite(bitStreams.front().data(), sizeof(uint), bitStreams.front().size(), fileComp);
            fclose(fileComp);
        }
    }

    for(size_t c = 0; c < channelCount; c++) {
        cudaSafeCall(cudaMemset(dpImages[c], 0, elemCountTotal * sizeof(float)));
    }

    for(size_t c = 0; c < channelCount; c++) {
        cudaSafeCall(cudaHostRegister(bitStreams[c].data(), bitStreams[c].size() * sizeof(uint), cudaHostRegisterDefault));
    }

    cudaProfilerStart();
    cudaSafeCall(cudaEventRecord(start));

    if(channelCount == 1) {
        for(uint i = 0; i < iterations; i++) {
            decompressVolumeFloat(shared, res, dpImages.front(), width, height, depth, numLevels, bitStreams.front(), quantStep, doRLEOnlyOnLvl0);
            //decompressVolumeFloatLowpass(dpImages.front(), width, height, depth, numLevels, bitStreams.front(), quantStep);
        }
    } else {
        std::vector<VolumeChannel> channels(channelCount);
        for(size_t c = 0; c < channelCount; c++) {
            channels[c].dpImage = dpImages[c];
            channels[c].pBits = bitStreams[c].data();
            channels[c].bitCount = uint(bitStreams[c].size() * sizeof(uint) * 8);
            channels[c].quantizationStepLevel0 = quantStep;
        }
        for(uint i = 0; i < iterations; i++) {
            decompressVolumeFloatMultiChannel(shared, res, channels.data(), (uint)channels.size(), width, height, depth, numLevels, doRLEOnlyOnLvl0);
        }
    }

    cudaSafeCall(cudaEventRecord(end));
    cudaProfilerStop();

    for(size_t c = 0; c < channelCount; c++) {
        cudaSafeCall(cudaHostUnregister(bitStreams[c].data()));
    }

    cudaSafeCall(cudaEventSynchronize(end));
    cudaSafeCall(cudaEventElapsedTime(&time, start, end));
    throughput = float(iterations * elemCountTotal) * 1000.0f / (time * 1024.0f * 1024.0f);
    printf("Decompress: %6.2f ms  (%7.2f MPix/s  %7.2f Mfloat/s)\n", time / float(iterations), throughput, throughput * float(channelCount));
    if(timingDetail != TIMING_DETAIL_NONE) {
        printf("Detailed Timings (sum for %i iterations):\n", iterations);
        printTimings(shared.m_pCuCompInstance);
        resetTimings(shared.m_pCuCompInstance);
        printf("\n");
    }

    for(size_t c = 0; c < channelCount; c++) {
        memset(data[c].data(), 0, elemCountTotal * sizeof(float));
        cudaSafeCall(cudaMemcpy(data[c].data(), dpImages[c], elemCountTotal * sizeof(float), cudaMemcpyDeviceToHost));
    }

    if(!filenameOut.empty()) {
        writeFloatRaw(filenameOut, elemCountTotal, data.front().data());
    }

    res.destroy();
    shared.destroy();


    for(size_t c = 0; c < channelCount; c++) {
        cudaSafeCall(cudaFree(dpImages[c]));
    }

    uint compressedSizeBytes = 0;
    for(size_t c = 0; c < channelCount; c++) {
        compressedSizeBytes += uint(bitStreams[c].size()) * sizeof(uint);
    }
    float compressionFactor = sizeof(float) * float(elemCountTotal) * float(channelCount) / float(compressedSizeBytes);
    printf("Compressed size: %u B  (%.2f : 1)\n", compressedSizeBytes, compressionFactor);

    std::vector<Stats> stats(channelCount);
    for(size_t c = 0; c < channelCount; c++) {
        computeStatsFloatArrays(dataBak[c].data(), data[c].data(), elemCountTotal, &stats[c].Range, &stats[c].MaxE, &stats[c].RMSE, &stats[c].PSNR, &stats[c].SNR);
        printf("C%u  Range: %.3f   MaxE: %.4f   RMSE: %.4f   PSNR: %.2f   SNR: %.2f\n", (uint)c, stats[c].Range, stats[c].MaxE, stats[c].RMSE, stats[c].PSNR, stats[c].SNR);
    }
    printf("\n\n");

    std::ofstream info("info.txt", std::ios_base::app | std::ios_base::ate);
    info << filenameOut << std::endl;
    info << "Quantization step: " << quantStep << std::endl;
    info << "Compressed size: " << compressedSizeBytes << " B  (" << compressionFactor << " : 1)" << std::endl;
    //info << "Range: " << range << "\nRMSE: " << rmse << "\nPSNR: " << psnr << "\nSNR: " << snr << "\n" << std::endl;
    info.close();

    return 0;
}

int benchmarkVolumeFloat(
    const std::string& filenameOrig,
    uint width, uint height, uint depth,
    uint numLevels, float quantStep,
    uint iterations, ETimingDetail timingDetail,
    const std::string& filenameOut, const std::string& filenameComp)
{
    std::vector<std::string> filenamesOrig(1, filenameOrig);
    return benchmarkVolumeFloat(filenamesOrig, width, height, depth, numLevels, quantStep, iterations, timingDetail, filenameOut, filenameComp);
}

int benchmarkVolumeFloatQuantFirst(
    const std::vector<std::string>& filenamesOrig,
    uint width, uint height, uint depth,
    uint numLevels, float quantStep,
    uint iterations, ETimingDetail timingDetail,
    const std::string& filenameOut)
{
    const bool doRLEOnlyOnLvl0 = true;

    const uint elemCountTotal = width * height * depth;
    const size_t channelCount = filenamesOrig.size();

    // read files
    std::vector<std::vector<float>> data(filenamesOrig.size());
    std::vector<std::vector<float>> dataBak(filenamesOrig.size());
    for(size_t c = 0; c < channelCount; c++) {
        data[c].resize(elemCountTotal);
        if(!readFloatRaw(filenamesOrig[c], elemCountTotal, data[c].data())) {
            printf("Failed opening file %s\n", filenamesOrig[c].c_str());
            return -1;
        }
        dataBak[c] = data[c];
    }

    // allocate GPU arrays and upload data
    std::vector<float*> dpImages(channelCount);
    for(size_t c = 0; c < channelCount; c++) {
        cudaSafeCall(cudaMalloc(&dpImages[c], elemCountTotal * sizeof(float)));
        cudaSafeCall(cudaMemcpy(dpImages[c], data[c].data(), elemCountTotal * sizeof(float), cudaMemcpyHostToDevice));
    }


    cudaEvent_t start, end;
    cudaSafeCall(cudaEventCreate(&start));
    cudaSafeCall(cudaEventCreate(&end));
    float time = 0.0f;

    uint huffmanBits = 0;
    GPUResources::Config config = CompressVolumeResources::getRequiredResources(width, height, depth, (uint)channelCount, huffmanBits);
    GPUResources shared;
    shared.create(config);
    CompressVolumeResources res;
    res.create(shared.getConfig());
    setTimingDetail(shared.m_pCuCompInstance, timingDetail);

    std::vector<std::vector<uint>> bitStreams(channelCount);

    //cudaProfilerStart();
    cudaSafeCall(cudaEventRecord(start));

    for(uint i = 0; i < iterations; i++) {
        for(size_t c = 0; c < channelCount; c++) {
            compressVolumeFloatQuantFirst(shared, res, dpImages[c], width, height, depth, numLevels, bitStreams[c], quantStep, doRLEOnlyOnLvl0);
        }
    }

    cudaSafeCall(cudaEventRecord(end));
    //cudaProfilerStop();

    cudaSafeCall(cudaEventSynchronize(end));
    cudaSafeCall(cudaEventElapsedTime(&time, start, end));
    float throughput = float(iterations * elemCountTotal) * 1000.0f / (time * 1024.0f * 1024.0f);
    printf("Compress:   %6.2f ms  (%7.2f MPix/s  %7.2f Mfloat/s)\n", time / float(iterations), throughput, throughput * float(channelCount));
    if(timingDetail != TIMING_DETAIL_NONE) {
        printf("Detailed Timings (sum for %i iterations):\n", iterations);
        printTimings(shared.m_pCuCompInstance);
        resetTimings(shared.m_pCuCompInstance);
        printf("\n");
    }

    //encodePrintBitCounts();
    //printf("\n");

    //printf("Press Enter to continue...\n");
    //getchar();

    for(size_t c = 0; c < channelCount; c++) {
        cudaSafeCall(cudaMemset(dpImages[c], 0, elemCountTotal * sizeof(float)));

        cudaSafeCall(cudaHostRegister(bitStreams[c].data(), bitStreams[c].size() * sizeof(uint), cudaHostRegisterDefault));
    }

    cudaProfilerStart();
    cudaSafeCall(cudaEventRecord(start));

    for(uint i = 0; i < iterations; i++) {
        std::vector<VolumeChannel> channels(channelCount);
        for(size_t c = 0; c < channelCount; c++) {
            channels[c].dpImage = dpImages[c];
            channels[c].pBits = bitStreams[c].data();
            channels[c].bitCount = uint(bitStreams[c].size() * sizeof(uint) * 8);
            channels[c].quantizationStepLevel0 = quantStep;
        }
        if(channelCount == 1) {
            decompressVolumeFloatQuantFirst(shared, res, dpImages.front(), width, height, depth, numLevels, bitStreams.front(), quantStep, doRLEOnlyOnLvl0);
        } else {
            decompressVolumeFloatQuantFirstMultiChannel(shared, res, channels.data(), (uint)channels.size(), width, height, depth, numLevels, doRLEOnlyOnLvl0);
        }
    }

    cudaSafeCall(cudaEventRecord(end));
    cudaProfilerStop();

    for(size_t c = 0; c < channelCount; c++) {
        cudaSafeCall(cudaHostUnregister(bitStreams[c].data()));
    }

    cudaSafeCall(cudaEventSynchronize(end));
    cudaSafeCall(cudaEventElapsedTime(&time, start, end));
    throughput = float(iterations * elemCountTotal) * 1000.0f / (time * 1024.0f * 1024.0f);
    printf("Decompress: %6.2f ms  (%7.2f MPix/s  %7.2f Mfloat/s)\n", time / float(iterations), throughput, throughput * float(channelCount));
    if(timingDetail != TIMING_DETAIL_NONE) {
        printf("Detailed Timings (sum for %i iterations):\n", iterations);
        printTimings(shared.m_pCuCompInstance);
        resetTimings(shared.m_pCuCompInstance);
        printf("\n");
    }

    for(size_t c = 0; c < channelCount; c++) {
        memset(data[c].data(), 0, elemCountTotal * sizeof(float));
        cudaSafeCall(cudaMemcpy(data[c].data(), dpImages[c], elemCountTotal * sizeof(float), cudaMemcpyDeviceToHost));
    }

    if(!filenameOut.empty()) {
        writeFloatRaw(filenameOut, elemCountTotal, data.front().data());
    }


    res.destroy();
    shared.destroy();

    for(size_t c = 0; c < channelCount; c++) {
        cudaSafeCall(cudaFree(dpImages[c]));
    }


    uint compressedSizeBytes = 0;
    for(size_t c = 0; c < channelCount; c++) {
        compressedSizeBytes += uint(bitStreams[c].size()) * sizeof(uint);
    }
    float compressionFactor = sizeof(float) * float(elemCountTotal) * float(channelCount) / float(compressedSizeBytes);
    printf("Compressed size: %u B  (%.2f : 1)\n", compressedSizeBytes, compressionFactor);

    std::vector<Stats> stats(channelCount);
    for(size_t c = 0; c < channelCount; c++) {
        computeStatsFloatArrays(dataBak[c].data(), data[c].data(), elemCountTotal, &stats[c].Range, &stats[c].MaxE, &stats[c].RMSE, &stats[c].PSNR, &stats[c].SNR);
        printf("C%u  Range: %.3f   MaxE: %.4f   RMSE: %.4f   PSNR: %.2f   SNR: %.2f\n", (uint)c, stats[c].Range, stats[c].MaxE, stats[c].RMSE, stats[c].PSNR, stats[c].SNR);
    }
    printf("\n\n");

    std::ofstream info("info.txt", std::ios_base::app | std::ios_base::ate);
    info << filenameOut << std::endl;
    info << "Quantization step: " << quantStep << std::endl;
    info << "Compressed size: " << compressedSizeBytes << " B  (" << compressionFactor << " : 1)" << std::endl;
    //info << "Range: " << range << "\nRMSE: " << rmse << "\nPSNR: " << psnr << "\nSNR: " << snr << "\n" << std::endl;
    info.close();

    return 0;
}

int benchmarkVolumeFloatQuantFirst(
    const std::string& filenameOrig,
    uint width, uint height, uint depth,
    uint numLevels, float quantStep,
    uint iterations, ETimingDetail timingDetail,
    const std::string& filenameOut)
{
    std::vector<std::string> filenamesOrig(1, filenameOrig);
    return benchmarkVolumeFloatQuantFirst(filenamesOrig, width, height, depth, numLevels, quantStep, iterations, timingDetail, filenameOut);
}



std::vector<int> getCudaDevices()
{
    std::vector<int> usableDevices;

    // loop over all cuda devices, collect those that we can use
    int deviceCount = 0;
    cudaSafeCall(cudaGetDeviceCount(&deviceCount));

    for(int device = 0; device < deviceCount; device++)
    {
        cudaDeviceProp prop;
        cudaSafeCall(cudaGetDeviceProperties(&prop, device));

        bool deviceOK = (prop.major >= 2);
        if(deviceOK)
        {
            printf("Found usable CUDA device %i: %s, compute %i.%i\n", device, prop.name, prop.major, prop.minor);
            usableDevices.push_back(device);
        }
    }

    return usableDevices;
}

struct TestVolumeFloatArgs
{
    TestVolumeFloatArgs()
        : step(0)
        , device(-1)
        , pData(nullptr)
        , width(0), height(0), depth(0), numLevels(0), quantStep(0.0f), iterations(0) {}

    TestVolumeFloatArgs(int device, float* pData, uint width, uint height, uint depth, uint numLevels, float quantStep, uint iterations, ETimingDetail timingDetail)
        : step(0)
        , device(device)
        , pData(pData)
        , width(width), height(height), depth(depth), numLevels(numLevels), quantStep(quantStep)
        , iterations(iterations)
        , timingDetail(timingDetail) {}

    void set(int device, float* pData, uint width, uint height, uint depth, uint numLevels, float quantStep, bool doRLEOnlyOnLvl0, uint iterations, ETimingDetail timingDetail)
    {
        this->device = device;
        this->pData = pData;
        this->width = width;
        this->height = height;
        this->depth = depth;
        this->numLevels = numLevels;
        this->quantStep = quantStep;
        this->doRLEOnlyOnLvl0 = doRLEOnlyOnLvl0;
        this->iterations = iterations;
        this->timingDetail = timingDetail;
    }

    int step;
    tthread::mutex mutex;
    tthread::condition_variable stepDone;

    int device;

    float* pData;
    uint width;
    uint height;
    uint depth;
    uint numLevels;
    float quantStep;
    bool doRLEOnlyOnLvl0;
    uint iterations;
    ETimingDetail timingDetail;
};

void benchmarkVolumeFloatMultiGPUThreadFunc(void* pArgsRaw)
{
    TestVolumeFloatArgs* pArgs = (TestVolumeFloatArgs*)pArgsRaw;

    const uint elemCountTotal = pArgs->width * pArgs->height * pArgs->depth;

    cudaSafeCall(cudaSetDevice(pArgs->device));

    // allocate GPU arrays and upload data
    float* dpImage = 0;
    cudaSafeCall(cudaMalloc(&dpImage, elemCountTotal * sizeof(float)));
    cudaSafeCall(cudaMemcpy(dpImage, pArgs->pData, elemCountTotal * sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t start, end;
    cudaSafeCall(cudaEventCreate(&start));
    cudaSafeCall(cudaEventCreate(&end));
    float time = 0.0f;

    uint huffmanBits = 0;
    GPUResources::Config config = CompressVolumeResources::getRequiredResources(pArgs->width, pArgs->height, pArgs->depth, 1, huffmanBits);
    GPUResources shared;
    shared.create(config);
    CompressVolumeResources res;
    res.create(shared.getConfig());
    setTimingDetail(shared.m_pCuCompInstance, pArgs->timingDetail);

    // signal that initialization is done and encoding is about to start
    { tthread::lock_guard<tthread::mutex> guard(pArgs->mutex);
        pArgs->step++;
    }
    pArgs->stepDone.notify_all();

    std::vector<uint> bitStream;

    cudaProfilerStart();
    cudaSafeCall(cudaEventRecord(start));

    for(uint i = 0; i < pArgs->iterations; i++) {
        compressVolumeFloat(shared, res, dpImage, pArgs->width, pArgs->height, pArgs->depth, pArgs->numLevels, bitStream, pArgs->quantStep, pArgs->doRLEOnlyOnLvl0);
    }

    cudaSafeCall(cudaEventRecord(end));
    cudaProfilerStop();

    // signal that encoding is done
    { tthread::lock_guard<tthread::mutex> guard(pArgs->mutex);
        pArgs->step++;
    }
    pArgs->stepDone.notify_all();

    cudaSafeCall(cudaEventSynchronize(end));
    cudaSafeCall(cudaEventElapsedTime(&time, start, end));
    float throughput = float(pArgs->iterations * elemCountTotal) * 1000.0f / (time * 1024.0f * 1024.0f);
    printf("Device %i: Compress:   %6.2f ms  (%7.2f MPix/s)\n", pArgs->device, time / float(pArgs->iterations), throughput);
    if(pArgs->timingDetail != TIMING_DETAIL_NONE) {
        printf("Detailed Timings (sum for %i iterations):\n", pArgs->iterations);
        printTimings(shared.m_pCuCompInstance);
        resetTimings(shared.m_pCuCompInstance);
        printf("\n");
    }

    cudaSafeCall(cudaMemset(dpImage, 0, elemCountTotal * sizeof(float)));

    cudaSafeCall(cudaHostRegister(bitStream.data(), bitStream.size() * sizeof(uint), cudaHostRegisterDefault));

    // signal that decoding is about to start
    { tthread::lock_guard<tthread::mutex> guard(pArgs->mutex);
        pArgs->step++;
    }
    pArgs->stepDone.notify_all();

    cudaProfilerStart();
    cudaSafeCall(cudaEventRecord(start));

    for(uint i = 0; i < pArgs->iterations; i++) {
        decompressVolumeFloat(shared, res, dpImage, pArgs->width, pArgs->height, pArgs->depth, pArgs->numLevels, bitStream, pArgs->quantStep, pArgs->doRLEOnlyOnLvl0);
    }

    cudaSafeCall(cudaEventRecord(end));
    cudaProfilerStop();

    // signal that decoding is done
    { tthread::lock_guard<tthread::mutex> guard(pArgs->mutex);
        pArgs->step++;
    }
    pArgs->stepDone.notify_all();

    cudaSafeCall(cudaHostUnregister(bitStream.data()));

    cudaSafeCall(cudaEventSynchronize(end));
    cudaSafeCall(cudaEventElapsedTime(&time, start, end));
    throughput = float(pArgs->iterations * elemCountTotal) * 1000.0f / (time * 1024.0f * 1024.0f);
    printf("Device %i: Decompress: %6.2f ms  (%7.2f MPix/s)\n", pArgs->device, time / float(pArgs->iterations), throughput);
    if(pArgs->timingDetail != TIMING_DETAIL_NONE) {
        printf("Detailed Timings (sum for %i iterations):\n", pArgs->iterations);
        printTimings(shared.m_pCuCompInstance);
        resetTimings(shared.m_pCuCompInstance);
        printf("\n");
    }

    res.destroy();
    shared.destroy();

    cudaSafeCall(cudaFree(dpImage));
}

int benchmarkVolumeFloatMultiGPU(const std::string& filenameOrig, uint width, uint height, uint depth, uint numLevels, float quantStep, uint iterations, ETimingDetail timingDetail)
{
    const bool doRLEOnlyOnLvl0 = true;

    const uint elemCountTotal = width * height * depth;

    // read file
    std::vector<float> data(elemCountTotal);
    if(!readFloatRaw(filenameOrig, elemCountTotal, data.data())) {
        printf("Failed opening file %s\n", filenameOrig.c_str());
        return -1;
    }

    std::vector<int> devices = getCudaDevices();

    TestVolumeFloatArgs* args = new TestVolumeFloatArgs[devices.size()];
    //std::vector<TestVolumeFloatArgs> args;
    for(size_t i = 0; i < devices.size(); i++)
    {
        args[i].set(devices[i], data.data(), width, height, depth, numLevels, quantStep, doRLEOnlyOnLvl0, iterations, timingDetail);
    }

    // start threads
    std::vector<tthread::thread*> threads;
    for(size_t i = 0; i < devices.size(); i++)
    {
        threads.push_back(new tthread::thread(benchmarkVolumeFloatMultiGPUThreadFunc, &args[i]));
    }

    // wait for threads to finish their steps one by one
    int stepMax = 4;
    for(int step = 1; step <= stepMax; step++)
    {
        for(size_t i = 0; i < threads.size(); i++)
        {
            tthread::lock_guard<tthread::mutex> guard(args[i].mutex);
            while(args[i].step < stepMax)
            {
                args[i].stepDone.wait(args[i].mutex);
            }
        }
    }

    // wait for threads to shut down
    for(size_t i = 0; i < threads.size(); i++)
    {
        threads[i]->join();
        delete threads[i];
    }

    delete[] args;

    printf("DONE.\n\n");

    return 0;
}


struct StatsEntry
{
    StatsEntry(float quantStep, float entropy) : quantStep(quantStep), entropy(entropy) {}

    float quantStep;
    float entropy;
    std::vector<float> times;
};

bool writeToCSV(const std::vector<std::string>& headings, const std::vector<StatsEntry>& stats, const std::string& filename)
{
    std::ofstream file(filename);
    if(!file.good()) {
        return false;
    }

    // headings
    file << "QuantStep;Entropy";
    for(const std::string& heading : headings) {
        file << ";" << heading;
    }
    file << "\n";

    // data
    for(const StatsEntry& entry : stats) {
        file << entry.quantStep;
        file << ";" << entry.entropy;

        for(float time : entry.times) {
            file << ";" << time;
        }
        file << "\n";
    }

    return true;
}

std::vector<StatsEntry> interpolateEvenSpaced(const std::vector<StatsEntry>& stats, float step, uint count)
{
    std::vector<StatsEntry> result;

    size_t indexSrc = stats.size() - 1;
    for(uint s = 0; s <= count; s++) {
        float entropy = float(s) * step;

        // find src index
        while(indexSrc - 1 > 0 && stats[indexSrc - 1].entropy < entropy) {
            indexSrc--;
        }

        // compute weight
        const StatsEntry& left  = stats[indexSrc];
        const StatsEntry& right = stats[indexSrc - 1];
        float entropyLeft  = left.entropy;
        float entropyRight = right.entropy;
        float t = (entropy - entropyLeft) / (entropyRight - entropyLeft);

        // interpolate
        float quant = (1.0f-t) * left.quantStep + t * right.quantStep;
        result.emplace_back(quant, entropy);
        StatsEntry& entry = result.back();
        entry.times.resize(left.times.size());
        for(size_t i = 0; i < left.times.size(); i++) {
            entry.times[i] = (1.0f-t) * left.times[i] + t * right.times[i];
        }
    }

    return result;
}

int benchmarkCoding()
{
    const std::string& filenameOrig("data/UtahTexSample1mGray.raw");
    const uint width = 2048;
    const uint height = 2048;

    const float quantMin =  0.2f;
    const float quantMax = 20.0f;
    const float quantFac =  1.05f;

    const uint levelCount = 2;
    const uint blockCountPerDim = 4;

    const uint iterations = 100;

    const uint elemCount = width * height;
    const uint blockCountTotal = blockCountPerDim * blockCountPerDim;
    const uint blockCountEncode = blockCountTotal - 1; // skip lowpass band
    const uint blockWidth  = width  / blockCountPerDim;
    const uint blockHeight = height / blockCountPerDim;
    const uint elemCountPerBlock = blockWidth * blockHeight;

    // read file
    std::vector<byte> image(elemCount);
    if(!readByteRaw(filenameOrig, elemCount, image.data())) {
        printf("Failed opening file %s\n", filenameOrig.c_str());
        return -1;
    }

    // expand to float
    std::vector<float> imageFloat(elemCount);
    for(uint i = 0; i < elemCount; i++) {
        imageFloat[i] = float(image[i]);
    }

    // init cudaCompress
    Instance* pInstance = createInstance(-1, blockCountEncode, elemCountPerBlock);
    setTimingDetail(pInstance, TIMING_DETAIL_LOW);

    // alloc GPU buffers
    float* dpData = nullptr;
    float* dpTemp = nullptr;
    cudaSafeCall(cudaMalloc(&dpData, elemCount * sizeof(float)));
    cudaSafeCall(cudaMalloc(&dpTemp, elemCount * sizeof(float)));
    std::vector<Symbol16*> dpSymbolStreams(blockCountEncode);
    for(uint i = 0; i < blockCountEncode; i++) {
        cudaSafeCall(cudaMalloc(&dpSymbolStreams[i], elemCountPerBlock * sizeof(Symbol16)));
    }

    // upload image
    cudaSafeCall(cudaMemcpy(dpData, imageFloat.data(), elemCount * sizeof(float), cudaMemcpyHostToDevice));


    // perform DWT
    for(uint level = 0; level < levelCount; level++) {
        uint widthLevel  = width  >> level;
        uint heightLevel = height >> level;
        util::dwtFloat2DForward(dpData, dpTemp, dpData, widthLevel, heightLevel, 1, width, width);
    }


    std::vector<std::string> headings;
    std::vector<StatsEntry> stats;

    for(float quant = quantMin; quant <= quantMax; quant *= quantFac) {
        // quantize highpass blocks
        for(uint block = 1; block < blockCountTotal; block++) {
            uint x = block % blockCountPerDim;
            uint y = (block / blockCountPerDim) % blockCountPerDim;
            uint z = block / (blockCountPerDim * blockCountPerDim);
            uint offset = x * blockWidth + y * blockHeight * width;

            uint stream = block - 1;
            util::quantizeToSymbols2D(dpSymbolStreams[stream], dpData + offset, blockWidth, blockHeight, quant, width);
        }


        // download quantized data and compute entropy
        std::vector<Symbol16> symbols(elemCountPerBlock);
        double entropySum = 0.0;
        for(Symbol16* dpSymbols : dpSymbolStreams) {
            cudaSafeCall(cudaMemcpy(symbols.data(), dpSymbols, elemCountPerBlock * sizeof(Symbol16), cudaMemcpyDeviceToHost));
            entropySum += computeEntropy(symbols.data(), symbols.size());
        }

        stats.emplace_back(quant, float(entropySum / blockCountEncode));


        bool recordNames = headings.empty();
        std::vector<std::string> namesTemp;


        // encode RL+Huff
        BitStream bitStreamRLHuff;
        for(uint i = 0; i < iterations; i++) {
            bitStreamRLHuff.setBitSize(0);
            encodeRLHuff(pInstance, bitStreamRLHuff, dpSymbolStreams.data(), blockCountEncode, elemCountPerBlock);
        }
        getTimings(pInstance, recordNames ? headings : namesTemp, stats.back().times);
        resetTimings(pInstance);

        // decode RL+Huff
        for(uint i = 0; i < iterations; i++) {
            bitStreamRLHuff.setBitPosition(0);
            decodeRLHuff(pInstance, bitStreamRLHuff, dpSymbolStreams.data(), blockCountEncode, elemCountPerBlock);
        }
        getTimings(pInstance, recordNames ? headings : namesTemp, stats.back().times);
        resetTimings(pInstance);


        // encode Huff
        BitStream bitStreamHuff;
        for(uint i = 0; i < iterations; i++) {
            bitStreamHuff.setBitSize(0);
            encodeHuff(pInstance, bitStreamHuff, dpSymbolStreams.data(), blockCountEncode, elemCountPerBlock);
        }
        getTimings(pInstance, recordNames ? headings : namesTemp, stats.back().times);
        resetTimings(pInstance);

        // decode Huff
        for(uint i = 0; i < iterations; i++) {
            bitStreamHuff.setBitPosition(0);
            decodeHuff(pInstance, bitStreamHuff, dpSymbolStreams.data(), blockCountEncode, elemCountPerBlock);
        }
        getTimings(pInstance, recordNames ? headings : namesTemp, stats.back().times);
        resetTimings(pInstance);


        printf(".");
    }

    for(StatsEntry& entry : stats) {
        for(float& time : entry.times) {
            time /= float(iterations);
        }
    }


    // release resources
    for(uint i = 0; i < blockCountEncode; i++) {
        cudaSafeCall(cudaFree(dpSymbolStreams[i]));
    }
    cudaSafeCall(cudaFree(dpTemp));
    cudaSafeCall(cudaFree(dpData));

    destroyInstance(pInstance);


    // write out results
    writeToCSV(headings, stats, "timings.csv");

    std::vector<StatsEntry> statsInterpol = interpolateEvenSpaced(stats, 0.1f, 50);
    writeToCSV(headings, statsInterpol, "timings_int.csv");


    return 0;
}


#pragma warning( pop )
