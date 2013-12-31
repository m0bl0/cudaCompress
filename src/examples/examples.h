#ifndef __TUM3D__EXAMPLES_H__
#define __TUM3D__EXAMPLES_H__


#include "global.h"

#include <string>
#include <vector>

#include <cudaCompress/Timing.h>


int compressImageScalable(const std::string& filenameOrig, uint width, uint height, uint levelCount, float quantStep);

int benchmarkOneLevelImage(const std::string& filenameOrig, uint width, uint height, uint tiles, float quantizationStep, uint iterations, cudaCompress::ETimingDetail timingDetail = cudaCompress::TIMING_DETAIL_NONE);

int benchmarkOneLevelHeightfield(const std::string& filenameOrig, uint width, uint height, uint iterations, cudaCompress::ETimingDetail timingDetail = cudaCompress::TIMING_DETAIL_NONE);

int benchmarkVolumeFloat(
    const std::vector<std::string>& filenamesOrig,
    uint width, uint height, uint depth,
    uint numLevels, float quantStep,
    uint iterations, cudaCompress::ETimingDetail timingDetail = cudaCompress::TIMING_DETAIL_NONE,
    const std::string& filenameOut = "out.raw", const std::string& filenameComp = "");

int benchmarkVolumeFloat(
    const std::string& filenameOrig,
    uint width, uint height, uint depth,
    uint numLevels, float quantStep,
    uint iterations, cudaCompress::ETimingDetail timingDetail = cudaCompress::TIMING_DETAIL_NONE,
    const std::string& filenameOut = "out.raw", const std::string& filenameComp = "");

int benchmarkVolumeFloatQuantFirst(
    const std::vector<std::string>& filenamesOrig,
    uint width, uint height, uint depth,
    uint numLevels, float quantStep,
    uint iterations, cudaCompress::ETimingDetail timingDetail = cudaCompress::TIMING_DETAIL_NONE,
    const std::string& filenameOut = "out.raw");

int benchmarkVolumeFloatQuantFirst(
    const std::string& filenameOrig,
    uint width, uint height, uint depth,
    uint numLevels, float quantStep,
    uint iterations, cudaCompress::ETimingDetail timingDetail = cudaCompress::TIMING_DETAIL_NONE,
    const std::string& filenameOut = "out.raw");

int benchmarkVolumeFloatMultiGPU(
    const std::string& filenameOrig,
    uint width, uint height, uint depth,
    uint numLevels, float quantStep,
    uint iterations, cudaCompress::ETimingDetail timingDetail = cudaCompress::TIMING_DETAIL_NONE);

int benchmarkCoding();


#endif
