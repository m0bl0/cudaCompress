#ifndef __ENCODER_TEST_SUITE_H__
#define __ENCODER_TEST_SUITE_H__


#include <cudaCompress/global.h>

#include <map>

#include "EncodeCPU.h"


namespace cudaCompress {


typedef std::map<ECoder, size_t> CoderSizes;

struct StatsEntry
{
    StatsEntry() : symbolCount(0), entropySize(0.0) {}

    size_t symbolCount;
    double entropySize;
    CoderSizes coderSizes;
};

struct Stats
{
    std::vector<StatsEntry> entries;
};


Stats runEncoderTestSuite(
    const std::string& filenamePattern,
    uint indexMin, uint indexMax, uint indexStep,
    uint dwtLevelFrom, uint dwtLevelTo,
    float quantStepMin, float quantStepMax, float quantStepFactor,
    bool blocked
);


}


#endif
