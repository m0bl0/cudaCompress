#include <cstdlib>
#include <fstream>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "EncoderTestSuite.h"

using namespace cudaCompress;


bool writeToCSV(const Stats& stats, const std::string& filename)
{
    std::ofstream file(filename);
    if(!file.good()) {
        return false;
    }

    // headings
    file << "Entropy;";
    // assume coders are the same for all entries
    const CoderSizes& coderSizes = stats.entries.begin()->coderSizes;
    // write out coder names three times - bit rate, relative overhead, absolute overhead
    for(int i = 0; i < 3; i++) {
        for(auto& coderSize : coderSizes) {
            file << getCoderName(coderSize.first) << ";";
        }
    }
    file << "\n";

    // data
    for(auto& entry : stats.entries) {
        float entropy = float(entry.entropySize / entry.symbolCount);
        file << entropy << ";";

        for(auto& coderSize : entry.coderSizes) {
            float bitrate = float(coderSize.second) / float(entry.symbolCount);
            // write out absolute bitrate
            file << bitrate << ";";
        }
        for(auto& coderSize : entry.coderSizes) {
            float bitrate = float(coderSize.second) / float(entry.symbolCount);
            // write out bitrate relative to entropy
            file << (bitrate / entropy) << ";";
        }
        for(auto& coderSize : entry.coderSizes) {
            float bitrate = float(coderSize.second) / float(entry.symbolCount);
            // write out bitrate minus entropy
            file << (bitrate - entropy) << ";";
        }
        file << "\n";
    }

    return true;
}


void runTests(const std::string& pattern, uint indexMin, uint indexMax, uint indexStep, const std::string& outputPrefix)
{
    float quantMin =  0.1f;
    float quantMax = 20.0f;
    float quantFac =  1.1f;

    bool write = true;

    Stats stats;
    stats = runEncoderTestSuite(pattern, indexMin, indexMax, indexStep, 1, 3, quantMin, quantMax, quantFac, true);
    if(write && !writeToCSV(stats, outputPrefix + "stats_13_blocked.csv")) {
        printf("Failed writing output file!\n");
    }
    stats = runEncoderTestSuite(pattern, indexMin, indexMax, indexStep, 1, 3, quantMin, quantMax, quantFac, false);
    if(write && !writeToCSV(stats, outputPrefix + "stats_13_sequential.csv")) {
        printf("Failed writing output file!\n");
    }
    //stats = runEncoderTestSuite(pattern, indexMin, indexMax, indexStep, 1, 2, quantMin, quantMax, quantFac, true);
    //if(write && !writeToCSV(stats, outputPrefix + "stats_12_blocked.csv")) {
    //    printf("Failed writing output file!\n");
    //}
    //stats = runEncoderTestSuite(pattern, indexMin, indexMax, indexStep, 1, 2, quantMin, quantMax, quantFac, false);
    //if(write && !writeToCSV(stats, outputPrefix + "stats_12_sequential.csv")) {
    //    printf("Failed writing output file!\n");
    //}
    //stats = runEncoderTestSuite(pattern, indexMin, indexMax, indexStep, 2, 3, quantMin, quantMax, quantFac, true);
    //if(write && !writeToCSV(stats, outputPrefix + "stats_23_blocked.csv")) {
    //    printf("Failed writing output file!\n");
    //}
    //stats = runEncoderTestSuite(pattern, indexMin, indexMax, indexStep, 2, 3, quantMin, quantMax, quantFac, false);
    //if(write && !writeToCSV(stats, outputPrefix + "stats_23_sequential.csv")) {
    //    printf("Failed writing output file!\n");
    //}
    //stats = runEncoderTestSuite(pattern, indexMin, indexMax, indexStep, 1, 1, quantMin, quantMax, quantFac, true);
    //if(write && !writeToCSV(stats, outputPrefix + "stats_1_blocked.csv")) {
    //    printf("Failed writing output file!\n");
    //}
    //stats = runEncoderTestSuite(pattern, indexMin, indexMax, indexStep, 1, 1, quantMin, quantMax, quantFac, false);
    //if(write && !writeToCSV(stats, outputPrefix + "stats_1_sequential.csv")) {
    //    printf("Failed writing output file!\n");
    //}
    //stats = runEncoderTestSuite(pattern, indexMin, indexMax, indexStep, 2, 2, quantMin, quantMax, quantFac, true);
    //if(write && !writeToCSV(stats, outputPrefix + "stats_2_blocked.csv")) {
    //    printf("Failed writing output file!\n");
    //}
    //stats = runEncoderTestSuite(pattern, indexMin, indexMax, indexStep, 2, 2, quantMin, quantMax, quantFac, false);
    //if(write && !writeToCSV(stats, outputPrefix + "stats_2_sequential.csv")) {
    //    printf("Failed writing output file!\n");
    //}
    //stats = runEncoderTestSuite(pattern, indexMin, indexMax, indexStep, 3, 3, quantMin, quantMax, quantFac, true);
    //if(write && !writeToCSV(stats, outputPrefix + "stats_3_blocked.csv")) {
    //    printf("Failed writing output file!\n");
    //}
    //stats = runEncoderTestSuite(pattern, indexMin, indexMax, indexStep, 3, 3, quantMin, quantMax, quantFac, false);
    //if(write && !writeToCSV(stats, outputPrefix + "stats_3_sequential.csv")) {
    //    printf("Failed writing output file!\n");
    //}
}

int main(int argc, char** argv)
{
    //BitStream bs;
    //std::vector<Symbol16> symbols;
    //std::vector<uint> probs;
    //symbols.push_back(1);
    //symbols.push_back(0);
    //symbols.push_back(2);
    //symbols.push_back(0);
    //symbols.push_back(0);
    //symbols.push_back(0);
    //symbols.push_back(0);
    //symbols.push_back(1);
    //symbols.push_back(1);
    //symbols.push_back(1);
    //probs.push_back(0);
    //probs.push_back(probs.back() + 5);
    //probs.push_back(probs.back() + 4);
    //probs.push_back(probs.back() + 1);
    //std::vector<uint> offsets;
    //arithmeticEncodeCPU(bs, symbols, probs, offsets, 128);

    runTests("data/kodim/kodim%02i.png", 1, 24, 1, "kodim_");
    //runTests("D:/testimg/new_rgb8_crop64/%02i.png", 1, 14, 1, "newtest_");
    //runTests("F:/Vorarlberg/tex12cm-raw/%02i.raw",  0, 99, 1, "vorarlberg_");

    return 0;
}
