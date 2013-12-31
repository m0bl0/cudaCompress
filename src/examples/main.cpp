#include "global.h"

#include <sstream>
#include <string>
#include <string.h>

#include <cudaCompress/Timing.h>
using namespace cudaCompress;

#include "cudaUtil.h"

#include "examples.h"


int cutilDeviceInit(int argc, char** argv)
{
    int deviceCount;
    cudaSafeCall(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "CUTIL CUDA error: no devices supporting CUDA.\n");
        exit(-1);
    }
    int dev = 0;
    //cutGetCmdLineArgumenti(ARGC, (const char **) ARGV, "device", &dev);
    for(int i = 1; i < argc; i++) {
        const char* arg = "--device=";
        size_t len = strlen(arg);
        if(strncmp(argv[i], arg, len) == 0) {
            dev = atoi(argv[i] + len);
            break;
        }
    }
    if (dev < 0) 
        dev = 0;
    if (dev > deviceCount-1) {
        fprintf(stderr, "\n");
        fprintf(stderr, ">> %d CUDA capable GPU device(s) detected. <<\n", deviceCount);
        fprintf(stderr, ">> cutilDeviceInit (-device=%d) is not a valid GPU device. <<\n", dev);
        fprintf(stderr, "\n");
        return -dev;
    }
    cudaDeviceProp deviceProp;
    cudaSafeCall(cudaGetDeviceProperties(&deviceProp, dev));
    if (deviceProp.major < 1) {
        fprintf(stderr, "cutil error: GPU device does not support CUDA.\n");
        exit(-1);
    }
    if (deviceProp.major < 1) {
        fprintf(stderr, "cutil error: GPU device does not support CUDA.\n");
        exit(-1);
    }
    if (deviceProp.major < 2) {
        fprintf(stderr, "cutil error: GPU device must support at least compute 2.0 (Fermi).\n");
        exit(-1);
    }
    printf("> Using CUDA device [%d]: %s\n", dev, deviceProp.name);
    cudaSafeCall(cudaSetDevice(dev));

    return dev;
}

int main(int argc, char **argv)
{
    // enable run-time memory check for debug builds
    #if defined( _WIN32 ) && ( defined( DEBUG ) || defined( _DEBUG ) )
    _CrtSetDbgFlag( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );
    #endif


    // init cuda (cmdline-specified device)
    cutilDeviceInit(argc, argv);
    printf("\n");

    //return benchmarkCoding();


    // set granularity of timing information - higher levels come with some performance overhead
    ETimingDetail timingDetail = TIMING_DETAIL_NONE;

    uint iterations = 10;
    #ifdef _DEBUG
        iterations = 1;
    #endif

    // speed test: compress and decompress the finest level of a texture/heightfield
    benchmarkOneLevelImage("data\\UtahTexSample1m.raw", 2048, 2048, 1, 3.0f, iterations, timingDetail);
    benchmarkOneLevelHeightfield("data\\UtahGeoSample1m.raw", 2048, 2048, iterations, timingDetail);

    // speed test: compress and decompress floating-point volumes
    benchmarkVolumeFloat("data\\Iso_128_128_128_t4000_VelocityX.raw", 128, 128, 128, 2, 0.00136f, iterations, timingDetail);
    benchmarkVolumeFloat("data\\Iso_256_256_256_t4000_VelocityX.raw", 256, 256, 256, 2, 0.00136f, iterations, timingDetail);
    benchmarkVolumeFloatQuantFirst("data\\Iso_128_128_128_t4000_VelocityX.raw", 128, 128, 128, 2, 0.005f, iterations, timingDetail);
    benchmarkVolumeFloatQuantFirst("data\\Iso_256_256_256_t4000_VelocityX.raw", 256, 256, 256, 2, 0.005f, iterations, timingDetail);
    //benchmarkVolumeFloatMultiGPU("data\\Iso_256_256_256_t4000_VelocityX.raw", 256, 256, 256, 2, 0.005f, iterations, timingDetail);

    // speed test: multi-channel volume
    std::vector<std::string> filenames;
    filenames.push_back("data\\Iso_128_128_128_t4000_VelocityX.raw");
    filenames.push_back("data\\Iso_128_128_128_t4000_VelocityY.raw");
    filenames.push_back("data\\Iso_128_128_128_t4000_VelocityZ.raw");
    benchmarkVolumeFloat(filenames, 128, 128, 128, 2, 0.00136f, iterations, timingDetail);
    benchmarkVolumeFloatQuantFirst(filenames, 128, 128, 128, 2, 0.005f, iterations, timingDetail);



    // quality test: compress a texture using different quantization steps
    //               speed is not representative because the coarser levels are too small to saturate the GPU
    //printf("QUALITY TEST:\n\n");
    //std::string filename("data\\Utah_City3.raw");
    //int width = 2048, height = 2048;
    //float qStart = 2.0f;
    //float qEnd = 5.0f;

    //for(float q = qStart; q <= qEnd; q += 0.2f) {
    //    compressImageScalable(filename, width, height, 5, q);
    //}


    return 0;
}
