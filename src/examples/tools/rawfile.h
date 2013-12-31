#ifndef __rawfile_h__
#define __rawfile_h__


#include "../global.h"

#include <string>


bool readByteRaw(const std::string& filename, uint elemCount, byte* pResult);
bool writeByteRaw(const std::string& filename, uint elemCount, const byte* pData);

bool readByteRawAsShort(const std::string& filename, uint elemCount, short* pResult);
bool writeByteRawFromShort(const std::string& filename, uint elemCount, const short* pData);

bool readByteRawAsFloat(const std::string& filename, uint elemCount, float* pResult);
bool writeByteRawFromFloat(const std::string& filename, uint elemCount, const float* pData);

bool compareByteRaws(const std::string& filename1, const std::string& filename2, uint elemCount);

float computePSNRByteRaws(const std::string& filename1, const std::string& filename2, uint elemCount);
bool  computePSNRByteRaws(const std::string& filename1, const std::string& filename2, uint elemCount, float* pResult, uint channelCount);


bool readShortRaw(const std::string& filename, uint elemCount, short* pResult);
bool writeShortRaw(const std::string& filename, uint elemCount, const short* pData);

bool compareShortRaws(const std::string& filename1, const std::string& filename2, uint elemCount);


bool readFloatRaw(const std::string& filename, uint elemCount, float* pResult);
bool writeFloatRaw(const std::string& filename, uint elemCount, const float* pData);

void computeStatsFloatArrays(const float* pData1, const float* pData2, uint elemCount, float* pRange, float* pMaxError, float* pRMSError, float* pPSNR, float* pSNR, uint channelCount = 1);
bool computeStatsFloatRaws(const std::string& filename1, const std::string& filename2, uint elemCount, float* pRange, float* pMaxError, float* pRMSError, float* pPSNR, float* pSNR, uint channelCount = 1);


#endif
