#ifndef __TUM3D_CUDACOMPRESS__ARITHMETIC_CPU_H__
#define __TUM3D_CUDACOMPRESS__ARITHMETIC_CPU_H__


#include <cudaCompress/global.h>

#include <vector>

#include <cudaCompress/BitStream.h>
#include <cudaCompress/EncodeCommon.h>


namespace cudaCompress {

bool arithmeticEncodeCPU(BitStream& bitStream, const std::vector<Symbol16>& symbolStream, const std::vector<uint>& symbolProbabilitiesCum, std::vector<uint>& offsets, uint codingBlockSize);
bool arithmeticDecodeCPU(BitStreamReadOnly& bitStream, uint symbolCount, std::vector<Symbol16>& symbolStream, const std::vector<uint>& symbolProbabilitiesCum, const std::vector<uint>& offsets, uint codingBlockSize);

}


#endif
