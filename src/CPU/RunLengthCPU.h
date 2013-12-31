#ifndef __TUM3D_CUDACOMPRESS__RUN_LENGTH_CPU_H__
#define __TUM3D_CUDACOMPRESS__RUN_LENGTH_CPU_H__


#include <cudaCompress/global.h>

#include <vector>

#include <cudaCompress/EncodeCommon.h>


namespace cudaCompress {

bool runLengthZeroesEncodeCPU(std::vector<Symbol16>& symbolsCompact, std::vector<ushort>& zeroCounts, const std::vector<Symbol16>& symbols, uint zeroCountMax);
bool runLengthZeroesDecodeCPU(const std::vector<Symbol16>& symbolsCompact, const std::vector<ushort>& zeroCounts, std::vector<Symbol16>& symbols, uint symbolCount);

bool runLengthEncodeCPU(std::vector<Symbol16>& runSymbols, std::vector<ushort>& runLengths, const std::vector<Symbol16>& symbols, uint runLengthMax);
bool runLengthDecodeCPU(const std::vector<Symbol16>& runSymbols, const std::vector<ushort>& runLengths, std::vector<Symbol16>& symbols, uint symbolCount);

}


#endif
