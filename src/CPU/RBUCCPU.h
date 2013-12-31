#ifndef __TUM3D_CUDACOMPRESS__RBUC_CPU_H__
#define __TUM3D_CUDACOMPRESS__RBUC_CPU_H__


#include <cudaCompress/global.h>

#include <vector>

#include <cudaCompress/BitStream.h>
#include <cudaCompress/EncodeCommon.h>


namespace cudaCompress {

bool rbucEncodeCPU(BitStream& bitStream, std::vector<uint>& offsets, const std::vector<Symbol16>& symbols, const std::vector<uint>& branchFactors);
bool rbucDecodeCPU(BitStreamReadOnly& bitStream, uint symbolCount, std::vector<Symbol16>& symbols, const std::vector<uint>& branchFactors);

}


#endif
