#ifndef __TUM3D_CUDACOMPRESS__HUFFMAN_CPU_H__
#define __TUM3D_CUDACOMPRESS__HUFFMAN_CPU_H__


#include <cudaCompress/global.h>

#include <vector>

#include <cudaCompress/BitStream.h>

#include "HuffmanTableCPU.h"


namespace cudaCompress {

bool huffmanEncodeCPU(BitStream& bitStream, const std::vector<Symbol16>& symbolStream, const HuffmanEncodeTableCPU& encodeTable, std::vector<uint>& offsets, uint codingBlockSize);
bool huffmanDecodeCPU(BitStreamReadOnly& bitStream, uint symbolCount, std::vector<Symbol16>& symbolStream, const HuffmanDecodeTableCPU& decodeTable);

}


#endif
