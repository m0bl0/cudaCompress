#ifndef __TUM3D_CUDACOMPRESS__HUFFMAN_TABLE_CPU_H__
#define __TUM3D_CUDACOMPRESS__HUFFMAN_TABLE_CPU_H__


#include <cudaCompress/global.h>

#include <vector>

#include <cudaCompress/BitStream.h>
#include <cudaCompress/EncodeCommon.h>


namespace cudaCompress {

class HuffmanTableCPU
{
public:
    void clear();
    void design(const std::vector<Symbol16>& symbolStream);

    void writeToBitStream(BitStream& bitstream) const;
    void readFromBitStream(BitStreamReadOnly& bitstream);

    bool operator==(const HuffmanTableCPU& rhs) const { return m_symbols == rhs.m_symbols && m_codewordCountPerLength == rhs.m_codewordCountPerLength; }
    bool operator!=(const HuffmanTableCPU& rhs) const { return !(*this == rhs); }

    std::vector<Symbol16> m_symbols;
    std::vector<uint>   m_codewordCountPerLength;
};

struct HuffmanEntry
{
    Symbol16 m_symbol;
    int    m_codeword;
    uint   m_codewordLength;
};

class HuffmanDecodeTableCPU
{
public:
    void clear();
    void build(const HuffmanTableCPU& table);

    // indexed by codeword index
    std::vector<HuffmanEntry> m_decodeTable;

    // indexed by codeword length
    std::vector<int> m_codewordFirstIndexPerLength;
    std::vector<int> m_codewordMinPerLength;
    std::vector<int> m_codewordMaxPerLength;
};

class HuffmanEncodeTableCPU
{
public:
    void clear();
    void build(const HuffmanDecodeTableCPU& decodeTable);

    // indexed by symbol
    std::vector<HuffmanEntry> m_encodeTable;
};

}


#endif
