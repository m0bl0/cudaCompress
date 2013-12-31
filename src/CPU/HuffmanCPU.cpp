#include "HuffmanCPU.h"

#include <cassert>


namespace cudaCompress {

bool huffmanEncodeCPU(BitStream& bitStream, const std::vector<Symbol16>& symbolStream, const HuffmanEncodeTableCPU& encodeTable, std::vector<uint>& offsets, uint codingBlockSize)
{
    uint offsetBase = bitStream.getBitPosition();
    for(uint i = 0; i < symbolStream.size(); i++) {
        if(codingBlockSize > 0 && i % codingBlockSize == 0) {
            offsets.push_back(bitStream.getBitPosition() - offsetBase);
        }
        Symbol16 symbol = symbolStream[i];
        const HuffmanEntry& entry = encodeTable.m_encodeTable[symbol];
        bitStream.writeBits(entry.m_codeword, entry.m_codewordLength);
    }

    return true;
}

bool huffmanDecodeCPU(BitStreamReadOnly& bitStream, uint symbolCount, std::vector<Symbol16>& symbolStream, const HuffmanDecodeTableCPU& decodeTable)
{
    for(uint i = 0; i < symbolCount; i++) {
        if(!decodeTable.m_codewordMaxPerLength.empty()) {
            uint codewordLength = 0;
            int codeword = 0;

            do {
                // get next bit of current codeword
                uint bit = 0;
                bitStream.readBit(bit);
                codewordLength++;
                codeword <<= 1;
                codeword |= bit;

                assert(codewordLength < sizeof(int) * 8);
            } while(codeword > decodeTable.m_codewordMaxPerLength[codewordLength - 1]);

            uint codewordIndex = decodeTable.m_codewordFirstIndexPerLength[codewordLength - 1] + codeword - decodeTable.m_codewordMinPerLength[codewordLength - 1];
            symbolStream.push_back(decodeTable.m_decodeTable[codewordIndex].m_symbol);
        } else {
            symbolStream.push_back(decodeTable.m_decodeTable[0].m_symbol);
        }
    }

    return true;
}

}
