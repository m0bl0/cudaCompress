#include "EncodeCPU.h"

#include <cassert>

#include <cudaCompress/util.h>

#include "HuffmanCPU.h"
#include "HuffmanTableCPU.h"
#include "GolombRiceCPU.h"
#include "RunLengthCPU.h"
#include "RBUCCPU.h"
#include "ArithmeticCPU.h"


namespace cudaCompress {

std::string getCoderName(ECoder coder)
{
    switch(coder) {
        case CODER_ARITHMETIC:
            return "Arithmetic";
        case CODER_GOLOMBRICE:
            return "Golomb-Rice";
        case CODER_HUFFMAN:
            return "Huffman";
        case CODER_RBUC:
            return "RBUC";
        case CODER_RUNLENGTH:
            return "Run-length";
        case CODER_RUNLENGTH_GOLOMBRICE:
            return "Run-length + Golomb-Rice";
        case CODER_RUNLENGTH_HUFFMAN:
            return "Run-length + Huffman";
        default:
            return false;
    }
}



bool encodeCPU(ECoder coder, BitStream* pBitStream, /*const*/ std::vector<Symbol16>* pSymbolStream, uint codingBlockSize)
{
    return encodeCPU(coder, &pBitStream, &pSymbolStream, 1, codingBlockSize);
}

bool decodeCPU(ECoder coder, BitStreamReadOnly* pBitStream, std::vector<Symbol16>* pSymbolStream, uint symbolCount, uint codingBlockSize)
{
    return decodeCPU(coder, &pBitStream, &pSymbolStream, symbolCount, 1, codingBlockSize);
}



bool encodeCPU(ECoder coder, BitStream* ppBitStreams[], /*const*/ std::vector<Symbol16>* ppSymbolStreams[], uint blockCount, uint codingBlockSize)
{
    switch(coder) {
        case CODER_ARITHMETIC:
            return encodeArithmeticCPU(ppBitStreams, ppSymbolStreams, blockCount, codingBlockSize);
        case CODER_GOLOMBRICE:
            return encodeGolombRiceCPU(ppBitStreams, ppSymbolStreams, blockCount, codingBlockSize);
        case CODER_HUFFMAN:
            return encodeHuffCPU(ppBitStreams, ppSymbolStreams, blockCount, codingBlockSize);
        case CODER_RBUC:
            return encodeRBUCCPU(ppBitStreams, ppSymbolStreams, blockCount);
        case CODER_RUNLENGTH:
            return encodeRunLengthCPU(ppBitStreams, ppSymbolStreams, blockCount);
        case CODER_RUNLENGTH_GOLOMBRICE:
            return encodeRLGolombRiceCPU(ppBitStreams, ppSymbolStreams, blockCount, codingBlockSize);
        case CODER_RUNLENGTH_HUFFMAN:
            return encodeRLHuffCPU(ppBitStreams, ppSymbolStreams, blockCount, codingBlockSize);
        default:
            return false;
    }
}

bool decodeCPU(ECoder coder, BitStreamReadOnly* ppBitStreams[], std::vector<Symbol16>* ppSymbolStreams[], uint symbolCountPerBlock, uint blockCount, uint codingBlockSize)
{
    switch(coder) {
        case CODER_ARITHMETIC:
            return decodeArithmeticCPU(ppBitStreams, ppSymbolStreams, symbolCountPerBlock, blockCount, codingBlockSize);
        case CODER_GOLOMBRICE:
            return decodeGolombRiceCPU(ppBitStreams, ppSymbolStreams, symbolCountPerBlock, blockCount, codingBlockSize);
        case CODER_HUFFMAN:
            return decodeHuffCPU(ppBitStreams, ppSymbolStreams, symbolCountPerBlock, blockCount, codingBlockSize);
        case CODER_RBUC:
            return decodeRBUCCPU(ppBitStreams, ppSymbolStreams, symbolCountPerBlock, blockCount);
        case CODER_RUNLENGTH:
            return decodeRunLengthCPU(ppBitStreams, ppSymbolStreams, symbolCountPerBlock, blockCount);
        case CODER_RUNLENGTH_GOLOMBRICE:
            return decodeRLGolombRiceCPU(ppBitStreams, ppSymbolStreams, symbolCountPerBlock, blockCount, codingBlockSize);
        case CODER_RUNLENGTH_HUFFMAN:
            return decodeRLHuffCPU(ppBitStreams, ppSymbolStreams, symbolCountPerBlock, blockCount, codingBlockSize);
        default:
            return false;
    }
}


bool encodeRLHuffCPU(BitStream* ppBitStreams[], /*const*/ std::vector<Symbol16>* ppSymbolStreams[], uint blockCount, uint codingBlockSize, bool zeroRunsOnly)
{
    for(uint block = 0; block < blockCount; block++) {
        BitStream& bitStream = *ppBitStreams[block];

        // run length encode
        std::vector<Symbol16> symbolsCompact;
        std::vector<ushort> zeroCounts;
        if(zeroRunsOnly) {
            runLengthZeroesEncodeCPU(symbolsCompact, zeroCounts, *ppSymbolStreams[block], ZERO_COUNT_MAX);
        } else {
            runLengthEncodeCPU(symbolsCompact, zeroCounts, *ppSymbolStreams[block], ZERO_COUNT_MAX);
        }

        // write compacted symbol count
        uint compactedSymbolCount = (uint)symbolsCompact.size();
        bitStream.writeAligned(&compactedSymbolCount, 1);


        // 1. compacted symbols
        HuffmanTableCPU table;
        table.design(symbolsCompact);

        table.writeToBitStream(bitStream);

        HuffmanDecodeTableCPU decodeTable;
        decodeTable.build(table);

        HuffmanEncodeTableCPU encodeTable;
        encodeTable.build(decodeTable);

        // leave room for codeword stream bitsize
        bitStream.writeAlign<uint>();
        uint codewordBitsizeIndex = bitStream.getBitPosition() / (8 * sizeof(uint));
        uint codewordBitsize = 0;
        bitStream.writeAligned(&codewordBitsize, 1);

        // encode, write codeword stream to bitstream
        std::vector<uint> offsets;
        uint codewordStartBit = bitStream.getBitPosition();
        huffmanEncodeCPU(bitStream, symbolsCompact, encodeTable, offsets, codingBlockSize);

        // write codeword bitsize to reserved position
        codewordBitsize = bitStream.getBitPosition() - codewordStartBit;
        bitStream.getRaw()[codewordBitsizeIndex] = codewordBitsize;

        bitStream.writeAlign<uint>();

        // make offsets incremental
        std::vector<ushort> offsetIncrements(offsets.size());
        for(uint i = 1; i < (uint)offsets.size(); i++) {
            offsetIncrements[i] = offsets[i] - offsets[i-1];
        }

        // write offsets
        //bitStream.writeAligned(&offsetBits, 1);
        //for(uint i = 0; i < offsets.size(); i++)
        //    bitStream.writeBits(offsets[i], offsetBits);
        //bitStream.writeAlign<uint>();
        bitStream.writeAligned(offsetIncrements.data(), (uint)offsetIncrements.size());


        // 2. zero counts
        table.design(zeroCounts);
        table.writeToBitStream(bitStream);

        decodeTable.build(table);
        encodeTable.build(decodeTable);

        // leave room for codeword stream bitsize
        bitStream.writeAlign<uint>();
        codewordBitsizeIndex = bitStream.getBitPosition() / (8 * sizeof(uint));
        codewordBitsize = 0;
        bitStream.writeAligned(&codewordBitsize, 1);

        // encode, write codeword stream to bitstream
        offsets.clear();
        codewordStartBit = bitStream.getBitPosition();
        huffmanEncodeCPU(bitStream, zeroCounts, encodeTable, offsets, codingBlockSize);

        // write codeword bitsize to reserved position
        codewordBitsize = bitStream.getBitPosition() - codewordStartBit;
        bitStream.getRaw()[codewordBitsizeIndex] = codewordBitsize;

        bitStream.writeAlign<uint>();

        // make offsets incremental
        for(uint i = 1; i < (uint)offsets.size(); i++) {
            offsetIncrements[i] = offsets[i] - offsets[i-1];
        }

        // write offsets
        //bitStream.writeAligned(&offsetBits, 1);
        //for(uint i = 0; i < offsets.size(); i++)
        //    bitStream.writeBits(offsets[i], offsetBits);
        //bitStream.writeAlign<uint>();
        bitStream.writeAligned(offsetIncrements.data(), (uint)offsetIncrements.size());
    }

    return true;
}

bool decodeRLHuffCPU(BitStreamReadOnly* ppBitStreams[], std::vector<Symbol16>* ppSymbolStreams[], uint symbolCountPerBlock, uint blockCount, uint codingBlockSize, bool zeroRunsOnly)
{
    for(uint block = 0; block < blockCount; block++) {
        BitStreamReadOnly& bitStream = *ppBitStreams[block];

        // read compacted symbol count
        uint compactedSymbolCount;
        bitStream.readAligned(&compactedSymbolCount, 1);


        // 1. compacted symbols
        HuffmanTableCPU table;
        table.readFromBitStream(bitStream);

        HuffmanDecodeTableCPU decodeTable;
        decodeTable.build(table);

        uint codewordBitsize = 0;
        bitStream.readAligned(&codewordBitsize, 1);

        uint codewordStartBit = bitStream.getBitPosition();
        std::vector<Symbol16> symbolsCompact;
        huffmanDecodeCPU(bitStream, compactedSymbolCount, symbolsCompact, decodeTable);

        assert(bitStream.getBitPosition() - codewordStartBit == codewordBitsize);

        bitStream.align<uint>();

        uint offsetCount = (compactedSymbolCount + codingBlockSize - 1) / codingBlockSize;
        // skip offsets
        //uint offsetBits = 0;
        //bitStream.readAligned(&offsetBits, 1);
        //bitStream.skipBits(offsetCount * offsetBits);
        //bitStream.align<uint>();
        bitStream.skipAligned<ushort>(offsetCount);


        // 2. zero counts
        table.readFromBitStream(bitStream);
        decodeTable.build(table);

        codewordBitsize = 0;
        bitStream.readAligned(&codewordBitsize, 1);

        codewordStartBit = bitStream.getBitPosition();
        std::vector<ushort> zeroCounts;
        huffmanDecodeCPU(bitStream, compactedSymbolCount, zeroCounts, decodeTable);

        assert(bitStream.getBitPosition() - codewordStartBit == codewordBitsize);

        bitStream.align<uint>();

        // skip offsets
        //offsetBits = 0;
        //bitStream.readAligned(&offsetBits, 1);
        //bitStream.skipBits(offsetCount * offsetBits);
        //bitStream.align<uint>();
        bitStream.skipAligned<ushort>(offsetCount);


        // run length decode
        if(zeroRunsOnly) {
            runLengthZeroesDecodeCPU(symbolsCompact, zeroCounts, *ppSymbolStreams[block], symbolCountPerBlock);
        } else {
            runLengthDecodeCPU(symbolsCompact, zeroCounts, *ppSymbolStreams[block], symbolCountPerBlock);
        }
    }

    return true;
}



bool encodeHuffCPU(BitStream* ppBitStreams[], /*const*/ std::vector<Symbol16>* ppSymbolStreams[], uint blockCount, uint codingBlockSize)
{
    for(uint block = 0; block < blockCount; block++) {
        BitStream& bitStream = *ppBitStreams[block];

        HuffmanTableCPU table;
        table.design(*ppSymbolStreams[block]);

        table.writeToBitStream(bitStream);

        HuffmanDecodeTableCPU decodeTable;
        decodeTable.build(table);

        HuffmanEncodeTableCPU encodeTable;
        encodeTable.build(decodeTable);

        // leave room for codeword stream bitsize
        bitStream.writeAlign<uint>();
        uint codewordBitsizeIndex = bitStream.getBitPosition() / (8 * sizeof(uint));
        uint codewordBitsize = 0;
        bitStream.writeAligned(&codewordBitsize, 1);

        // encode, write codeword stream to bitstream
        std::vector<uint> offsets;
        uint codewordStartBit = bitStream.getBitPosition();
        huffmanEncodeCPU(bitStream, *ppSymbolStreams[block], encodeTable, offsets, codingBlockSize);

        // write codeword bitsize to reserved position
        codewordBitsize = bitStream.getBitPosition() - codewordStartBit;
        bitStream.getRaw()[codewordBitsizeIndex] = codewordBitsize;

        bitStream.writeAlign<uint>();

        // make offsets incremental
        std::vector<ushort> offsetIncrements(offsets.size());
        for(uint i = 1; i < (uint)offsets.size(); i++) {
            offsetIncrements[i] = offsets[i] - offsets[i-1];
        }

        // write offsets
        //bitStream.writeAligned(&offsetBits, 1);
        //for(uint i = 0; i < offsets.size(); i++)
        //    bitStream.writeBits(offsets[i], offsetBits);
        //bitStream.writeAlign<uint>();
        bitStream.writeAligned(offsetIncrements.data(), (uint)offsetIncrements.size());
    }

    return true;
}

bool decodeHuffCPU(BitStreamReadOnly* ppBitStreams[], std::vector<Symbol16>* ppSymbolStreams[], uint symbolCountPerBlock, uint blockCount, uint codingBlockSize)
{
    for(uint block = 0; block < blockCount; block++) {
        BitStreamReadOnly& bitStream = *ppBitStreams[block];

        HuffmanTableCPU table;
        table.readFromBitStream(bitStream);

        HuffmanDecodeTableCPU decodeTable;
        decodeTable.build(table);

        uint codewordBitsize = 0;
        bitStream.readAligned(&codewordBitsize, 1);

        uint codewordStartBit = bitStream.getBitPosition();
        huffmanDecodeCPU(bitStream, symbolCountPerBlock, *ppSymbolStreams[block], decodeTable);

        assert(bitStream.getBitPosition() - codewordStartBit == codewordBitsize);

        bitStream.align<uint>();

        uint offsetCount = (symbolCountPerBlock + codingBlockSize - 1) / codingBlockSize;
        // skip offsets
        //uint offsetBits = 0;
        //bitStream.readAligned(&offsetBits, 1);
        //bitStream.skipBits(offsetCount * offsetBits);
        //bitStream.align<uint>();
        bitStream.skipAligned<ushort>(offsetCount);
    }

    return true;
}



bool encodeRLGolombRiceCPU(BitStream* ppBitStreams[], /*const*/ std::vector<Symbol16>* ppSymbolStreams[], uint blockCount, uint codingBlockSize, bool zeroRunsOnly)
{
    for(uint block = 0; block < blockCount; block++) {
        BitStream& bitStream = *ppBitStreams[block];

        // run length encode
        std::vector<Symbol16> symbolsCompact;
        std::vector<ushort> zeroCounts;
        if(zeroRunsOnly) {
            runLengthZeroesEncodeCPU(symbolsCompact, zeroCounts, *ppSymbolStreams[block], ZERO_COUNT_MAX);
        } else {
            runLengthEncodeCPU(symbolsCompact, zeroCounts, *ppSymbolStreams[block], ZERO_COUNT_MAX);
        }

        // write compacted symbol count
        uint compactedSymbolCount = (uint)symbolsCompact.size();
        bitStream.writeAligned(&compactedSymbolCount, 1);


        // 1. compacted symbols

        // leave room for codeword stream bitsize
        bitStream.writeAlign<uint>();
        uint codewordBitsizeIndex = bitStream.getBitPosition() / (8 * sizeof(uint));
        uint codewordBitsize = 0;
        bitStream.writeAligned(&codewordBitsize, 1);

        // encode, write codeword stream to bitstream
        std::vector<uint> offsets;
        uint codewordStartBit = bitStream.getBitPosition();
        golombRiceEncodeCPU(bitStream, symbolsCompact, codingBlockSize, offsets, 1);

        // write codeword bitsize to reserved position
        codewordBitsize = bitStream.getBitPosition() - codewordStartBit;
        bitStream.getRaw()[codewordBitsizeIndex] = codewordBitsize;

        bitStream.writeAlign<uint>();

        // make offsets incremental
        std::vector<ushort> offsetIncrements(offsets.size());
        for(uint i = 1; i < (uint)offsets.size(); i++) {
            offsetIncrements[i] = offsets[i] - offsets[i-1];
        }

        // write offsets
        bitStream.writeAligned(offsetIncrements.data(), (uint)offsetIncrements.size());


        // 2. zero counts

        // leave room for codeword stream bitsize
        bitStream.writeAlign<uint>();
        codewordBitsizeIndex = bitStream.getBitPosition() / (8 * sizeof(uint));
        codewordBitsize = 0;
        bitStream.writeAligned(&codewordBitsize, 1);

        // encode, write codeword stream to bitstream
        offsets.clear();
        codewordStartBit = bitStream.getBitPosition();
        golombRiceEncodeCPU(bitStream, zeroCounts, codingBlockSize, offsets, 1);

        // write codeword bitsize to reserved position
        codewordBitsize = bitStream.getBitPosition() - codewordStartBit;
        bitStream.getRaw()[codewordBitsizeIndex] = codewordBitsize;

        bitStream.writeAlign<uint>();

        // make offsets incremental
        for(uint i = 1; i < (uint)offsets.size(); i++) {
            offsetIncrements[i] = offsets[i] - offsets[i-1];
        }

        // write offsets
        bitStream.writeAligned(offsetIncrements.data(), (uint)offsetIncrements.size());
    }

    return true;
}

bool decodeRLGolombRiceCPU(BitStreamReadOnly* ppBitStreams[], std::vector<Symbol16>* ppSymbolStreams[], uint symbolCountPerBlock, uint blockCount, uint codingBlockSize, bool zeroRunsOnly)
{
    for(uint block = 0; block < blockCount; block++) {
        BitStreamReadOnly& bitStream = *ppBitStreams[block];

        // read compacted symbol count
        uint compactedSymbolCount;
        bitStream.readAligned(&compactedSymbolCount, 1);


        // 1. compacted symbols
        uint codewordBitsize = 0;
        bitStream.readAligned(&codewordBitsize, 1);

        uint codewordStartBit = bitStream.getBitPosition();
        std::vector<Symbol16> symbolsCompact;
        golombRiceDecodeCPU(bitStream, compactedSymbolCount, symbolsCompact, codingBlockSize);

        assert(bitStream.getBitPosition() - codewordStartBit == codewordBitsize);

        bitStream.align<uint>();

        uint offsetCount = (compactedSymbolCount + codingBlockSize - 1) / codingBlockSize;
        // skip offsets
        bitStream.skipAligned<ushort>(offsetCount);


        // 2. zero counts
        codewordBitsize = 0;
        bitStream.readAligned(&codewordBitsize, 1);

        codewordStartBit = bitStream.getBitPosition();
        std::vector<ushort> zeroCounts;
        golombRiceDecodeCPU(bitStream, compactedSymbolCount, zeroCounts, codingBlockSize);

        assert(bitStream.getBitPosition() - codewordStartBit == codewordBitsize);

        bitStream.align<uint>();

        // skip offsets
        bitStream.skipAligned<ushort>(offsetCount);


        // run length decode
        if(zeroRunsOnly) {
            runLengthZeroesDecodeCPU(symbolsCompact, zeroCounts, *ppSymbolStreams[block], symbolCountPerBlock);
        } else {
            runLengthDecodeCPU(symbolsCompact, zeroCounts, *ppSymbolStreams[block], symbolCountPerBlock);
        }
    }

    return true;
}



bool encodeGolombRiceCPU(BitStream* ppBitStreams[], /*const*/ std::vector<Symbol16>* ppSymbolStreams[], uint blockCount, uint codingBlockSize)
{
    uint kBlockSize = 128;
    if(codingBlockSize % kBlockSize != 0) {
        return false;
    }
    for(uint block = 0; block < blockCount; block++) {
        BitStream& bitStream = *ppBitStreams[block];

        // leave room for codeword stream bitsize
        bitStream.writeAlign<uint>();
        uint codewordBitsizeIndex = bitStream.getBitPosition() / (8 * sizeof(uint));
        uint codewordBitsize = 0;
        bitStream.writeAligned(&codewordBitsize, 1);

        // encode, write codeword stream to bitstream
        std::vector<uint> offsets;
        uint codewordStartBit = bitStream.getBitPosition();
        golombRiceEncodeCPU(bitStream, *ppSymbolStreams[block], kBlockSize, offsets, codingBlockSize / kBlockSize);

        // write codeword bitsize to reserved position
        codewordBitsize = bitStream.getBitPosition() - codewordStartBit;
        bitStream.getRaw()[codewordBitsizeIndex] = codewordBitsize;

        bitStream.writeAlign<uint>();

        // make offsets incremental
        std::vector<ushort> offsetIncrements(offsets.size());
        for(uint i = 1; i < (uint)offsets.size(); i++) {
            offsetIncrements[i] = offsets[i] - offsets[i-1];
        }

        // write offsets
        bitStream.writeAligned(offsetIncrements.data(), (uint)offsetIncrements.size());
    }

    return true;
}

bool decodeGolombRiceCPU(BitStreamReadOnly* ppBitStreams[], std::vector<Symbol16>* ppSymbolStreams[], uint symbolCountPerBlock, uint blockCount, uint codingBlockSize)
{
    uint kBlockSize = 128;
    if(codingBlockSize % kBlockSize != 0) {
        return false;
    }
    for(uint block = 0; block < blockCount; block++) {
        BitStreamReadOnly& bitStream = *ppBitStreams[block];

        uint codewordBitsize = 0;
        bitStream.readAligned(&codewordBitsize, 1);

        uint codewordStartBit = bitStream.getBitPosition();
        golombRiceDecodeCPU(bitStream, symbolCountPerBlock, *ppSymbolStreams[block], kBlockSize);

        assert(bitStream.getBitPosition() - codewordStartBit == codewordBitsize);

        bitStream.align<uint>();

        uint offsetCount = (symbolCountPerBlock + codingBlockSize - 1) / codingBlockSize;
        // skip offsets
        bitStream.skipAligned<ushort>(offsetCount);
    }

    return true;
}



bool encodeRBUCCPU(BitStream* ppBitStreams[], /*const*/ std::vector<Symbol16>* ppSymbolStreams[], uint blockCount)
{
    std::vector<uint> branchFactors;
    branchFactors.push_back(3);
    branchFactors.push_back(6);
    branchFactors.push_back(9);

    for(uint block = 0; block < blockCount; block++) {
        BitStream& bitStream = *ppBitStreams[block];

        // leave room for codeword stream bitsize
        bitStream.writeAlign<uint>();
        uint codewordBitsizeIndex = bitStream.getBitPosition() / (8 * sizeof(uint));
        uint codewordBitsize = 0;
        bitStream.writeAligned(&codewordBitsize, 1);

        // encode, write codeword stream to bitstream
        std::vector<uint> offsets;
        uint codewordStartBit = bitStream.getBitPosition();
        rbucEncodeCPU(bitStream, offsets, *ppSymbolStreams[block], branchFactors);

        // write codeword bitsize to reserved position
        codewordBitsize = bitStream.getBitPosition() - codewordStartBit;
        bitStream.getRaw()[codewordBitsizeIndex] = codewordBitsize;

        bitStream.writeAlign<uint>();

        // make offsets incremental
        std::vector<ushort> offsetIncrements(offsets.size());
        for(uint i = 1; i < (uint)offsets.size(); i++) {
            offsetIncrements[i] = offsets[i] - offsets[i-1];
        }

        // write offsets
        //bitStream.writeAligned(&offsetBits, 1);
        //for(uint i = 0; i < offsets.size(); i++)
        //    bitStream.writeBits(offsets[i], offsetBits);
        //bitStream.align<uint>();
        bitStream.writeAligned(offsetIncrements.data(), (uint)offsetIncrements.size());
    }

    return true;
}

bool decodeRBUCCPU(BitStreamReadOnly* ppBitStreams[], std::vector<Symbol16>* ppSymbolStreams[], uint symbolCountPerBlock, uint blockCount)
{
    std::vector<uint> branchFactors;
    branchFactors.push_back(3);
    branchFactors.push_back(6);
    branchFactors.push_back(9);
    uint codingBlockSize = 3 * 6 * 9;

    for(uint block = 0; block < blockCount; block++) {
        BitStreamReadOnly& bitStream = *ppBitStreams[block];

        uint codewordBitsize = 0;
        bitStream.readAligned(&codewordBitsize, 1);

        uint codewordStartBit = bitStream.getBitPosition();
        rbucDecodeCPU(bitStream, symbolCountPerBlock, *ppSymbolStreams[block], branchFactors);

        //FIXME assert(bitStream.getBitPosition() - codewordStartBit == codewordBitsize);

        bitStream.align<uint>();

        uint offsetCount = (symbolCountPerBlock + codingBlockSize - 1) / codingBlockSize;
        // skip offsets
        //uint offsetBits = 0;
        //bitStream.readAligned(&offsetBits, 1);
        //bitStream.skipBits(offsetCount * offsetBits);
        //bitStream.align<uint>();
        bitStream.skipAligned<ushort>(offsetCount);
    }

    return true;
}


bool encodeArithmeticCPU(BitStream* ppBitStreams[], /*const*/ std::vector<Symbol16>* ppSymbolStreams[], uint blockCount, uint codingBlockSize)
{
    for(uint block = 0; block < blockCount; block++) {
        BitStream& bitStream = *ppBitStreams[block];
        const std::vector<Symbol16>& symbolStream = *ppSymbolStreams[block];

        // find max symbol
        Symbol16 symbolMax = 0;
        for(uint i = 0; i < symbolStream.size(); i++) {
            if(symbolStream[i] > symbolMax) {
                symbolMax = symbolStream[i];
            }
        }

        uint distinctSymbolCount = symbolMax + 1;

        // write distinct symbol count
        bitStream.writeBits(distinctSymbolCount, 16);

        // count symbol probabilities
        std::vector<uint> symbolProbabilities(distinctSymbolCount);
        for(uint i = 0; i < symbolStream.size(); i++) {
            symbolProbabilities[symbolStream[i]]++;
        }

        // write symbol probabilities
        uint probMax = *std::max_element(symbolProbabilities.begin(), symbolProbabilities.end());
        uint probBits = getRequiredBits(probMax);
        bitStream.writeBits(probBits, 5);
        for(uint prob : symbolProbabilities) {
            bitStream.writeBits(prob, probBits);
        }

        // compute cumulative symbol probabilities
        std::vector<uint> symbolProbabilitiesCum(distinctSymbolCount + 1);
        symbolProbabilitiesCum[0] = 0;
        for(uint i = 1; i < symbolProbabilitiesCum.size(); i++) {
            symbolProbabilitiesCum[i] = symbolProbabilitiesCum[i-1] + symbolProbabilities[i-1];
        }


        // leave room for codeword stream bitsize
        bitStream.writeAlign<uint>();
        uint codewordBitsizeIndex = bitStream.getBitPosition() / (8 * sizeof(uint));
        uint codewordBitsize = 0;
        bitStream.writeAligned(&codewordBitsize, 1);

        // encode, write codeword stream to bitstream
        std::vector<uint> offsets;
        uint codewordStartBit = bitStream.getBitPosition();
        arithmeticEncodeCPU(bitStream, symbolStream, symbolProbabilitiesCum, offsets, codingBlockSize);

        // write codeword bitsize to reserved position
        codewordBitsize = bitStream.getBitPosition() - codewordStartBit;
        bitStream.getRaw()[codewordBitsizeIndex] = codewordBitsize;

        bitStream.writeAlign<uint>();

        // make offsets incremental
        std::vector<ushort> offsetIncrements(offsets.size());
        for(uint i = 1; i < (uint)offsets.size(); i++) {
            offsetIncrements[i] = offsets[i] - offsets[i-1];
        }

        // write offsets
        bitStream.writeAligned(offsetIncrements.data(), (uint)offsetIncrements.size());
    }

    return true;
}

bool decodeArithmeticCPU(BitStreamReadOnly* ppBitStreams[], std::vector<Symbol16>* ppSymbolStreams[], uint symbolCountPerBlock, uint blockCount, uint codingBlockSize)
{
    for(uint block = 0; block < blockCount; block++) {
        BitStreamReadOnly& bitStream = *ppBitStreams[block];

        // read distinct symbol count
        uint distinctSymbolCount = 0;
        bitStream.readBits(distinctSymbolCount, 16);

        // read symbol probabilities
        uint probBits = 0;
        bitStream.readBits(probBits, 5);
        std::vector<uint> symbolProbabilities(distinctSymbolCount);
        for(uint& prob : symbolProbabilities) {
            bitStream.readBits(prob, probBits);
        }

        // compute cumulative symbol probabilities
        std::vector<uint> symbolProbabilitiesCum(distinctSymbolCount + 1);
        symbolProbabilitiesCum[0] = 0;
        for(uint i = 1; i < symbolProbabilitiesCum.size(); i++) {
            symbolProbabilitiesCum[i] = symbolProbabilitiesCum[i-1] + symbolProbabilities[i-1];
        }

        uint codewordBitsize = 0;
        bitStream.readAligned(&codewordBitsize, 1);

        uint codewordStartPos = bitStream.getBitPosition();

        // skip codewords for now
        bitStream.skipBits(codewordBitsize),

        // read offsets
        bitStream.align<uint>();
        uint offsetCount = (symbolCountPerBlock + codingBlockSize - 1) / codingBlockSize;
        std::vector<ushort> offsetIncrements(offsetCount);
        bitStream.readAligned<ushort>(offsetIncrements.data(), offsetCount);

        uint offsetEndPos = bitStream.getBitPosition();

        // make absolute
        std::vector<uint> offsets(offsetCount);
        offsets[0] = offsetIncrements[0];
        for(size_t i = 1; i < offsets.size(); i++) {
            offsets[i] = offsets[i-1] + offsetIncrements[i];
        }

        // go back to codewords and decode
        bitStream.setBitPosition(codewordStartPos);
        arithmeticDecodeCPU(bitStream, symbolCountPerBlock, *ppSymbolStreams[block], symbolProbabilitiesCum, offsets, codingBlockSize);
        bitStream.setBitPosition(offsetEndPos);
    }

    return true;
}

bool encodeRunLengthCPU(BitStream* ppBitStreams[], /*const*/ std::vector<Symbol16>* ppSymbolStreams[], uint blockCount, bool zeroRunsOnly)
{
    for(uint block = 0; block < blockCount; block++) {
        BitStream& bitStream = *ppBitStreams[block];

        // run length encode
        std::vector<Symbol16> symbolsCompact;
        std::vector<ushort> zeroCounts;
        if(zeroRunsOnly) {
            runLengthZeroesEncodeCPU(symbolsCompact, zeroCounts, *ppSymbolStreams[block], ZERO_COUNT_MAX);
        } else {
            runLengthEncodeCPU(symbolsCompact, zeroCounts, *ppSymbolStreams[block], ZERO_COUNT_MAX);
        }

        // write compacted symbol count
        uint compactedSymbolCount = (uint)symbolsCompact.size();
        bitStream.writeAligned(&compactedSymbolCount, 1);


        // 1. compacted symbols
        Symbol16 symbolMax = 0;
        for(size_t s = 0; s < symbolsCompact.size(); s++) {
            symbolMax = max(symbolMax, symbolsCompact[s]);
        }

        uint symbolBits = getRequiredBits(symbolMax);
        bitStream.writeBits(symbolBits, 5);
        for(size_t s = 0; s < symbolsCompact.size(); s++) {
            bitStream.writeBits(symbolsCompact[s], symbolBits);
        }

        // 2. zero counts
        ushort zeroCountMax = 0;
        for(size_t s = 0; s < zeroCounts.size(); s++) {
            zeroCountMax = max(zeroCountMax, zeroCounts[s]);
        }

        uint zeroCountBits = getRequiredBits(zeroCountMax);
        bitStream.writeBits(zeroCountBits, 5);
        for(size_t s = 0; s < zeroCounts.size(); s++) {
            bitStream.writeBits(zeroCounts[s], zeroCountBits);
        }
    }

    return true;
}

bool decodeRunLengthCPU(BitStreamReadOnly* ppBitStreams[], std::vector<Symbol16>* ppSymbolStreams[], uint symbolCountPerBlock, uint blockCount, bool zeroRunsOnly)
{
    for(uint block = 0; block < blockCount; block++) {
        BitStreamReadOnly& bitStream = *ppBitStreams[block];

        // read compacted symbol count
        uint compactedSymbolCount;
        bitStream.readAligned(&compactedSymbolCount, 1);


        // 1. compacted symbols
        uint symbolBits = 0;
        bitStream.readBits(symbolBits, 5);
        std::vector<Symbol16> symbolsCompact;
        symbolsCompact.reserve(compactedSymbolCount);
        for(uint i = 0; i < compactedSymbolCount; i++) {
            uint symbol = 0;
            bitStream.readBits(symbol, symbolBits);
            symbolsCompact.push_back(Symbol16(symbol));
        }

        // 2. zero counts
        uint zeroCountBits = 0;
        bitStream.readBits(zeroCountBits, 5);
        std::vector<ushort> zeroCounts;
        zeroCounts.reserve(compactedSymbolCount);
        for(uint i = 0; i < compactedSymbolCount; i++) {
            uint zeroCount = 0;
            bitStream.readBits(zeroCount, zeroCountBits);
            zeroCounts.push_back(ushort(zeroCount));
        }


        // run length decode
        if(zeroRunsOnly) {
            runLengthZeroesDecodeCPU(symbolsCompact, zeroCounts, *ppSymbolStreams[block], symbolCountPerBlock);
        } else {
            runLengthDecodeCPU(symbolsCompact, zeroCounts, *ppSymbolStreams[block], symbolCountPerBlock);
        }
    }

    return true;
}

}
