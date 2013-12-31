#include "GolombRiceCPU.h"

#include <cmath>

#include <cudaCompress/util.h>


namespace cudaCompress {

static const uint GLIMIT = 32; // maximum allowable codeword length

void encodeSymbol(BitStream& bitStream, Symbol16 symbol, uint k, uint qmax)
{
    uint m = 1 << k;
    uint q = symbol / m;

    if(q < GLIMIT - qmax - 1) {
        // regular golomb-rice coding

        uint r = symbol - q * m;

        // write q zeros followed by a single one
        while(q > 31) {
            bitStream.writeBits(0, 32);
            q -= 32;
        }
        bitStream.writeBits(1, q + 1);

        // write r in k bits
        bitStream.writeBits(r, k);
    } else {
        // length-limited mode

        // write glimit-qmax-1 zeros followed by a single one
        uint zeros = GLIMIT - qmax - 1;
        while(zeros > 31) {
            bitStream.writeBits(0, 32);
            zeros -= 32;
        }
        bitStream.writeBits(1, zeros + 1);

        // write symbol in qmax bits
        bitStream.writeBits(symbol - 1, qmax);
    }
}

uint getCodewordLengthForSymbol(Symbol16 symbol, uint k, uint qmax)
{
    uint m = 1 << k;
    uint q = symbol / m;

    uint result = 0;

    if(q < GLIMIT - qmax - 1) {
        // regular golomb-rice coding

        uint r = symbol - q * m;

        // write q zeros followed by a single one
        result += q + 1;

        // write r in k bits
        result += k;
    } else {
        // length-limited mode

        // write glimit-qmax-1 zeros followed by a single one
        result += GLIMIT - qmax;

        // write symbol in qmax bits
        result += qmax;
    }

    return result;
}

Symbol16 decodeSymbol(BitStreamReadOnly& bitStream, uint k, uint qmax)
{
    // read until first one bit; number of zeros == q
    uint q = 0;
    uint temp = 0;
    bitStream.readBit(temp);
    while(!temp) {
        ++q;
        bitStream.readBit(temp);
    }

    if(q == GLIMIT - qmax - 1) {
        // length-limited mode

        // read symbol
        uint val = 0;
        bitStream.readBits(val, qmax);
        return Symbol16(val + 1);
    } else {
        // regular golomb-rice coding

        // read r (k bits)
        uint r = 0;
        bitStream.readBits(r, k);

        uint m = 1 << k;
        return Symbol16(q * m + r);
    }
}

Symbol16 findSymbolMax(const std::vector<Symbol16>& symbolStream, uint offset, uint count)
{
    Symbol16 symbolMax = 0;
    for(uint index = offset; index < offset + count; index++) {
        symbolMax = max(symbolMax, symbolStream[index]);
    }
    return symbolMax;
}

uint findReasonableK(const std::vector<Symbol16>& symbolStream, uint offset, uint count, Symbol16 symbolMax)
{
    uint64 symbolSum = 0;
    for(uint index = offset; index < offset + count; index++) {
        symbolSum += symbolStream[index];
    }
    double symbolAvg = double(symbolSum) / double(count);

    uint k = (uint)clamp(int(ceil(log(0.5 * symbolAvg) / log(2.0))), 0, 15);

    return k;
}

uint findOptimalK(const std::vector<Symbol16>& symbolStream, uint offset, uint count, Symbol16 symbolMax)
{
    // encode using all possible values of k and use the lowest
    uint kBest = 0;
    uint sizeBest = UINT_MAX;

    // number of bits in (symbolMax-1)
    uint qmax = getRequiredBits(max(uint(symbolMax), 1u) - 1u);

    BitStream bitStream;
    for(uint k = 0; k < 16; k++) {
        bitStream.setBitSize(0);

        // encode symbols
        uint size = 0;
        for(uint index = offset; index < offset + count; index++) {
            size += getCodewordLengthForSymbol(symbolStream[index], k, qmax);
        }

        if(size < sizeBest) {
            sizeBest = size;
            kBest = k;
        }
    }

    return kBest;
}

bool golombRiceEncodeCPU(BitStream& bitStream, const std::vector<Symbol16>& symbolStream, uint kBlockSize, std::vector<uint>& offsets, uint codingBlockSizeInKBlocks)
{
    uint offsetBase = bitStream.getBitPosition();
    for(uint block = 0, blockOffset = 0; blockOffset < symbolStream.size(); ++block, blockOffset += kBlockSize) {
        if(codingBlockSizeInKBlocks > 0 && block % codingBlockSizeInKBlocks == 0) {
            offsets.push_back(bitStream.getBitPosition() - offsetBase);
        }

        uint blockSizeCur = min(blockOffset + kBlockSize, (uint)symbolStream.size()) - blockOffset;

        // compute k
        Symbol16 symbolMax = findSymbolMax(symbolStream, blockOffset, blockSizeCur);
        // guesstimate a good value for k
        uint k = findReasonableK(symbolStream, blockOffset, blockSizeCur, symbolMax);
        //// find optimal k by exhaustive search - encoding will take a bit more than twice as long
        //uint k = findOptimalK(symbolStream, blockOffset, blockSizeCur, symbolMax);
        bitStream.writeBits(k, 4);

        // number of bits in (symbolMax-1)
        uint qmax = getRequiredBits(max(uint(symbolMax), 1u) - 1u);
        bitStream.writeBits(qmax, 4);

        // encode symbols
        for(uint index = blockOffset; index < blockOffset + blockSizeCur; index++) {
            encodeSymbol(bitStream, symbolStream[index], k, qmax);
        }
    }

    return true;
}

bool golombRiceDecodeCPU(BitStreamReadOnly& bitStream, uint symbolCount, std::vector<Symbol16>& symbolStream, uint kBlockSize)
{
    for(uint offset = 0; offset < symbolCount; offset += kBlockSize) {
        uint blockSizeCur = min(offset + kBlockSize, symbolCount) - offset;

        // read k and qmax
        uint k = 0;
        bitStream.readBits(k, 4);
        uint qmax = 0;
        bitStream.readBits(qmax, 4);

        // decode symbols
        for(uint index = offset; index < offset + blockSizeCur; index++) {
            symbolStream.push_back(decodeSymbol(bitStream, k, qmax));
        }
    }

    return true;
}

}
