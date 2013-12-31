#include "ArithmeticCPU.h"

#include <cmath>

#include <cudaCompress/util.h>


namespace cudaCompress {

bool arithmeticEncodeCPU(BitStream& bitStream, const std::vector<Symbol16>& symbolStream, const std::vector<uint>& symbolProbabilitiesCum, std::vector<uint>& offsets, uint codingBlockSize)
{
    uint offsetBase = bitStream.getBitPosition();

    uint intervalBits = 2 + getRequiredBits(symbolProbabilitiesCum.back() - 1);

    uint intervalMask = (1 << intervalBits) - 1;
    uint intervalMid = 1 << (intervalBits - 1);
    uint intervalOneQuarter   = 1 << (intervalBits - 2);
    uint intervalThreeQuarter = intervalMid + intervalOneQuarter;

    // interval bounds
    uint lower = 0;
    uint upper = intervalMask;

    uint symbolCount = uint(symbolStream.size());
    uint scale3 = 0;
    for(uint s = 0; s < symbolCount; s++) {
        if(codingBlockSize > 0 && s % codingBlockSize == 0) {
            offsets.push_back(bitStream.getBitPosition() - offsetBase);
        }

        Symbol16 symbol = symbolStream[s];

        // update interval
        uint64 len = uint64(upper - lower) + 1;
        uint lowerNew = uint(lower + (len * uint64(symbolProbabilitiesCum[symbol]  ) / symbolCount));
        uint upperNew = uint(lower + (len * uint64(symbolProbabilitiesCum[symbol+1]) / symbolCount)) - 1;
        lower = lowerNew;
        upper = upperNew;

        // shift out and send
        for(;;) {
            if(lower >= intervalMid || upper < intervalMid) {
                // first bit of lower and upper agrees
                uint b = (lower & intervalMid) >> (intervalBits - 1);
                bitStream.writeBits(b, 1);

                lower <<= 1;
                upper <<= 1; ++upper;

                lower &= intervalMask;
                upper &= intervalMask;

                while(scale3 > 0) {
                    bitStream.writeBits(1 - b, 1);
                    --scale3;
                }
            } else if(lower >= intervalOneQuarter && upper < intervalThreeQuarter) {
                // middle straddling case, rescale
                lower <<= 1;
                upper <<= 1; ++upper;

                lower &= intervalMask;
                upper &= intervalMask;

                lower ^= intervalMid;
                upper ^= intervalMid;

                ++scale3;
            } else {
                break;
            }
        }

        // was this the end of a block?
        if((codingBlockSize > 0 && s % codingBlockSize == codingBlockSize - 1) || (s == symbolCount - 1)) {
            // send final tag for this block - use value of lower
            uint b = (lower & intervalMid) >> (intervalBits - 1);
            assert(b == 0);
            bitStream.writeBits(b, 1);
            while(scale3 > 0) {
                bitStream.writeBits(1 - b, 1);
                --scale3;
            }
            lower <<= 1;
            lower &= intervalMask;
            // write only as many bits as required to identify this interval
            while(lower > 0) {
                uint b = (lower & intervalMid) >> (intervalBits - 1);
                if(b == 0) {
                    bitStream.writeBits(1, 1);
                    break;
                }
                bitStream.writeBits(b, 1);
                lower <<= 1;
                lower &= intervalMask;
            }
            // reset bounds
            lower = 0;
            upper = intervalMask;
        }
    }

    return true;
}

bool arithmeticDecodeCPU(BitStreamReadOnly& bitStream, uint symbolCount, std::vector<Symbol16>& symbolStream, const std::vector<uint>& symbolProbabilitiesCum, const std::vector<uint>& offsets, uint codingBlockSize)
{
    uint offsetBase = bitStream.getBitPosition();

    uint intervalBits = 2 + getRequiredBits(symbolProbabilitiesCum.back() - 1);

    uint intervalMask = (1 << intervalBits) - 1;
    uint intervalMid = 1 << (intervalBits - 1);
    uint intervalOneQuarter   = 1 << (intervalBits - 2);
    uint intervalThreeQuarter = intervalMid + intervalOneQuarter;

    // interval bounds
    uint lower = 0;
    uint upper = intervalMask;

    uint tag = 0;

    uint block = 0;
    for(uint s = 0; s < symbolCount; s++) {
        // is this a new block?
        if((codingBlockSize > 0 && s % codingBlockSize == 0) || (s == 0)) {
            // reset interval and get new tag
            lower = 0;
            upper = intervalMask;
            tag = 0;
            bitStream.setBitPosition(offsetBase + offsets[block++]);
            bitStream.readBits(tag, intervalBits);
        }

        // current position in the interval
        uint pos = ((uint64(tag - lower) + 1) * symbolCount - 1) / (uint64(upper - lower) + 1);
        // find matching symbol
        uint symbol = 0;
        //TODO binary search
        while(pos >= symbolProbabilitiesCum[symbol+1])
            ++symbol;
        // and write it out
        symbolStream.push_back(symbol);

        // update interval
        uint64 len = uint64(upper - lower) + 1;
        uint lowerNew = uint(lower + (len * uint64(symbolProbabilitiesCum[symbol]  ) / symbolCount));
        uint upperNew = uint(lower + (len * uint64(symbolProbabilitiesCum[symbol+1]) / symbolCount)) - 1;
        lower = lowerNew;
        upper = upperNew;

        // shift out and read next bits
        for(;;) {
            if(lower >= intervalMid || upper < intervalMid) {
                // first bit of lower and upper agrees
                lower <<= 1;
                upper <<= 1; ++upper;
                tag   <<= 1;
                bitStream.readBits(tag, 1);

                lower &= intervalMask;
                upper &= intervalMask;
                tag   &= intervalMask;
            } else if(lower >= intervalOneQuarter && upper < intervalThreeQuarter) {
                // middle straddling case, rescale
                lower <<= 1;
                upper <<= 1; ++upper;
                tag   <<= 1;
                bitStream.readBits(tag, 1);

                lower &= intervalMask;
                upper &= intervalMask;
                tag   &= intervalMask;

                lower ^= intervalMid;
                upper ^= intervalMid;
                tag   ^= intervalMid;
            } else {
                break;
            }
        }
    }

    return false;
}

}
