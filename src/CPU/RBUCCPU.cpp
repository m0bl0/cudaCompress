#include "RBUCCPU.h"

#include <algorithm>
#include <cassert>

#include <cudaCompress/util.h>


namespace cudaCompress {

void rbucEncodeCPUBuildLengthsOneLevel(std::vector<Symbol16>& lengths, const std::vector<Symbol16>& symbols, uint branchFactor)
{
    uint symbolCount = (uint)symbols.size();

    uint lengthCount = (symbolCount + branchFactor - 1) / branchFactor;
    lengths.resize(lengthCount);

    for(uint i = 0; i < lengthCount; i++) {
        Symbol16 symbolMax = symbols[i * branchFactor];
        for(uint j = i * branchFactor + 1; j < std::min((i+1) * branchFactor, symbolCount); j++) {
            symbolMax = std::max(symbolMax, symbols[j]);
        }

        uint bits = getRequiredBits(symbolMax);
        lengths[i] = Symbol16(bits);
    }
}

void rbucEncodeCPUOneLevel(BitStream& bitStream, std::vector<Symbol16>& lengths, const std::vector<Symbol16>& symbols, uint branchFactor)
{
    rbucEncodeCPUBuildLengthsOneLevel(lengths, symbols, branchFactor);

    uint symbolCount = (uint)symbols.size();

    for(uint i = 0; i < symbolCount; i++) {
        uint length = lengths[i / branchFactor];
        bitStream.writeBits(symbols[i], length);
    }
}

void rbucDecodeCPUOneLevel(BitStream& bitStream, const std::vector<Symbol16>& lengths, std::vector<Symbol16>& symbols, uint symbolCount, uint branchFactor)
{
    for(uint i = 0; i < symbolCount; i++) {
        uint length = lengths[i / branchFactor];
        uint symbol = 0;
        bitStream.readBits(symbol, length);
        symbols.push_back(symbol);
    }
}

void rbucEncodeCPUWriteTree(BitStream& bitStream, std::vector<uint>& offsets, const std::vector<std::vector<Symbol16>>& tree, const std::vector<uint>& branchFactors, uint level, uint index)
{
    if(level+2 == tree.size()) {
        offsets.push_back(bitStream.getBitPosition());
    }

    uint bits = (level+1 >= tree.size() ? 8 : tree[level+1][index / branchFactors[level]]);
    bitStream.writeBits(tree[level][index], bits);
    if(level > 0) {
        uint branchFactor = branchFactors[level - 1];
        uint childrenCount = std::min(branchFactor, (uint)tree[level - 1].size() - index * branchFactor);
        for(uint i = 0; i < childrenCount; i++) {
            rbucEncodeCPUWriteTree(bitStream, offsets, tree, branchFactors, level - 1, index * branchFactor + i);
        }
    }
}

void rbucDecodeCPUReadTree(BitStreamReadOnly& bitStream, std::vector<std::vector<Symbol16>>& tree, const std::vector<uint>& treeSizes, const std::vector<uint>& branchFactors, uint level, uint index)
{
    uint bits = (level+1 >= tree.size() ? 8 : tree[level+1][index / branchFactors[level]]);
    uint value = 0;
    bitStream.readBits(value, bits);
    assert(level == 0 || value <= 32); // lengths must be <= 32
    tree[level].push_back(Symbol16(value));
    if(level > 0) {
        uint branchFactor = branchFactors[level - 1];
        uint childrenCount = std::min(branchFactor, treeSizes[level - 1] - index * branchFactor);
        for(uint i = 0; i < childrenCount; i++) {
            rbucDecodeCPUReadTree(bitStream, tree, treeSizes, branchFactors, level - 1, index * branchFactor + i);
        }
    }
}

bool rbucEncodeCPU(BitStream& bitStream, std::vector<uint>& offsets, const std::vector<Symbol16>& symbols, const std::vector<uint>& branchFactors)
{
    if(symbols.empty())
        return true;

    uint levelCount = (uint)branchFactors.size();



    std::vector<std::vector<Symbol16>> tree;
    tree.resize(levelCount + 2);
    tree.front() = symbols;

    std::vector<uint> treeBranchFactors = branchFactors;

    for(uint level = 0; level <= levelCount; level++) {
        if(level >= levelCount) treeBranchFactors.push_back((uint)tree[level].size());
        uint branchFactor = treeBranchFactors[level];
        rbucEncodeCPUBuildLengthsOneLevel(tree[level + 1], tree[level], branchFactor);
    }
    treeBranchFactors.push_back(1);

    rbucEncodeCPUWriteTree(bitStream, offsets, tree, treeBranchFactors, levelCount + 1, 0);



    //std::vector<std::vector<Symbol16>> lengths;
    //lengths.resize(levelCount + 1);

    //BitStream* pBitStreams = new BitStream[levelCount + 1];

    //const std::vector<Symbol16>* pIn = &symbols;
    //std::vector<Symbol16>* pOut = nullptr;

    //for(uint level = 0; level <= levelCount; level++) {
    //    pOut = &lengths[level];
    //    uint branchFactor = (level < levelCount) ? branchFactors[level] : (uint)pIn->size();
    //    rbucEncodeCPUOneLevel(pBitStreams[level], *pOut, *pIn, branchFactor);
    //    pIn = pOut;
    //}

    //assert(lengths.back()[0] <= 255);
    //byte lengthTopLevel = (byte)lengths.back()[0];
    //bitStream.writeAligned(&lengthTopLevel, 1);
    //for(uint i = levelCount + 1; i > 0; i--) {
    //    uint level = i - 1;
    //    uint bytes = pBitStreams[level].getBitSize() / 8;
    //    bitStream.writeAligned(pBitStreams[level].getRaw(), pBitStreams[level].getRawSizeUInts());
    //}

    //delete[] pBitStreams;



    return true;
}

bool rbucDecodeCPU(BitStreamReadOnly& bitStream, uint symbolCount, std::vector<Symbol16>& symbols, const std::vector<uint>& branchFactors)
{
    if(symbolCount == 0)
        return true;

    uint levelCount = (uint)branchFactors.size();



    std::vector<std::vector<Symbol16>> tree;
    tree.resize(levelCount + 2);

    std::vector<uint> treeSizes;
    treeSizes.resize(levelCount + 1);
    treeSizes.front() = symbolCount;
    for(uint i = 1; i <= levelCount; i++) {
        uint branchFactor = branchFactors[i - 1];
        treeSizes[i] = (treeSizes[i - 1] + branchFactor - 1) / branchFactor;
    }

    std::vector<uint> treeBranchFactors = branchFactors;
    treeBranchFactors.push_back(treeSizes[levelCount]);

    rbucDecodeCPUReadTree(bitStream, tree, treeSizes, treeBranchFactors, levelCount + 1, 0);

    symbols = tree.front();



    //std::vector<std::vector<Symbol16>> lengths;
    //lengths.resize(levelCount + 1);

    //byte lengthTopLevel;
    //bitStream.readAligned(&lengthTopLevel, 1);

    //lengths.back().push_back(lengthTopLevel);

    //std::vector<uint> symbolCounts;
    //symbolCounts.resize(levelCount + 1);
    //symbolCounts[0] = symbolCount;
    //for(uint level = 0; level < levelCount; level++) {
    //    symbolCounts[level + 1] = (symbolCounts[level] + branchFactors[level] - 1) / branchFactors[level];
    //}

    //for(uint i = levelCount + 1; i > 0; i--) {
    //    uint level = i - 1;
    //    std::vector<Symbol16>& symbolsOut = level >= 1 ? lengths[level-1] : symbols;
    //    uint branchFactor = (level < levelCount) ? branchFactors[level] : symbolCounts.back();
    //    bitStream.align<uint>();
    //    rbucDecodeCPUOneLevel(bitStream, lengths[level], symbolsOut, symbolCounts[level], branchFactor);
    //}



    return true;
}

}
