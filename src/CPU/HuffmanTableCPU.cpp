#include "HuffmanTableCPU.h"

#include <cassert>
#include <queue>

#include <cudaCompress/HuffmanDesign.h>
#include <cudaCompress/util.h>


namespace cudaCompress {

// FIXME ...
static const uint LOG2_MAX_SYMBOL_BITS = 14;


void HuffmanTableCPU::clear()
{
    m_codewordCountPerLength.clear();
    m_symbols.clear();
}

void HuffmanTableCPU::design(const std::vector<Symbol16>& symbolStream)
{
    clear();

    // find max symbol
    Symbol16 symbolMax = 0;
    for(uint i = 0; i < symbolStream.size(); i++) {
        if(symbolStream[i] > symbolMax) {
            symbolMax = symbolStream[i];
        }
    }

    uint distinctSymbolCount = symbolMax + 1;

    // find symbol probabilities
    std::vector<uint> symbolProbabilities(distinctSymbolCount);
    for(uint i = 0; i < symbolStream.size(); i++) {
        symbolProbabilities[symbolStream[i]]++;
    }

    std::vector<HuffmanTreeNode> huffmanNodes(2 * distinctSymbolCount - 1);
    uint nextNodeIndex = 0;

    // build list of all used symbols, packed in HuffmanTreeNodes
    // these will be the leaves of the huffman tree
    std::vector<HuffmanTreeNode*> treeLeaves;
    for(uint symbol = 0; symbol < distinctSymbolCount; symbol++) {
        if(symbolProbabilities[symbol] > 0) {
            huffmanNodes[nextNodeIndex].init(symbol, symbolProbabilities[symbol]);
            treeLeaves.push_back(&huffmanNodes[nextNodeIndex]);
            nextNodeIndex++;
        }
    }

    if(treeLeaves.empty())
        return;

    // list of huffman nodes to process
    std::priority_queue<HuffmanTreeNode*, std::vector<HuffmanTreeNode*>, HuffmanTreeNodeProbabilityInvComparer> treeNodesTodo;
    for(uint i = 0; i < treeLeaves.size(); i++)
        treeNodesTodo.push(treeLeaves[i]);

    // build the huffman tree by successively combining the lowest-probability nodes
    while(treeNodesTodo.size() > 1) {
        uint newNodeIndex = nextNodeIndex++;
        HuffmanTreeNode& newNode = huffmanNodes[newNodeIndex];

        newNode.init(INVALID_SYMBOL16, 0);

        // get nodes with lowest probability as children
        HuffmanTreeNode* pLeftChild = treeNodesTodo.top();
        treeNodesTodo.pop();
        HuffmanTreeNode* pRightChild = treeNodesTodo.top();
        treeNodesTodo.pop();

        newNode.m_pLeftChild = pLeftChild;
        newNode.m_pRightChild = pRightChild;

        // combine probabilities
        newNode.m_probability = pLeftChild->m_probability + pRightChild->m_probability;

        // insert into todo list
        treeNodesTodo.push(&newNode);
    }

    HuffmanTreeNode& rootNode = *treeNodesTodo.top();

    // assign codeword length = tree level
    int codewordLengthMax = rootNode.assignCodewordLength(0);

    // sort leaves (ie actual symbols) by codeword length
    std::sort(treeLeaves.begin(), treeLeaves.end(), HuffmanTreeNodeCodewordLengthComparer());

    // fill codeword count list and symbol list from leaves list
    m_codewordCountPerLength.resize(codewordLengthMax);
    m_symbols.resize(treeLeaves.size());
    for(uint i = 0; i < treeLeaves.size(); i++) {
        const HuffmanTreeNode& node = *treeLeaves[i];

        if(node.m_codewordLength > 0)
            m_codewordCountPerLength[node.m_codewordLength - 1]++;
        m_symbols[i] = node.m_symbol;
    }

}

void HuffmanTableCPU::writeToBitStream(BitStream& bitstream) const
{
    // find max symbol
    Symbol16 symbolMax = 0;
    for(uint i = 0; i < m_symbols.size(); i++) {
        symbolMax = max(symbolMax, m_symbols[i]);
    }

    // compute needed bits per symbol
    uint symbolBits = 0;
    while(symbolMax > 0) {
        symbolMax >>= 1;
        symbolBits++;
    }

    const uint log2HuffmanDistinctSymbolCountMax = 14;

    // write
    bitstream.writeBits(uint(m_codewordCountPerLength.size()), LOG2_MAX_CODEWORD_BITS);
    for(uint i = 0; i < m_codewordCountPerLength.size(); i++) {
        bitstream.writeBits(m_codewordCountPerLength[i], log2HuffmanDistinctSymbolCountMax);
    }
    bitstream.writeBits(uint(m_symbols.size()), log2HuffmanDistinctSymbolCountMax);
    bitstream.writeBits(symbolBits, LOG2_MAX_SYMBOL_BITS);
    for(uint i = 0; i < m_symbols.size(); i++) {
        bitstream.writeBits(m_symbols[i], symbolBits);
    }
}

void HuffmanTableCPU::readFromBitStream(BitStreamReadOnly& bitstream)
{
    clear();

    const uint log2HuffmanDistinctSymbolCountMax = 14;

    uint codewordCountPerLengthSize = 0;
    bitstream.readBits(codewordCountPerLengthSize, LOG2_MAX_CODEWORD_BITS);
    for(uint i = 0; i < codewordCountPerLengthSize; i++) {
        uint codewordCount = 0;
        bitstream.readBits(codewordCount, log2HuffmanDistinctSymbolCountMax);
        m_codewordCountPerLength.push_back(codewordCount);
    }
    uint symbolsSize = 0;
    uint symbolBits = 0;
    bitstream.readBits(symbolsSize, log2HuffmanDistinctSymbolCountMax);
    bitstream.readBits(symbolBits, LOG2_MAX_SYMBOL_BITS);
    for(uint i = 0; i < symbolsSize; i++) {
        uint symbol = 0;
        bitstream.readBits(symbol, symbolBits);
        m_symbols.push_back(symbol);
    }
}

void HuffmanDecodeTableCPU::clear()
{
    m_decodeTable.clear();

    m_codewordMinPerLength.clear();
    m_codewordMaxPerLength.clear();
    m_codewordFirstIndexPerLength.clear();
}

void HuffmanDecodeTableCPU::build(const HuffmanTableCPU& table)
{
    clear();

    if(table.m_symbols.empty())
        return;

    // count total number of codewords
    uint codewordCount = 0;
    for(uint i = 0; i < table.m_codewordCountPerLength.size(); i++) {
        codewordCount += table.m_codewordCountPerLength[i];
    }
    if(table.m_codewordCountPerLength.empty()) {
        // this can happen when all symbols are the same -> only a single "codeword" with length 0
        codewordCount++;
    }

    assert(codewordCount == table.m_symbols.size());

    // alloc m_decodeTable (indexed by codeword index)
    m_decodeTable.resize(codewordCount);

    // copy symbols into m_decodeTable
    for(uint entry = 0; entry < m_decodeTable.size(); entry++) {
        m_decodeTable[entry].m_symbol = table.m_symbols[entry];
    }

    // find codeword lengths
    uint codewordIndexLocal = 0; // within current length
    uint codewordLengthCur = 1;
    if(!table.m_codewordCountPerLength.empty()) {
        for(uint entry = 0; entry < m_decodeTable.size(); entry++, codewordIndexLocal++) {
            // exhausted all codewords of this length?
            while(codewordIndexLocal >= table.m_codewordCountPerLength[codewordLengthCur - 1]) {
                codewordLengthCur++;
                codewordIndexLocal = 0;
            }

            m_decodeTable[entry].m_codewordLength = codewordLengthCur;
        }
    }

    // find codewords
    int codeword = 0;
    for(uint entry = 0;;) {
        // assign codeword
        m_decodeTable[entry].m_codeword = codeword;
        codeword++;
        entry++;

        // done?
        if(entry >= m_decodeTable.size())
            break;

        // if next codeword is longer, make room on the right for next bit(s)
        if(m_decodeTable[entry].m_codewordLength > m_decodeTable[entry-1].m_codewordLength) {
            uint lengthDiff = m_decodeTable[entry].m_codewordLength - m_decodeTable[entry-1].m_codewordLength;
            codeword <<= lengthDiff;
        }
    }

    // build indices (by codeword length) into table
    uint codewordMaxLength = uint(table.m_codewordCountPerLength.size());
    m_codewordFirstIndexPerLength.resize(codewordMaxLength);
    m_codewordMinPerLength.resize(codewordMaxLength);
    m_codewordMaxPerLength.resize(codewordMaxLength);
    // loop over codeword lengths (actually (length-1))
    for(uint codewordLength = 0, entry = 0; codewordLength < codewordMaxLength; codewordLength++) {
        if(table.m_codewordCountPerLength[codewordLength] > 0) {
            // current entry is first codeword of this length
            m_codewordFirstIndexPerLength[codewordLength] = entry;
            // store value of first codeword of this length
            m_codewordMinPerLength[codewordLength] = m_decodeTable[entry].m_codeword;
            // move to last codeword of this length
            entry += table.m_codewordCountPerLength[codewordLength] - 1;
            // store value of last codeword of this length
            m_codewordMaxPerLength[codewordLength] = m_decodeTable[entry].m_codeword;
            // move to first codeword of next length
            entry++;
        } else {
            m_codewordFirstIndexPerLength[codewordLength] = -1;
            m_codewordMinPerLength[codewordLength] = -1;
            m_codewordMaxPerLength[codewordLength] = -1;
        }
    }
}

void HuffmanEncodeTableCPU::clear()
{
    m_encodeTable.clear();
}

void HuffmanEncodeTableCPU::build(const HuffmanDecodeTableCPU& decodeTable)
{
    clear();

    // find max symbol
    Symbol16 symbolMax = 0;
    for(uint i = 0; i < decodeTable.m_decodeTable.size(); i++) {
        symbolMax = max(symbolMax, decodeTable.m_decodeTable[i].m_symbol);
    }

    // alloc m_encodeTable (indexed by symbol)
    m_encodeTable.resize(symbolMax + 1);

    // fill m_encodeTable with symbols (and invalid lengths)
    for(uint entry = 0; entry < m_encodeTable.size(); entry++) {
        // entry == symbol
        m_encodeTable[entry].m_symbol = entry;
        m_encodeTable[entry].m_codewordLength = -1;
    }

    // copy codewords and lengths from decodeTable
    for(uint entry = 0; entry < decodeTable.m_decodeTable.size(); entry++) {
        const HuffmanEntry& decodeEntry = decodeTable.m_decodeTable[entry];
        HuffmanEntry& encodeEntry = m_encodeTable[decodeEntry.m_symbol];

        encodeEntry.m_codeword       = decodeEntry.m_codeword;
        encodeEntry.m_codewordLength = decodeEntry.m_codewordLength;
    }
}

}
