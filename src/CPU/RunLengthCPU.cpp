#include "RunLengthCPU.h"

#include <cassert>


namespace cudaCompress {

bool runLengthZeroesEncodeCPU(std::vector<Symbol16>& symbolsCompact, std::vector<ushort>& zeroCounts, const std::vector<Symbol16>& symbols, uint zeroCountMax)
{
    uint zeroCountCur = 0;
    for(uint i = 0; i < symbols.size(); i++) {
        if(symbols[i] == 0) {
            zeroCountCur++;
            if(zeroCountCur > zeroCountMax) {
                symbolsCompact.push_back(0);
                zeroCounts.push_back(zeroCountCur - 1);
                zeroCountCur = 0;
            }
        } else {
            symbolsCompact.push_back(symbols[i]);
            zeroCounts.push_back(zeroCountCur);
            zeroCountCur = 0;
        }
    }

    // remove any trailing zero symbols (not needed for decoding)
    while(!symbolsCompact.empty() && symbolsCompact.back() == 0) {
        symbolsCompact.pop_back();
        zeroCounts.pop_back();
    }

    return true;
}

bool runLengthZeroesDecodeCPU(const std::vector<Symbol16>& symbolsCompact, const std::vector<ushort>& zeroCounts, std::vector<Symbol16>& symbols, uint symbolCount)
{
    assert(symbolsCompact.size() == zeroCounts.size());

    symbols.clear();
    symbols.resize(symbolCount);

    uint index = 0;
    for(uint i = 0; i < symbolsCompact.size(); i++) {
        index += zeroCounts[i];
        symbols[index] = symbolsCompact[i];
        index++;
    }

    return true;
}

bool runLengthEncodeCPU(std::vector<Symbol16>& runSymbols, std::vector<ushort>& runLengths, const std::vector<Symbol16>& symbols, uint runLengthMax)
{
    if(symbols.empty())
        return true;

    runSymbols.push_back(symbols[0]);
    uint runLengthCur = 0;
    for(uint i = 1; i < symbols.size(); i++) {
        if(symbols[i] == runSymbols.back()) {
            runLengthCur++;
            if(runLengthCur > runLengthMax) {
                runLengths.push_back(runLengthCur - 1);
                runLengthCur = 0;
                runSymbols.push_back(runSymbols.back());
            }
        } else {
            runLengths.push_back(runLengthCur);
            runLengthCur = 0;
            runSymbols.push_back(symbols[i]);
        }
    }
    runLengths.push_back(runLengthCur);

    //// remove any trailing zero symbols (not needed for decoding)
    //while(!symbolsCompact.empty() && symbolsCompact.back() == 0) {
    //    symbolsCompact.pop_back();
    //    zeroCounts.pop_back();
    //}

    return true;
}

bool runLengthDecodeCPU(const std::vector<Symbol16>& runSymbols, const std::vector<ushort>& runLengths, std::vector<Symbol16>& symbols, uint symbolCount)
{
    assert(runSymbols.size() == runLengths.size());

    symbols.clear();
    symbols.resize(symbolCount);

    uint index = 0;
    for(uint i = 0; i < runSymbols.size(); i++) {
        for(uint j = 0; j <= runLengths[i]; j++, index++) {
            symbols[index] = runSymbols[i];
        }
    }

    return true;
}

}
