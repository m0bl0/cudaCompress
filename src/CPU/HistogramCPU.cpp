#include "HistogramCPU.h"


namespace cudaCompress {

void histogramCPU(uint* pHistogram, const ushort* pData, uint elemCount, uint binCount)
{
    for(uint i = 0; i < binCount; i++)
        pHistogram[i] = 0;

    for(uint i = 0; i < elemCount; i++){
        ushort data = pData[i];
        if(data < binCount)
            pHistogram[data]++;
    }
}

}