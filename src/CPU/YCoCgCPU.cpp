#include "YCoCgCPU.h"

#include <memory>

#include <cudaCompress/util.h>


namespace cudaCompress {

void convertRGBToYCoCgCPU(byte* pTarget, const byte* pData, int pixelCount)
{
    byte rgb[3];
    for(int i = 0; i < pixelCount; i++) {
        const byte* pRGB = pData + 3 * i;
        memcpy(rgb, pRGB, 3 * sizeof(byte));
        byte* pYCoCg = pTarget + 3 * i;
        pYCoCg[0] = byte(( int(rgb[0]) + int(rgb[1])*2 + int(rgb[2]) + 2) / 4      );
        pYCoCg[1] = byte(( int(rgb[0])                 - int(rgb[2]) + 1) / 2 + 127);
        pYCoCg[2] = byte((-int(rgb[0]) + int(rgb[1])*2 - int(rgb[2]) + 2) / 4 + 127);
    }
}

void convertYCoCgToRGBCPU(byte* pTarget, const byte* pData, int pixelCount)
{
    byte ycocg[3];
    for(int i = 0; i < pixelCount; i++) {
        const byte* pYCoCg = pData + 3 * i;
        memcpy(ycocg, pYCoCg, 3 * sizeof(byte));
        byte* pRGB = pTarget + 3 * i;
        pRGB[0] = (byte)min(max((int)ycocg[0] + ycocg[1] - ycocg[2]      , 0), 255);
        pRGB[1] = (byte)min(max((int)ycocg[0]            + ycocg[2] - 127, 0), 255);
        pRGB[2] = (byte)min(max((int)ycocg[0] - ycocg[1] - ycocg[2] + 254, 0), 255);
    }
}

}
