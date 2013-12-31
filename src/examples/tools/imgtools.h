#ifndef __imgtools_h__
#define __imgtools_h__


#include <algorithm>
#include <cmath>
#include <limits>


template<typename T>
T* computeLuma(const T* pData, unsigned int count, unsigned int numcomps)
{
    if(numcomps < 3)
        return 0;

    T* pResult = new T[count];
    for(unsigned int i = 0; i < count; i++) {
        double r = double(pData[i * numcomps + 0]);
        double g = double(pData[i * numcomps + 1]);
        double b = double(pData[i * numcomps + 2]);

        double y = 0.2126 * r + 0.7152 * g + 0.0722 * b;

        pResult[i] = T(y);
    }

    return pResult;
}


template<typename T>
double computeRange(const T* pData, unsigned int count, unsigned int numcomps = 1, unsigned int comp = 0)
{
    double min =  std::numeric_limits<double>::max();
    double max = -std::numeric_limits<double>::max();

    for(unsigned int i = 0; i < count; i++) {
        double val = double(pData[i * numcomps + comp]);
        if(val < min)
            min = val;
        if(val > max)
            max = val;
    }

    return max - min;
}

template<typename T>
double computeAverage(const T* pData, unsigned int count, unsigned int numcomps = 1, unsigned int comp = 0)
{
    double sum = 0.0;
    for(unsigned int i = 0; i < count; i++) {
        sum += double(pData[i * numcomps + comp]);
    }

    return sum / double(count);
}

template<typename T>
double computeVariance(const T* pData, unsigned int count, unsigned int numcomps = 1, unsigned int comp = 0)
{
    double sum = 0.0;
    double sumSq = 0.0;
    for(unsigned int i = 0; i < count; i++) {
        double val = double(pData[i * numcomps + comp]);
        sum   += val;
        sumSq += val * val;
    }
    double avg   = sum   / double(count);
    double avgSq = sumSq / double(count);

    return avgSq - avg * avg;
}

template<typename T>
double computeAvgError(const T* pData, const T* pReconst, unsigned int count, unsigned int numcomps = 1, unsigned int comp = 0)
{
    double result = 0.0;
    for(unsigned int i = 0; i < count; i++) {
        double diff = double(pReconst[i * numcomps + comp]) - double(pData[i * numcomps + comp]);
        result += diff;
    }
    result /= double(count);

    return result;
}

template<typename T>
double computeAvgAbsError(const T* pData, const T* pReconst, unsigned int count, unsigned int numcomps = 1, unsigned int comp = 0)
{
    double result = 0.0;
    for(unsigned int i = 0; i < count; i++) {
        double diff = double(pReconst[i * numcomps + comp]) - double(pData[i * numcomps + comp]);
        result += abs(diff);
    }
    result /= double(count);

    return result;
}

template<typename T>
double computeMaxAbsError(const T* pData, const T* pReconst, unsigned int count, unsigned int numcomps = 1, unsigned int comp = 0)
{
    double result = 0.0;
    for(unsigned int i = 0; i < count; i++) {
        double diff = double(pReconst[i * numcomps + comp]) - double(pData[i * numcomps + comp]);
        result = std::max(result, abs(diff));
    }

    return result;
}

template<typename T>
double computeRMSError(const T* pData, const T* pReconst, unsigned int count, unsigned int numcomps = 1, unsigned int comp = 0)
{
    double result = 0.0;
    for(unsigned int i = 0; i < count; i++) {
        double diff = double(pData[i * numcomps + comp]) - double(pReconst[i * numcomps + comp]);
        result += diff * diff;
    }
    result /= double(count);
    result = sqrt(result);

    return result;
}

template<typename T>
double computeSNR(const T* pData, const T* pReconst, unsigned int count, unsigned int numcomps = 1, unsigned int comp = 0)
{
    double var = computeVariance(pData, count, numcomps, comp);
    double rmse = computeRMSError(pData, pReconst, count, numcomps, comp);

    return 20.0 * log10(sqrt(var) / rmse);
}

template<typename T>
double computePSNR(const T* pData, const T* pReconst, unsigned int count, unsigned int numcomps = 1, unsigned int comp = 0)
{
    double range = computeRange(pData, count, numcomps, comp);
    double rmse = computeRMSError(pData, pReconst, count, numcomps, comp);

    return 20.0 * log10(range / rmse);
}

// normalized cross-correlation
template<typename T>
double computeNCC(const T* pData, const T* pReconst, unsigned int count, unsigned int numcomps = 1, unsigned int comp = 0)
{
    double avgData    = computeAverage(pData, count, numcomps, comp);
    double varData    = computeVariance(pData, count, numcomps, comp);
    double avgReconst = computeAverage(pReconst, count, numcomps, comp);
    double varReconst = computeVariance(pReconst, count, numcomps, comp);

    double ncc = 0.0;
    for(unsigned int i = 0; i < count; i++) {
        ncc += (double(pData[i * numcomps + comp]) - avgData) * (double(pReconst[i * numcomps + comp]) - avgReconst);
    }

    ncc /= double(count - 1) * sqrt(varData) * sqrt(varReconst);

    return ncc;
}


#endif
