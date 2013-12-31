#ifndef __entropy_h__
#define __entropy_h__


#include <cassert>
#include <cmath>
#include <limits>
#include <vector>


template<typename T>
inline double computeEntropy(const T* pData, size_t count)
{
    static_assert(std::numeric_limits<T>::is_integer, "Input type must be integer");
    static_assert(!std::numeric_limits<T>::is_signed, "Input type must be unsigned");

    T max(0);
    for(size_t i = 0; i < count; i++) {
        if(pData[i] > max) max = pData[i];
    }
    size_t symbolCount = max + 1;

    std::vector<double> probabilities(symbolCount);
    for(size_t i = 0; i < count; i++) {
        probabilities[pData[i]]++;
    }

    double sum = 0.0;
    for(size_t i = 0; i < symbolCount; i++) {
        probabilities[i] /= count;

        if (probabilities[i] > 0.0) {
            sum -= probabilities[i] * log(probabilities[i]);
        }
    }

    return sum / log(2.0);
}


#endif
