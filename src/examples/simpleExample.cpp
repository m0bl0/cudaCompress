#include <fstream>

#include <cuda_runtime.h>

#include <cudaCompress/Instance.h>
#include <cudaCompress/Encode.h>
#include <cudaCompress/util/DWT.h>
#include <cudaCompress/util/Quantize.h>

#include "tools/imgtools.h"

// Global resources shared by compressImage and decompressImage.
cudaCompress::Instance* pInstance = nullptr; // the cudaCompress Instance.
float* dpScratch = nullptr;                  // scratch buffer for DWT.
float* dpBuffer = nullptr;                   // output buffer for DWT.
cudaCompress::Symbol16* dpSymbols = nullptr; // input/output for entropy coder.


cudaCompress::BitStream compressImage(
    const unsigned char* dpImage,  // input image in GPU memory
    int sizeX, int sizeY,          // image size
    float quantStep)               // quantization step
{
    // Expand image values to float and do first-level DWT.
    cudaCompress::util::dwtFloat2DForwardFromByte(
        dpBuffer, dpScratch, dpImage, sizeX, sizeY);
    // Do second-level DWT in the same buffers. Need to specify pitch now!
    cudaCompress::util::dwtFloat2DForward(
        dpBuffer, dpScratch, dpBuffer,  sizeX/2, sizeY/2, 1, sizeX, sizeY);
    // dpBuffer now contains the multi-level DWT decomposition.

    // Quantize the coefficients and convert them to unsigned values (symbols).
    // For better compression, quantStep should be adapted to the transform level!
    cudaCompress::util::quantizeToSymbols2D(dpSymbols, dpBuffer, sizeX, sizeY, quantStep);

    // Run-length + Huffman encode the quantized coefficients.
    cudaCompress::BitStream bitStream;
    cudaCompress::encodeRLHuff(pInstance, bitStream, &dpSymbols, 1, sizeX * sizeY);
    return bitStream;
}

void decompressImage(
    cudaCompress::BitStream& bitStream, // compressed image data
    unsigned char* dpImage, int sizeX, int sizeY, float quantStep)
{
    cudaCompress::decodeRLHuff(pInstance, bitStream, &dpSymbols, 1, sizeX * sizeY);

    cudaCompress::util::unquantizeFromSymbols2D(dpBuffer, dpSymbols, sizeX, sizeY, quantStep);

    cudaCompress::util::dwtFloat2DInverse(
        dpBuffer, dpScratch, dpBuffer, sizeX/2, sizeY/2, 1, sizeX, sizeY);
    cudaCompress::util::dwtFloat2DInverseToByte(
        dpImage, dpScratch, dpBuffer, sizeX, sizeY);
}


void main2()
{
    int sizeX = 1024, sizeY = 1024;
    float quantStep = 4.0f;
    unsigned char* dpImage = nullptr;

    // Read image data from file.
    std::vector<unsigned char> image(sizeX * sizeY);
    std::ifstream file("image.raw", std::ifstream::binary);
    if(!file.good()) return;
    file.read((char*)image.data(), sizeX * sizeY);
    file.close();

    // Initialize cudaCompress, allocate GPU resources and upload data.
    pInstance = cudaCompress::createInstance(-1, 1, sizeX * sizeY);

    cudaMalloc(&dpImage, sizeX * sizeY);
    cudaMemcpy(dpImage, image.data(), sizeX * sizeY, cudaMemcpyHostToDevice);

    cudaMalloc(&dpScratch, sizeX * sizeY * sizeof(float));
    cudaMalloc(&dpBuffer, sizeX * sizeY * sizeof(float));
    cudaMalloc(&dpSymbols, sizeX * sizeY * sizeof(cudaCompress::Symbol16));

    // Compress the image.
    cudaCompress::BitStream bitStream = compressImage(dpImage, sizeX, sizeY, quantStep);

    // Write compression rate to stdout.
    int compressedSize = bitStream.getBitSize();
    float ratio = float(sizeX * sizeY * 8) / float(compressedSize);
    printf("Compressed size: %i b  (%.2f : 1)\n", compressedSize, ratio);

    // Rewind bitstream and decompress.
    bitStream.setBitPosition(0);
    decompressImage(bitStream, dpImage, sizeX, sizeY, quantStep);

    // Download reconstructed image and write to file.
    std::vector<unsigned char> imageReconst(sizeX * sizeY);
    cudaMemcpy(imageReconst.data(), dpImage, sizeX * sizeY, cudaMemcpyDeviceToHost);
    double psnr = computePSNR(image.data(), imageReconst.data(), sizeX * sizeY);
    printf("PSNR: %.2f\n", psnr);
    std::ofstream out("image_reconst.raw", std::ofstream::binary);
    out.write((char*)imageReconst.data(), sizeX * sizeY);
    out.close();

    // Cleanup.
    cudaFree(dpSymbols);
    cudaFree(dpBuffer);
    cudaFree(dpScratch);

    cudaFree(dpImage);

    cudaCompress::destroyInstance(pInstance);
}
