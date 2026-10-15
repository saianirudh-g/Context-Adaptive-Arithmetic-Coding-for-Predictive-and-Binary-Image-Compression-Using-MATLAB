# Context-Adaptive Arithmetic Coding for Predictive and Binary Image Compression Using MATLAB

## Overview

This project implements a complete image compression framework using predictive coding, Exp-Golomb binarization, adaptive binary arithmetic coding, quantization, image reconstruction, PSNR evaluation, and context-based coding.

The project is implemented in MATLAB and uses the Lena image as the test image.

The main goal is to study how predictive residual coding and context-adaptive arithmetic coding can reduce the number of bits required to represent an image while maintaining acceptable reconstruction quality.

The project also investigates the relationship between:

- Quantization level
- Bit rate
- Reconstruction quality
- Context modeling
- Binary image compression efficiency

---

## Project Objectives

The main objectives of this project are:

1. Convert a color image into grayscale.
2. Perform intra-prediction using neighboring pixels.
3. Generate the prediction residual image.
4. Quantize the prediction residual using different quantization values.
5. Convert signed residual values into binary representation using Exp-Golomb coding.
6. Compress the Exp-Golomb bitstream using adaptive binary arithmetic coding.
7. Reconstruct the image from the quantized prediction residual.
8. Measure reconstruction quality using PSNR.
9. Implement a context-based arithmetic coding model for improved compression.
10. Convert the grayscale image into a binary image and compress it using an 8-context arithmetic coding scheme.

---

## System Architecture

The overall compression process is:

```text
Input Image
     |
     v
RGB to Grayscale Conversion
     |
     v
Predictive Coding
     |
     v
Prediction Residual
     |
     v
Quantization
     |
     v
Exp-Golomb Binarization
     |
     v
Adaptive Binary Arithmetic Coding
     |
     v
Compressed Bitstream
