# Image-Processing

This repository provides a comprehensive Python implementation for image compression using K-Means clustering and error correction techniques. It includes various methods for optimizing compressed image quality and handling potential transmission errors.

## Features

### Image Compression:
- Implements **Vector Quantization (VQ)** with K-Means clustering.
- Optimized using **Binary Switching Algorithm (BSA)** for better cluster label assignment.
- Pseudo-Gray coding support for efficient data representation.

### Error Simulation:
- Introduces random errors to simulate transmission issues.
- Binary Error Rate (BER) modeling included.

### Error Correction:
- **Median Filtering**: Corrects errors by applying median filters to the damaged image.
- **Binary Switching Algorithm (BSA)**: Optimizes label assignments to minimize distortion.

### Evaluation Metrics:
- Calculates **Compression Ratio** to measure storage efficiency.
- Computes **PSNR (Peak Signal-to-Noise Ratio)** for quality assessment of compressed and corrected images.

---

## Installation

### Prerequisites
Ensure the following libraries are installed:
- **Python 3.8+**
- `numpy`
- `matplotlib`
- `pillow`
- `scikit-image`
- `scipy`

### Install the required packages:
```bash
pip install numpy matplotlib pillow scikit-image scipy
```

---

## Usage

### Command-Line Interface
This script is designed to run via the command line. You can specify the mode, input file path, and parameters for clustering, error simulation, and correction.

### Command-Line Arguments

| Argument        | Type     | Description                                                                                  | Example                               |
|------------------|----------|----------------------------------------------------------------------------------------------|---------------------------------------|
| `--path`        | String   | Path to the input image file.                                                                | `--path input.jpg`                    |
| `--mode`        | String   | Mode of operation. Options: `compress`, `error_model`, `error_correction`, `bsa`.            | `--mode compress`                     |
| `--clusters`    | Integer  | Number of clusters for K-Means clustering.                                                   | `--clusters 4`                        |
| `--error`       | Float    | Error rate (0-1) for simulating transmission errors.                                         | `--error 0.05`                        |
| `--correct`     | String   | Error correction method. Options: `median`, `bsa`.                                           | `--correct median`                    |

---

## Examples

### Compress an Image
```bash
python main.py --path input.jpg --mode compress --clusters 4
```

### Simulate Errors
```bash
python main.py --path input.jpg --mode error_model --clusters 4 --error 0.05
```

### Correct Errors with Median Filtering
```bash
python main.py --path input.jpg --mode error_correction --clusters 4 --error 0.05 --correct median
```

### Use BSA for Compression
```bash
python main.py --path input.jpg --mode bsa --clusters 4
```

---

## Project Structure

```
├── main.py                 # Main script with compression and error correction functions
├── requirements.txt        # Required Python libraries
├── README.md               # Documentation
├── results/                # Output directory for compressed and corrected images
└── utils/                  # Utility functions (if separated for modularity)
```

---

## Metrics

- **Compression Ratio (CR):**
  \[
  CR = \frac{\text{Original Size (bits)}}{\text{Compressed Size (bits)}}
  \]

- **PSNR (Peak Signal-to-Noise Ratio):**
  \[
  PSNR = 10 \cdot \log_{10}\left(\frac{\text{Max Pixel Intensity}^2}{\text{MSE}}\right)
  \]
