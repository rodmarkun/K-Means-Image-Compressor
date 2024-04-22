# K-Means Image Compressor

![imagen](https://github.com/rodmarkun/K-Means-Image-Compressor/assets/75074498/6ad89b44-61f8-4cb1-85b5-d4227fd205d9)

## Overview

The K-Means Image Compressor is a straightforward Flask web application that employs the K-Means clustering algorithm for image compression. This tool allows users to reduce the color palette of images, thereby compressing them. In addition to providing a means for compression, the application also offers visualizations to help users understand the clustering process and the impact on the compressed image. A selection of sample images is included for quick testing.

## Setup

### Prerequisites

To run this application, you'll need Python 3.x and Git installed on your computer.

### Installation

Use Git to clone the project's repository to your local machine.

```bash
git clone https://github.com/rodmarkun/K-Means-Image-Compressor && cd K-Means-Image-Compressor/
```

Install the necessary Python packages.

```bash
pip install -r ./requirements.txt
```

### Usage

To run the web server, execute the following command:

```bash
python app.py
```

Then, navigate to http://127.0.0.1:5000/ in your web browser. You can upload a PNG image to compress and view the compression results and various generated plots. Please take into account that, the larger the image to compress, the higher the load on your CPU will be. For a standard CPU it is recommended not to upload >4Mpx PNG images (2000x2000px) to avoid long waiting times.

## How does it work?

The application uses the K-Means clustering algorithm to analyze the colors in an image and group them into clusters. This process reduces the overall number of colors in the image, which in turn compresses the image. The application provides plots that show how the algorithm groups colors and the effect on the final image. This straightforward approach offers a glimpse into how unsupervised learning algorithms can be applied for practical purposes like image compression.
