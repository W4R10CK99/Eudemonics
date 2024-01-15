# OCR Attribute Finder

This project utilizes Tesseract OCR and OpenAI's GPT-3.5-turbo model to extract attributes and values from images with different page segmentation modes.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)

## Introduction

The OCR Attribute Finder project combines Tesseract OCR with OpenAI's GPT-3.5-turbo model to extract attributes and values from images using various page segmentation modes. This tool is particularly useful for processing images containing text with different layouts.

## Installation

1. Clone the repository:

   ```
   git clone https://github.com/yourusername/ocr-attribute-finder.git
   cd ocr-attribute-finder
   ```
2. Install the required dependencies:
   ```
    pip install pytesseract
    pip install pillow
    pip install openai  # Ensure you have OpenAI API credentials
   ```
3. Download and install Tesseract OCR from https://github.com/tesseract-ocr/tesseract. Make sure to add the Tesseract binary to your system's PATH.

## Usage
1. Run the script by providing the path to the image file:
   ```
   python ocr_attribute_finder.py img1.png
   ```
   This will use Tesseract OCR to extract text from the image and then send a user message to OpenAI's GPT-3.5-turbo model to find attributes and values in the extracted        text.
2. Adjust the page segmentation mode in the script as needed. The default mode is set to 6.

## Contributing

If you would like to contribute to this project, follow these steps:

Fork the repository.
1. Create a new branch for your feature or bug fix.
2. Make your changes and submit a pull request.
3. Please follow our Contribution Guidelines for more details.

