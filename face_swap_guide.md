
# Batch Face Swapping with `facefusion.py`

This guide provides instructions on how to use the provided Python script to batch process face swapping tasks using `facefusion.py`. The script automates the creation and execution of face swapping jobs for multiple image pairs.

## Table of Contents

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Setting Up the Environment](#setting-up-the-environment)
- [Script Overview](#script-overview)
- [Running the Script](#running-the-script)
  - [Script Usage](#script-usage)
  - [Command-Line Arguments](#command-line-arguments)
- [Example Usage](#example-usage)
- [Notes and Tips](#notes-and-tips)
- [Troubleshooting](#troubleshooting)
- [Conclusion](#conclusion)

## Introduction

The script automates face swapping between multiple images using the `facefusion.py` tool. It creates a job, adds steps for each pair of images (excluding self-pairs), submits the job, and runs it using the specified execution provider (e.g., CUDA).

By using this script, you can efficiently process face swapping tasks on a batch of images without manually creating and submitting jobs for each pair.

## Prerequisites

Before running the script, ensure that you have the following:

- **Python 3.6 or higher** installed on your system.
- The `facefusion.py` script available and accessible.
- Necessary models for face swapping and enhancement:
  - Face Swapper Model: `inswapper_128_fp16`
  - Face Enhancer Model: `gfpgan_1.4`
- Required Python packages:
  - `os`
  - `glob`
  - `subprocess`
  - `argparse`
  - `uuid`

> **Note:** The script assumes that `facefusion.py` and the models are properly set up and configured.

## Setting Up the Environment

1. **Clone or Download the `facefusion.py` Repository:**

   If you haven't already, clone or download the repository containing `facefusion.py` to your local machine.

   ```bash
   git clone https://github.com/yourusername/facefusion.git
   ```

2. **Install Required Python Packages:**

   Ensure all required packages are installed. If you're using a virtual environment, activate it first.

   ```bash
   pip install -r requirements.txt
   ```

   If there is no `requirements.txt`, you can install packages individually:

   ```bash
   pip install argparse uuid
   ```

3. **Download and Place the Models:**

   Download the face swapper and face enhancer models and place them in the appropriate directories as expected by `facefusion.py`.

## Script Overview

The script performs the following steps:

1. **Parses Command-Line Arguments:**

   - `--input_directory`: Directory containing input images.
   - `--output_directory`: Directory to save output images.
   - `--num_images`: Number of images to process (default is 5).

2. **Prepares the Image List:**

   - Collects and filters image files from the input directory.
   - Ensures there are enough images to process.

3. **Creates a Unique Job ID:**

   - Generates a UUID-based job ID for tracking.

4. **Creates a New Drafted Job:**

   - Uses `facefusion.py job-create` to create a new job.

5. **Adds Steps to the Job:**

   - For each pair of images (excluding self-pairs), adds a step to the job with the specified processors and models.

6. **Submits the Job:**

   - Uses `facefusion.py job-submit` to submit the job to the queue.

7. **Runs the Job:**

   - Uses `facefusion.py job-run` to execute the job with the specified execution provider.

## Running the Script

### Script Usage

Save the script to a file, for example, `batch_face_swap.py`. Make sure it's in the same directory as `facefusion.py` or adjust the `facefusion_script` variable accordingly.

To run the script, use the following command:

```bash
python batch_face_swap.py --input_directory <input_dir> --output_directory <output_dir> [--num_images N]
```

### Command-Line Arguments

- `--input_directory`: **(Required)** Path to the directory containing input images.

- `--output_directory`: **(Required)** Path to the directory where output images will be saved.

- `--num_images`: *(Optional)* Number of images to process. Default is 5.

**Example:**

```bash
python batch_face_swap.py --input_directory ./input_images --output_directory ./output_images --num_images 10
```

## Example Usage

1. **Prepare Input Images:**

   - Place your source images in a directory, e.g., `./input_images`.
   - Supported image formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.gif`.

2. **Run the Script:**

   ```bash
   python batch_face_swap.py --input_directory ./input_images --output_directory ./output_images
   ```

   This will process up to 5 images by default.

3. **Specify Number of Images:**

   If you have more images and want to process them, use the `--num_images` argument.

   ```bash
   python batch_face_swap.py --input_directory ./input_images --output_directory ./output_images --num_images 10
   ```

4. **Check Output:**

   - Processed images will be saved in `./output_images`.
   - Each output image is named in the format: `output_<source>_to_<target>.<extension>`.

## Notes and Tips

- **Execution Provider:**

  The script uses `cuda` as the execution provider for GPU acceleration. Ensure that your system has a compatible NVIDIA GPU and CUDA installed.

- **Adjusting Models:**

  If you wish to use different models for face swapping or enhancement, modify the `face_swapper_model` and `face_enhancer_model` variables in the script.

- **Excluding Self-Pairs:**

  The script skips processing image pairs where the source and target are the same.

- **Output Directory:**

  The script creates the output directory if it doesn't exist.

- **Job ID:**

  A unique job ID is generated using UUID to prevent conflicts.

## Troubleshooting

- **Not Enough Images Error:**

  If you receive the error:

  ```
  Not enough images in the directory. Found X, but expected Y.
  ```

  Ensure that the input directory contains enough valid image files.

- **Facefusion.py Errors:**

  If any of the subprocess calls to `facefusion.py` fail, the script will output the error message.

  - Check that `facefusion.py` is accessible and executable.
  - Ensure that all required models and dependencies for `facefusion.py` are properly installed.

- **CUDA Errors:**

  If you encounter errors related to CUDA:

  - Verify that CUDA is installed and the version is compatible with your GPU.
  - Check that your GPU drivers are up to date.

- **Permission Issues:**

  If the script cannot read from the input directory or write to the output directory, check the permissions.

## Conclusion

This script automates the batch processing of face swapping tasks using `facefusion.py`. By following this guide, you should be able to set up the environment and run the script to generate face-swapped images efficiently.

For further customization or enhancement, consider modifying the script to suit your specific needs.

> **Disclaimer:** Ensure that you have the rights and permissions to use the images for face swapping and that you comply with any applicable laws and ethical guidelines.
