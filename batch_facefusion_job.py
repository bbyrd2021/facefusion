import os
import glob
import subprocess
import argparse
import uuid  # For UUID-based job_id generation
import tempfile  # For temporary files
import cv2

# Import face detection functions from facefusion module
from facefusion.face_detector import detect_faces, pre_check, clear_inference_pool
from facefusion.state_manager import set_item

# Parameters for models and execution
facefusion_script = 'facefusion.py'  # Path to the facefusion script
face_swapper_model = 'inswapper_128_fp16'
face_enhancer_model = 'gfpgan_1.4'
execution_provider = 'cuda'    # Use CUDA for execution


def has_face(image_path):
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image {image_path}")
        return False
    # Call detect_faces
    bounding_boxes, face_scores, face_landmarks_5 = detect_faces(img)
    return len(bounding_boxes) > 0


def resize_and_save_image(image_path, resize_dimensions):
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image {image_path}")
        return None
    # Resize the image
    resized_img = cv2.resize(img, resize_dimensions)
    # Save to a temporary file
    temp_dir = tempfile.gettempdir()
    temp_filename = f"{uuid.uuid4().hex}_{os.path.basename(image_path)}"
    temp_path = os.path.join(temp_dir, temp_filename)
    cv2.imwrite(temp_path, resized_img)
    return temp_path


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Batch process face swapping with facefusion.')
    parser.add_argument('--input_directory', required=True, help='Directory containing images to process.')
    parser.add_argument('--output_directory', required=True, help='Directory to save output images.')
    parser.add_argument('--num_images', type=int, default=5, help='Number of images to process (default: 5).')
    parser.add_argument('--resize', type=str, help='Resize dimensions in format WIDTHxHEIGHT (e.g., 512x512).')

    args = parser.parse_args()

    input_directory = args.input_directory
    output_directory = args.output_directory
    num_images = args.num_images

    # Parse resize dimensions
    resize_dimensions = None
    if args.resize:
        try:
            width, height = map(int, args.resize.lower().split('x'))
            resize_dimensions = (width, height)
        except ValueError:
            print("Invalid resize dimensions. Use WIDTHxHEIGHT format (e.g., 512x512).")
            exit(1)

    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Get the list of images from the input directory
    image_paths = sorted(glob.glob(os.path.join(input_directory, '*')))
    image_paths = [path for path in image_paths if os.path.isfile(path)]  # Ensure it's a file

    # Filter image files by common image extensions
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    image_paths = [path for path in image_paths if path.lower().endswith(valid_extensions)]

    # Take the first N images
    image_paths = image_paths[:num_images]

    # Ensure we have enough images
    if len(image_paths) < num_images:
        print(f"Not enough images in the directory. Found {len(image_paths)}, but expected {num_images}.")
        exit(1)

    # Resize images if resize_dimensions is set
    if resize_dimensions:
        resized_image_paths = {}
        for image_path in image_paths:
            resized_path = resize_and_save_image(image_path, resize_dimensions)
            if resized_path:
                resized_image_paths[image_path] = resized_path
            else:
                print(f"Skipping image {image_path} due to resize failure.")
                image_paths.remove(image_path)
    else:
        resized_image_paths = {image_path: image_path for image_path in image_paths}

    # Initialize face detection
    # Set the required state items
    set_item('face_detector_model', 'yoloface')  # can use 'retinaface' too
    set_item('face_detector_size', '640x640')
    set_item('face_detector_score', 0.5)
    set_item('execution_providers', ['CUDAExecutionProvider'])

    # Perform pre-check to download models if necessary
    if not pre_check():
        print("Failed to download or verify face detection models.")
        exit(1)

    # Generate a unique job_id using UUID
    job_id = f"job_{uuid.uuid4().hex}"
    print(f"Creating job with ID {job_id}")

    # Step 1: Create a new drafted job with the job_id
    cmd_create = [
        'python', facefusion_script, 'job-create',
        job_id
    ]
    result_create = subprocess.run(cmd_create, capture_output=True, text=True)
    if result_create.returncode != 0:
        print(f"Error creating job {job_id}:")
        print(result_create.stderr)
        exit(1)

    print(f"Created job with ID {job_id}")

    # Size of output images
    output_image_resolution = '512x512'

    # For debugging purposes
    log_level = 'debug'

    # Step 2: Add steps to the drafted job for each image pair
    for source_image in image_paths:
        source_basename = os.path.basename(source_image)
        for target_image in image_paths:
            target_basename = os.path.basename(target_image)
            # Exclude self-pairs
            if os.path.abspath(source_image) == os.path.abspath(target_image):
                continue  # Skip self-pairs

            # Use resized images
            source_image_resized = resized_image_paths[source_image]
            target_image_resized = resized_image_paths[target_image]

            # Check for faces in both images
            if not has_face(source_image_resized):
                print(f"No face detected in source image {source_basename}. Skipping.")
                continue
            if not has_face(target_image_resized):
                print(f"No face detected in target image {target_basename}. Skipping.")
                continue

            # Get the target image's extension
            target_extension = os.path.splitext(target_basename)[1]

            # Generate the output filename with the same extension as the target image
            output_filename = f'output_{os.path.splitext(source_basename)[0]}_to_{os.path.splitext(target_basename)[0]}{target_extension}'
            output_path = os.path.join(output_directory, output_filename)

            # Add a step to the job
            cmd_add_step = [
                'python', facefusion_script, 'job-add-step',
                job_id,  # Provide the job_id as a positional argument
                '--source-paths', source_image_resized,
                '--target-path', target_image_resized,
                '--output-path', output_path,
                '--processors', 'face_swapper', 'face_enhancer',
                '--face-swapper-model', face_swapper_model,
                # '--face-enhancer-model', face_enhancer_model,
                '--output-image-resolution', output_image_resolution,
            ]
            result_add_step = subprocess.run(cmd_add_step, capture_output=True, text=True)
            if result_add_step.returncode != 0:
                print(f"Error adding step to job {job_id} for {source_basename} -> {target_basename}:")
                print(result_add_step.stderr)
            else:
                print(f"Added step to job {job_id} for {source_basename} -> {target_basename}")

    # Step 3: Submit the job to the queue
    cmd_submit = [
        'python', facefusion_script, 'job-submit',
        job_id,  # Provide the job_id as a positional argument
        '--log-level', log_level
    ]

    result_submit = subprocess.run(cmd_submit, capture_output=True, text=True)
    if result_submit.returncode != 0:
        print(f"Error submitting job {job_id}:")
        print(result_submit.stderr)
        exit(1)
    print(f"Submitted job {job_id}")

    # Step 4: Run the queued job with execution provider
    cmd_run = [
        'python', facefusion_script, 'job-run',
        job_id,  # Provide the job_id as a positional argument
        '--execution-providers', execution_provider,
        '--log-level', log_level
    ]

    print(f"Running command: {' '.join(cmd_run)}")
    
    try:
        with open('facefusion_stdout.log', 'w') as stdout_file, open('facefusion_stderr.log', 'w') as stderr_file:
            result_run = subprocess.run(
                cmd_run,
                stdout=stdout_file,
                stderr=stderr_file,
                text=True
            )

        print(f"Subprocess exited with return code: {result_run.returncode}")

        # Check the return code and handle errors
        if result_run.returncode != 0:
            print(f"\nError running job {job_id}. Command returned non-zero exit code: {result_run.returncode}")
            print("\nReading stderr log content:")
            with open('facefusion_stderr.log', 'r') as f:
                print(f.read())
        else:
            print(f"\nSuccessfully ran job {job_id}.")
            print("\nReading stdout log content:")
            with open('facefusion_stdout.log', 'r') as f:
                print(f.read())
          
    except Exception as e:
        # Handle all other unexpected errors
        print(f"\nAn unexpected error occurred while running job {job_id}: {e}")
    finally:
        # Always print the return code at the end
        if 'result_run' in locals():
            print(f"\nProcess finished with exit code: {result_run.returncode if result_run.returncode is not None else 'Unknown'}")


    # Clear the inference pool to release resources
    clear_inference_pool()

    # Clean up temporary resized images
    if resize_dimensions:
        temp_paths = set(resized_image_paths.values())
        for temp_path in temp_paths:
            if os.path.exists(temp_path):
                os.remove(temp_path)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
