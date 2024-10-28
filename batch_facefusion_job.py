import os
import glob
import subprocess
import argparse
import uuid  # For UUID-based job_id generation

# Parameters for models and execution
facefusion_script = 'facefusion.py'  # Path to the facefusion script
face_swapper_model = 'inswapper_128_fp16'
face_enhancer_model = 'gfpgan_1.4'
execution_provider = 'cuda'    # Use CUDA for execution

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Batch process face swapping with facefusion.')
    parser.add_argument('--input_directory', required=True, help='Directory containing images to process.')
    parser.add_argument('--output_directory', required=True, help='Directory to save output images.')
    parser.add_argument('--num_images', type=int, default=5, help='Number of images to process (default: 5).')

    args = parser.parse_args()

    input_directory = args.input_directory
    output_directory = args.output_directory
    num_images = args.num_images

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

    # Step 2: Add steps to the drafted job for each image pair
    for source_image in image_paths:
        source_basename = os.path.basename(source_image)
        for target_image in image_paths:
            target_basename = os.path.basename(target_image)
            # Exclude self-pairs
            if os.path.abspath(source_image) == os.path.abspath(target_image):
                continue  # Skip self-pairs

            # Get the target image's extension
            target_extension = os.path.splitext(target_basename)[1]

            # Generate the output filename with the same extension as the target image
            output_filename = f'output_{os.path.splitext(source_basename)[0]}_to_{os.path.splitext(target_basename)[0]}{target_extension}'
            output_path = os.path.join(output_directory, output_filename)

            # Add a step to the job
            cmd_add_step = [
                'python', facefusion_script, 'job-add-step',
                job_id,  # Provide the job_id as a positional argument
                '--source-paths', source_image,
                '--target-path', target_image,
                '--output-path', output_path,
                '--processors', 'face_swapper', 'face_enhancer',
                '--face-swapper-model', face_swapper_model,
                '--face-enhancer-model', face_enhancer_model,
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
        job_id  # Provide the job_id as a positional argument
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
        '--execution-providers', execution_provider
    ]
    result_run = subprocess.run(cmd_run, capture_output=True, text=True)
    if result_run.returncode != 0:
        print(f"Error running job {job_id}:")
        print(result_run.stderr)
    else:
        print(f"Successfully ran job {job_id}")
        print(result_run.stdout)

if __name__ == '__main__':
    main()

