import os
import shutil
from datetime import datetime
import subprocess
from pathlib import Path

def continue_training(existing_model_path, new_images_dir, new_ground_truth_dir, checkpoint_path=None):
    
    # Create timestamped training directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    training_dir = f'training_continuation_{timestamp}'
    os.makedirs(training_dir, exist_ok=True)
    
    # Backup existing model
    backup_dir = os.path.join(training_dir, 'backup')
    os.makedirs(backup_dir)
    shutil.copy2(existing_model_path, backup_dir)
    
    print("1. Creating training data...")
    # Create training list file
    training_list = []
    for img_file in os.listdir(new_images_dir):
        if img_file.endswith(('.png', '.jpg', '.jpeg')):
            base_name = os.path.splitext(img_file)[0]
            img_path = os.path.join(new_images_dir, img_file)
            gt_path = os.path.join(new_ground_truth_dir, f"{base_name}.txt")
            
            if os.path.exists(gt_path):
                # Convert image to TIFF
                tiff_path = os.path.join(training_dir, f"{base_name}.tiff")
                cmd = f'convert "{img_path}" "{tiff_path}"'
                subprocess.run(cmd, shell=True, check=True)
                
                # Create box file
                box_path = os.path.join(training_dir, f"{base_name}.box")
                cmd = f'tesseract "{tiff_path}" "{os.path.join(training_dir, base_name)}" batch.nochop makebox'
                subprocess.run(cmd, shell=True, check=True)
                
                training_list.append(f"{os.path.join(training_dir, base_name)}")
    
    # Write training list file
    list_file = os.path.join(training_dir, 'training_files.txt')
    with open(list_file, 'w') as f:
        f.write('\n'.join(training_list))
    
    print("2. Generating training command...")
    # Generate training command
    start_model = checkpoint_path if checkpoint_path else existing_model_path
    
    cmd = f"""lstmtraining \
        --continue_from "{start_model}" \
        --model_output "{os.path.join(training_dir, 'output')}" \
        --traineddata "{existing_model_path}" \
        --train_listfile "{list_file}" \
        --max_iterations 400 \
        --target_error_rate 0.01 \
        --debug_interval 100"""
    
    print("\nReady to continue training!")
    print("\nTraining command:")
    print(cmd)
    
    print("\n3. Starting training...")
    try:
        subprocess.run(cmd, shell=True, check=True)
        print("\nTraining completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\n{e}")
        print("You can continue from the last checkpoint later.")
    
    return training_dir

if __name__ == "__main__":
    # Example usage
    training_dir = continue_training(
        existing_model_path="/Users/rishikeshanand/Documents/Projects/fine-tuning-ocr/eng.traineddata",
        new_images_dir="/Users/rishikeshanand/Documents/Projects/fine-tuning-ocr/images",
        new_ground_truth_dir="/Users/rishikeshanand/Documents/Projects/fine-tuning-ocr/ground_truth",
        checkpoint_path=None  # Optional: path to checkpoint if resuming interrupted training
    )
    
    print(f"\nTraining files and output saved in: {training_dir}")