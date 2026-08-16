import os
import shutil
import random

def organize_files_by_action(base_dir):
    """
    Organize files into directories based on the base name of the filenames.
    """
    # Define the categories
    categories = ['angle', 'doppler', 'range']

    # Iterate over each category
    for category in categories:
        source_path = os.path.join(base_dir, category, 'all')
        
        # Check if source directory exists
        if not os.path.exists(source_path):
            print(f"Source directory {source_path} does not exist.")
            continue

        files = [f for f in os.listdir(source_path) if f.endswith('.npy')]

        for file in files:
            # Extract the base name (excluding number and extension)
            base_name = ''.join(filter(str.isalpha, file.split('.')[0]))

            # Create a new directory for the base name if it doesn't exist
            action_dir = os.path.join(base_dir, category, base_name)
            os.makedirs(action_dir, exist_ok=True)

            # Move the file to the action directory
            source_file_path = os.path.join(source_path, file)
            dest_file_path = os.path.join(action_dir, file)
            shutil.move(source_file_path, dest_file_path)
            print(f"Moved {file} to {action_dir}")


def move_files_for_evaluation(base_dir, eval_dir, split_ratio):
    os.makedirs(eval_dir, exist_ok=True)
    categories = ['angle', 'doppler', 'range']

    for action_dir in os.listdir(os.path.join(base_dir, categories[0])):
        action_files = {}
        
        # Collect files from all categories
        for category in categories:
            category_path = os.path.join(base_dir, category, action_dir)
            if not os.path.exists(category_path):
                print(f"Category directory does not exist: {category_path}")
                continue
            
            files = [f for f in os.listdir(category_path) if f.endswith('.npy')]
            for file in files:
                base_name = os.path.splitext(file)[0]
                if base_name not in action_files:
                    action_files[base_name] = []
                action_files[base_name].append((category, file))
        
        # Determine number of sets to move
        num_files = len(action_files)
        num_files_to_move = max(1, int(num_files * split_ratio))

        # Randomly select sets of matching files to move
        files_to_move = random.sample(list(action_files.keys()), num_files_to_move)

        for base_name in files_to_move:
            for category, file in action_files[base_name]:
                source_path = os.path.join(base_dir, category, action_dir, file)
                dest_dir = os.path.join(eval_dir, category, action_dir)
                os.makedirs(dest_dir, exist_ok=True)
                dest_path = os.path.join(dest_dir, file)
                shutil.move(source_path, dest_path)
                print(f"Moved {file} from {source_path} to {dest_path}")


# Example usage
base_directory = '/data/sxu7/heatmaps/recorded/finetune-cls/train'
evaluation_directory = '/data/sxu7/heatmaps/recorded/finetune-cls/eval/'

# Organize files by action
organize_files_by_action(base_directory)

# Uncomment to move files back to 'all'
move_files_for_evaluation(base_directory, evaluation_directory, 0.2)
