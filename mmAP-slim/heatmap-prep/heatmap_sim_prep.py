import os
import shutil
import random

def organize_files_by_action(base_dir):
    # Define the categories
    categories = ['angle', 'doppler', 'range']

    # Iterate over each category
    for category in categories:
        source_path = os.path.join(base_dir, category, 'all')
        files = [f for f in os.listdir(source_path) if f.endswith('.npy')]

        for file in files:
            # Extract the action identifier from the filename
            if '_ta' in file:
                action_id = file.split('_ta')[0]
            elif '_td' in file:
                action_id = file.split('_td')[0]
            elif '_tr' in file:
                action_id = file.split('_tr')[0]
            else:
                continue  # Skip files that don't match the expected pattern

            # Create a new directory for the action if it doesn't exist
            action_dir = os.path.join(base_dir, category, action_id)
            os.makedirs(action_dir, exist_ok=True)

            # Move the file to the action directory
            source_file_path = os.path.join(source_path, file)
            dest_file_path = os.path.join(action_dir, file)
            shutil.move(source_file_path, dest_file_path)
            print(f"Moved {file} to {action_dir}")
            
            
            
def move_files_for_evaluation(base_dir, eval_dir, split_ratio):
    # Ensure the evaluation directory exists
    os.makedirs(eval_dir, exist_ok=True)

    # Define the categories
    categories = ['angle', 'doppler', 'range']
    action_files = {}

    # Collect all files and group them by class directory and number
    for category in categories:
        category_path = os.path.join(base_dir, category)
        class_dirs = [d for d in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, d))]

        for class_dir in class_dirs:
            source_path = os.path.join(category_path, class_dir)
            files = [f for f in os.listdir(source_path) if f.endswith('.npy')]

            for file in files:
                # Extract the number from the filename
                number = file.split('_')[-1].split('.')[0][2:]  # Extract the number after _ta, _td, _tr

                if class_dir not in action_files:
                    action_files[class_dir] = {}
                if number not in action_files[class_dir]:
                    action_files[class_dir][number] = {}
                action_files[class_dir][number][category] = file

    # Move 20% of files with matching numbers from each class directory
    for class_dir, numbers_dict in action_files.items():
        num_files = len(numbers_dict)
        num_actions_to_move = int(num_files * split_ratio)

        # Randomly select numbers to move
        numbers_to_move = random.sample(list(numbers_dict.keys()), num_actions_to_move)

        for number in numbers_to_move:
            if all(category in numbers_dict[number] for category in categories):
                for category in categories:
                    file_name = numbers_dict[number][category]
                    source_path = os.path.join(base_dir, category, class_dir, file_name)
                    dest_path = os.path.join(eval_dir, category, class_dir)
                    os.makedirs(dest_path, exist_ok=True)
                    shutil.move(source_path, os.path.join(dest_path, file_name))
                    print(f"Moved {file_name} to {dest_path}")


# Example usage
base_directory = '/home/sxu7/data/heatmaps/simulation/finetune-cls/train/'  # Your current base directory
evaluation_directory = '/home/sxu7/data/heatmaps/simulation/finetune-cls/eval/'  # Your target evaluation directory
organize_files_by_action(base_directory)
move_files_for_evaluation(base_directory, evaluation_directory, split_ratio=0.2)
