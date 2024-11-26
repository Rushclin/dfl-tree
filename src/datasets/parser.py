import os
import shutil
import random
import logging
from torchvision.datasets import ImageFolder

logger = logging.getLogger(__name__)

def split_dataset(input_folder: str, output_folder: str, train_ratio=0.8) -> None:
    """
    Splits the dataset into two groups: Training and Validation.
    
    Args:
        input_folder (str): Path to the folder containing the dataset (organized in subfolders for each class).
        output_folder (str): Path where the split dataset (train/validation) will be saved.
        train_ratio (float): Proportion of the data to be used for training. Default is 0.8 (80% for training).
    """
    
    logger.info("[LOAD] Organizing dataset folders...")

    # Check if output folder already exists
    if os.path.exists(output_folder):
        logger.info(
            "[LOAD] The output folder already exists. No operation is performed.")
        return

    os.makedirs(output_folder, exist_ok=True)

    # Iterate over each class folder in the input dataset
    for class_folder in os.listdir(input_folder):
        class_path = os.path.join(input_folder, class_folder)

        if not os.path.isdir(class_path):
            continue  # Skip if it's not a directory (e.g., a file)

        # List all files (images) in the current class folder
        all_files = os.listdir(class_path)

        # Split files into training and validation sets based on the train_ratio
        num_train = int(len(all_files) * train_ratio)
        train_files = random.sample(all_files, num_train)
        validation_files = [
            file for file in all_files if file not in train_files]

        # Create corresponding directories for train and validation sets
        train_path = os.path.join(output_folder, "train", class_folder)
        validation_path = os.path.join(
            output_folder, "validation", class_folder)

        os.makedirs(train_path, exist_ok=True)
        os.makedirs(validation_path, exist_ok=True)

        # Copy training files to the train directory
        for file in train_files:
            shutil.copy(os.path.join(class_path, file),
                        os.path.join(train_path, file))

        # Copy validation files to the validation directory
        for file in validation_files:
            shutil.copy(os.path.join(class_path, file),
                        os.path.join(validation_path, file))

    logger.info("[LOAD] Dataset folder organization complete.")


def fetch_dataset(args, transforms):
    """
    Fetches and loads the dataset by organizing it into training and validation sets, 
    then applies the given transformations.
    
    Args:
        args: Command-line arguments or configuration settings containing input_folder, output_folder, train_ratio.
        transforms (tuple): A tuple containing transformations for training and validation datasets.
        
    Returns:
        train_dataset: A PyTorch dataset object for the training set.
        validation_dataset: A PyTorch dataset object for the validation set.
    """
    
    logger.info(f'[LOAD] Loading the dataset...')

    # Organize dataset into train/validation folders
    split_dataset(input_folder=args.input_folder,
                  output_folder=args.output_folder, train_ratio=args.train_ratio)

    # Define data transformations for training and validation
    data_transform = {
        'train': transforms[0],
        'validation': transforms[1]
    }

    # Load datasets using ImageFolder
    image_datasets = {
        'train':
        ImageFolder(f'{args.output_folder}/train',
                    data_transform['train']),
        'validation':
        ImageFolder(f'{args.output_folder}/validation',
                    data_transform['validation'])
    }

    logger.info(f'[LOAD] Dataset loading complete!')

    # Return the training and validation datasets
    return image_datasets['train'], image_datasets['validation']
