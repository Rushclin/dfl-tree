import gc 
import os
import torch
import random
import logging
import numpy as np
from typing import Dict
import concurrent.futures
from src import TqdmToLogger
from collections import defaultdict
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, Grayscale

logger = logging.getLogger(__name__)

class SubsetWrapper(Dataset):
    """
    Custom Dataset class for individual nodes.
    Each node will receive a subset of the MNIST dataset.
    """
    
    def __init__(self, subset, suffix):
        self.subset = subset
        self.suffix = suffix

    def __getitem__(self, index):
        """
        Retrieve an item from the subset by its index.
        
        Args:
            index (int): Index of the item to retrieve.
            
        Returns:
            tuple: Input (image or data point) and its corresponding target (label).
        """

        inputs, targets = self.subset[index]
        return inputs, targets

    def __len__(self):
        """
        Get the length of the subset (i.e., the number of items).
        
        Returns:
            int: The length of the subset.
        """
        return len(self.subset)

    def __repr__(self):
        """
        Custom string representation for the SubsetWrapper.
        
        Returns:
            str: Representation of the dataset with the custom suffix.
        """
        return f'{repr(self.subset.dataset)} {self.suffix}'

    
    
def split(args, dataset):
    """
    Split the dataset based on the specified split strategy (iid or non-iid).

    Args:
        args: The arguments containing split parameters like 'split_type' and the number of clients 'K'.
        dataset: The full dataset to be split.

    Returns:
        split_map: A dictionary mapping each client index (0 to K-1) to their respective dataset indices.
    """

    # IID (Independent and Identically Distributed) split
    if args.split_type == "iid":
        # Shuffle the dataset indices randomly
        shuffled_indices = np.random.permutation(len(dataset))

        # Split the shuffled indices into K equal parts (one for each client)
        split_indices = np.array_split(shuffled_indices, args.K)

        # Construct a hash map (split_map) where each client has a set of dataset indices
        split_map = {k: split_indices[k] for k in range(args.K)}
        
        logger.info(f'[SPLIT] Performed IID split for {args.K} nodes.')
        
        return split_map

    # Non-IID (Non-Independent and Identically Distributed) split
    if args.split_type == 'non-iid':
        # Shuffle the dataset indices randomly
        shuffled_indices = np.random.permutation(len(dataset))

        # Split the shuffled indices into K parts (one for each client)
        split_indices = np.array_split(shuffled_indices, args.K)

        # Apply a random keep ratio between 95% and 99% for each client
        keep_ratio = np.random.uniform(low=0.95, high=0.99, size=len(split_indices))

        # Adjust each client's data by keeping only a fraction of their dataset based on the keep_ratio
        split_indices = [indices[:int(len(indices) * ratio)] 
                         for indices, ratio in zip(split_indices, keep_ratio)]

        # Construct a hash map (split_map) with the adjusted non-IID split
        split_map = {k: split_indices[k] for k in range(args.K)}
        
        logger.info(f'[SPLIT] Performed non-IID split for {args.K} nodes.')
        
        return split_map

    
    
def fetch_dataset(args, transforms):
    """
    Load a dataset dynamically based on its name (e.g., MNIST, CIFAR-10, EMNIST).
    Args:
        args: Object containing the dataset name as `args.dataset`.
        transforms: Tuple of (train_transform, test_transform).
    
    Returns:
        train_dataset: Training dataset.
        test_dataset: Testing dataset.
    Raises:
        ValueError: If the dataset name is not found in torchvision.datasets.
    """    
    
    if not hasattr(datasets, args.dataset):
        raise ValueError(f"Dataset '{args.dataset}' is not supported. Check the name and try again.")

    dataset_class = getattr(datasets, args.dataset)  # Get the dataset class
    train_dataset = dataset_class(root="data", train=True, transform=transforms[0], download=True)
    test_dataset = dataset_class(root="data", train=False, transform=transforms[1], download=True)

    return train_dataset, test_dataset


def create_dataloaders(args, node_datasets):
    
    return DataLoader(node_datasets, batch_size=args.B, shuffle=True)


def stratified_split(raw_dataset: Subset, test_size: int):
    """
    Split the dataset into training and testing sets in a stratified manner (i.e., balanced across classes).
    
    Args:
        raw_dataset: The dataset to split, wrapped in a PyTorch Subset.
        test_size: The proportion of data to be used for testing.
        
    Returns:
        train_subset: The training subset.
        test_subset: The testing subset.
    """
    indices_per_label = defaultdict(list)
    
    # Group indices by label
    for index, label in enumerate(np.array(raw_dataset.dataset.targets)[raw_dataset.indices]):
        indices_per_label[label.item()].append(index)
    
    train_indices, test_indices = [], []
    for label, indices in indices_per_label.items():
        # Determine the number of samples for testing based on the test_size ratio
        n_samples_for_label = round(len(indices) * test_size)
        random_indices_sample = random.sample(indices, n_samples_for_label)
        test_indices.extend(random_indices_sample)
        train_indices.extend(set(indices) - set(random_indices_sample))
    
    return torch.utils.data.Subset(raw_dataset, train_indices), torch.utils.data.Subset(raw_dataset, test_indices)



def load_dataset(args: Dict, node_id: int):
    """
    Load and split the dataset into training and testing sets.
    
    This function applies image transformations, splits the dataset for multiple clients, and 
    returns both the test dataset and the training dataset for clients.

    Args:
        args: The arguments containing settings for the dataset, such as resize dimensions, 
              split type, and number of clients.

    Returns:
        test_dataset (torch.utils.data.Dataset): The test dataset.
        client_datasets (list): A list containing subsets of the dataset for each client.
    """

    def _get_transform(args):
        """
        Define the transformations applied to the dataset.
        
        Args:
            args: Arguments specifying transformations, like resizing and normalization.
            
        Returns:
            transform: A composed list of transformations for resizing, normalizing, 
                       and converting images to grayscale (specific for MNIST).
        """
        transform = Compose(
            [
                Resize((args.resize, args.resize)),  
                ToTensor(),  
                Normalize((0.5), (0.5)),
                Grayscale(num_output_channels=1)  
            ]
        )
        return transform
    
    def _construct_dataset(train_dataset, idx, sample_indices):
        """
        Create subsets for each node and split the subset into training and testing sets.
        
        Args:
            train_dataset: The full training dataset.
            idx (int): Index representing the node.
            sample_indices: The indices for this node's subset.
        
        Returns:
            tuple: A tuple containing the training set and the testing set for the client.
        """
        subset = Subset(train_dataset, sample_indices)  # Create a subset from the dataset

        # Split the subset into training and testing sets using stratified sampling
        training_set, test_set = stratified_split(subset, args.test_size)

        # Wrap the training set and testing set with custom identifiers
        training_dataset = SubsetWrapper(
            training_set, f'< {str(idx).zfill(8)} > (train)')
        if len(subset) * args.test_size > 0:
            test_set = SubsetWrapper(
                test_set, f'< {str(idx).zfill(8)} > (test)')
        else:
            test_set = None
        return (training_dataset, test_dataset)


    transforms = [_get_transform(args), _get_transform(args)]  

    train_dataset, test_dataset = fetch_dataset(args=args, transforms=transforms)
    
    split_map = split(args, train_dataset)
    node_datasets = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(args.K, os.cpu_count() -1)) as workhorse:
        for ids, sample_indices in TqdmToLogger(
            enumerate(split_map.values()),
            logger=logger,
            desc=f'[SIMULATION] ...Create a node dataset... ',
            total=len(split_map)
        ):
            node_datasets.append(workhorse.submit(_construct_dataset, train_dataset, ids, sample_indices).result())
    logger.info(f"End create a nodes dataset")
      
    gc.collect() # memory liberation
    return node_datasets[node_id], test_dataset
