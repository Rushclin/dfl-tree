import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, Grayscale
import logging
import numpy as np
import random
from collections import defaultdict



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

    
    

# def split(args, dataset):
#     """
#     Split the dataset into subsets for nodes, supporting both IID and Non-IID distributions.
    
#     Args:
#         args: Object containing split parameters (args.split_type).
#         dataset: PyTorch dataset to split (e.g., MNIST).
#         num_classes_per_node: Number of classes per node (only for Non-IID split).
    
#     Returns:
#         node_datasets: A list of SubsetWrapper objects, one for each node.
#     """
#     if args.split_type == 'non-iid':
#         if args.num_classes_per_node is None:
#             raise ValueError("For 'non-iid' split, 'num_classes_per_node' must be specified.")

#         # Group indices by class
#         classes = dataset.targets.unique()  # Unique classes
#         class_indices = {cls.item(): (dataset.targets == cls).nonzero(as_tuple=True)[0] for cls in classes}

#         node_datasets = []

#         # Distribute classes to nodes
#         for i in range(args.K):
#             # Select classes for the current node
#             start_idx = (i * args.num_classes_per_node) % len(classes)
#             end_idx = start_idx + args.num_classes_per_node
#             selected_classes = classes[start_idx:end_idx]

#             # Get indices for the selected classes
#             selected_indices = torch.cat([class_indices[cls.item()] for cls in selected_classes])
#             data = dataset.data[selected_indices]
#             targets = dataset.targets[selected_indices]

#             # Create dataset for the node
#             node_datasets.append(SubsetWrapper(data, targets))

#         return node_datasets

#     elif args.split_type == 'iid':
#         # IID split: Distribute data equally across nodes
#         data_per_node = len(dataset) // args.K
#         indices = torch.randperm(len(dataset))  # Shuffle indices for randomness
#         node_datasets = []

#         for i in range(args.K):
#             subset_indices = indices[i * data_per_node:(i + 1) * data_per_node]
#             data = dataset.data[subset_indices]
#             targets = dataset.targets[subset_indices]

#             # Create dataset for the node
#             node_datasets.append(SubsetWrapper(data, targets))

#         return node_datasets

#     else:
#         raise ValueError("Invalid split type. Use 'iid' or 'non-iid'.")


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
        
        logger.info(f'[SPLIT] Performed IID split for {args.K} clients.')
        
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
        
        logger.info(f'[SPLIT] Performed non-IID split for {args.K} clients.')
        
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


def create_dataloaders(node_datasets, batch_size):
    
    # dataloaders = []
    # for node_dataset in node_datasets:
    return DataLoader(node_datasets, batch_size=batch_size, shuffle=True)
        # dataloaders.append(dataloader)
        
    # return dataloaders



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
    print(type(raw_dataset.dataset));
    
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



# def stratified_split(raw_dataset: Subset, test_size: float):
#     """
#     Split the dataset into training and testing sets in a stratified manner (i.e., balanced across classes).
    
#     Args:
#         raw_dataset: The dataset to split, wrapped in a PyTorch Subset.
#         test_size: The proportion of data to be used for testing (between 0 and 1).
        
#     Returns:
#         train_subset: The training subset.
#         test_subset: The testing subset.
#     """
#     from collections import defaultdict
#     import random
#     import torch

#     # Group indices by their labels
#     indices_per_label = defaultdict(list)
#     for subset_index in raw_dataset.indices:
#         # Get the label by directly querying the dataset
#         _, label = raw_dataset.dataset[subset_index]
#         indices_per_label[label].append(subset_index)

#     train_indices, test_indices = [], []
#     for label, indices in indices_per_label.items():
#         # Determine the number of samples for testing based on the test_size ratio
#         n_samples_for_label = round(len(indices) * test_size)
#         test_sample_indices = random.sample(indices, n_samples_for_label)
#         train_sample_indices = list(set(indices) - set(test_sample_indices))

#         test_indices.extend(test_sample_indices)
#         train_indices.extend(train_sample_indices)

#     return torch.utils.data.Subset(raw_dataset.dataset, train_indices), torch.utils.data.Subset(raw_dataset.dataset, test_indices)




def load_dataset(args):
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

    # Image transformation function to apply resizing, normalization, and grayscaling.
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
                Resize((args.resize, args.resize)),  # Resize the image
                ToTensor(),  # Convert the image to a PyTorch tensor
                # Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize the tensor
                Normalize((0.5), (0.5)),
                Grayscale(num_output_channels=1)  # Convert images to grayscale (for MNIST)
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
            subset, f'< {str(idx).zfill(8)} > (train)')
        if len(subset) * args.test_size > 0:
            test_set = SubsetWrapper(
                test_set, f'< {str(idx).zfill(8)} > (test)')
        else:
            test_set = None
        return (training_dataset, test_dataset)


    
    transforms = [_get_transform(args), _get_transform(args)]  # Get transformations for training and testing

    # Fetch the actual datasets with the transformations applied
    train_dataset, test_dataset = fetch_dataset(args=args, transforms=transforms)
    
    split_map = split(args, train_dataset)
    node_datasets = []
    for idx, sample_indices in split_map.items():
        node_datasets.append(_construct_dataset(train_dataset, idx, sample_indices))
        # print(f"Sample indice {sample_indices} IDx {idx}")
    # node_datasets = split(args, train_dataset)

    # Create the dataset for clients if it hasn't been created yet
    # if client_datasets is None:
    #     logger.info(f'[SIMULATION] Creating the dataset for clients!')

    #     client_datasets = []

    #     # Use a ThreadPoolExecutor to parallelize dataset creation for multiple clients
    #     with concurrent.futures.ThreadPoolExecutor(max_workers=min(args.K, os.cpu_count() - 1)) as workhorse:
    #         for idx, sample_indices in TqdmToLogger(
    #             enumerate(split_map.values()),
    #             logger=logger,
    #             desc=f'[SIMULATION] ...Creating client datasets... ',
    #             total=len(split_map)
    #         ):
    #             # Submit dataset construction tasks for each client in parallel
    #             client_datasets.append(workhorse.submit(
    #                 _construct_dataset, train_dataset, idx, sample_indices).result())
    #     logger.info(f'[SIMULATION] ...Client dataset creation completed!')

    # # Run the garbage collector to free up memory
    # gc.collect() 

    return test_dataset, node_datasets













# Example Usage
# if __name__ == "__main__":
#     # Load MNIST dataset
#     train_dataset, test_dataset = fetch_dataset()

#     # Split into IID subsets for 10 nodes
#     num_nodes = 10
#     node_datasets = split(train_dataset, num_nodes)

#     # Create DataLoaders for each node
#     batch_size = 32
#     dataloaders = create_dataloaders(node_datasets, batch_size)

#     # Verify the split
#     for i, dataloader in enumerate(dataloaders):
#         print(f"Node {i+1}: Number of samples = {len(dataloader.dataset)}")




# import os
# import gc
# import logging
# import concurrent.futures
# from torch.utils.data import Subset, Dataset
# from torchvision.transforms import Compose, Resize, ToTensor, Normalize, Grayscale

# from src import TqdmToLogger, stratified_split
# from .split import split
# from src.datasets import *

# logger = logging.getLogger(__name__)


# class SubsetWrapper(Dataset):
#     """
#     Wrapper class for PyTorch Subset that adds custom representation.
    
#     This class allows us to create subsets of the dataset and provides 
#     a custom suffix that can represent whether the dataset is for training or testing.
#     """

#     def __init__(self, subset, suffix):
#         """
#         Initialize the SubsetWrapper with a subset and a suffix.
        
#         Args:
#             subset (torch.utils.data.Subset): A subset of the original dataset.
#             suffix (str): A string suffix to describe the subset (e.g., 'train', 'test').
#         """
#         self.subset = subset
#         self.suffix = suffix

#     def __getitem__(self, index):
#         """
#         Retrieve an item from the subset by its index.
        
#         Args:
#             index (int): Index of the item to retrieve.
            
#         Returns:
#             tuple: Input (image or data point) and its corresponding target (label).
#         """
#         inputs, targets = self.subset[index]
#         return inputs, targets

#     def __len__(self):
#         """
#         Get the length of the subset (i.e., the number of items).
        
#         Returns:
#             int: The length of the subset.
#         """
#         return len(self.subset)

#     def __repr__(self):
#         """
#         Custom string representation for the SubsetWrapper.
        
#         Returns:
#             str: Representation of the dataset with the custom suffix.
#         """
#         return f'{repr(self.subset.dataset.dataset)} {self.suffix}'


# def load_dataset(args):
#     """
#     Load and split the dataset into training and testing sets.
    
#     This function applies image transformations, splits the dataset for multiple clients, and 
#     returns both the test dataset and the training dataset for clients.

#     Args:
#         args: The arguments containing settings for the dataset, such as resize dimensions, 
#               split type, and number of clients.

#     Returns:
#         test_dataset (torch.utils.data.Dataset): The test dataset.
#         client_datasets (list): A list containing subsets of the dataset for each client.
#     """

#     # Image transformation function to apply resizing, normalization, and grayscaling.
#     def _get_transform(args):
#         """
#         Define the transformations applied to the dataset.
        
#         Args:
#             args: Arguments specifying transformations, like resizing and normalization.
            
#         Returns:
#             transform: A composed list of transformations for resizing, normalizing, 
#                        and converting images to grayscale (specific for MNIST).
#         """
#         transform = Compose(
#             [
#                 Resize((args['resize'], args['resize'])),  # Resize the image
#                 ToTensor(),  # Convert the image to a PyTorch tensor
#                 Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize the tensor
#                 Grayscale(num_output_channels=1)  # Convert images to grayscale (for MNIST)
#             ]
#         )
#         return transform

#     # Function to construct dataset for each client, splitting it into training and testing sets.
#     def _construct_dataset(train_dataset, idx, sample_indices):
#         """
#         Create subsets for each client and split the subset into training and testing sets.
        
#         Args:
#             train_dataset: The full training dataset.
#             idx (int): Index representing the client.
#             sample_indices: The indices for this client's subset.
        
#         Returns:
#             tuple: A tuple containing the training set and the testing set for the client.
#         """
#         subset = Subset(train_dataset, sample_indices)  # Create a subset from the dataset

#         # Split the subset into training and testing sets using stratified sampling
#         training_set, test_set = stratified_split(subset, args.test_size)

#         # Wrap the training set and testing set with custom identifiers
#         training_set = SubsetWrapper(
#             training_set, f'< {str(idx).zfill(8)} > (train)')
#         if len(subset) * args.test_size > 0:
#             test_set = SubsetWrapper(
#                 test_set, f'< {str(idx).zfill(8)} > (test)')
#         else:
#             test_set = None
#         return (training_set, test_set)

#     # Initialize the training and test datasets
#     train_dataset, test_dataset = None, None

#     # Variables for split mapping and client datasets
#     split_map, client_datasets = None, None

#     # Prepare transformations for the datasets
#     transforms = [None, None]
#     transforms = [_get_transform(args), _get_transform(args)]  # Get transformations for training and testing

#     # Fetch the actual datasets with the transformations applied
#     train_dataset, test_dataset = fetch_dataset(args=args, transforms=transforms)

#     # If we are working with local evaluation, check if we need to remove the test dataset
#     if args.eval_type == 'local':
#         if args.test_size == -1:
#             assert test_dataset is not None  # Ensure test dataset exists
#         test_dataset = None  # Set test_dataset to None if not needed

#     # If no split map exists, split the dataset according to the specified strategy
#     if split_map is None:
#         logger.info(f'[SIMULATION] Distributing the dataset using the strategy: `{args.split_type.upper()}`!')
#         split_map = split(args, train_dataset)  # Perform the dataset splitting
#         logger.info(f'[SIMULATION] ...Finished distribution with the strategy: `{args.split_type.upper()}`!')

#     # Create the dataset for clients if it hasn't been created yet
#     if client_datasets is None:
#         logger.info(f'[SIMULATION] Creating the dataset for clients!')

#         client_datasets = []

#         # Use a ThreadPoolExecutor to parallelize dataset creation for multiple clients
#         with concurrent.futures.ThreadPoolExecutor(max_workers=min(args.K, os.cpu_count() - 1)) as workhorse:
#             for idx, sample_indices in TqdmToLogger(
#                 enumerate(split_map.values()),
#                 logger=logger,
#                 desc=f'[SIMULATION] ...Creating client datasets... ',
#                 total=len(split_map)
#             ):
#                 # Submit dataset construction tasks for each client in parallel
#                 client_datasets.append(workhorse.submit(
#                     _construct_dataset, train_dataset, idx, sample_indices).result())
#         logger.info(f'[SIMULATION] ...Client dataset creation completed!')

#     # Run the garbage collector to free up memory
#     gc.collect() 

#     return test_dataset, client_datasets
