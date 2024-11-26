import logging
import numpy as np
import torch
from .data import SubsetWrapper

logger = logging.getLogger(__name__)

# def split(args, dataset):
#     """
#     Split the dataset based on the specified split strategy (iid or non-iid).

#     Args:
#         args: The arguments containing split parameters like 'split_type' and the number of clients 'K'.
#         dataset: The full dataset to be split.

#     Returns:
#         split_map: A dictionary mapping each client index (0 to K-1) to their respective dataset indices.
#     """

#     # IID (Independent and Identically Distributed) split
#     if args.split_type == "iid":
#         # Shuffle the dataset indices randomly
#         shuffled_indices = np.random.permutation(len(dataset))

#         # Split the shuffled indices into K equal parts (one for each client)
#         split_indices = np.array_split(shuffled_indices, args.K)

#         # Construct a hash map (split_map) where each client has a set of dataset indices
#         split_map = {k: split_indices[k] for k in range(args.K)}
        
#         logger.info(f'[SPLIT] Performed IID split for {args.K} nodes.')
        
#         return split_map

#     # Non-IID (Non-Independent and Identically Distributed) split
#     if args.split_type == 'non-iid':
#         # Shuffle the dataset indices randomly
#         shuffled_indices = np.random.permutation(len(dataset))

#         # Split the shuffled indices into K parts (one for each client)
#         split_indices = np.array_split(shuffled_indices, args.K)

#         # Apply a random keep ratio between 95% and 99% for each client
#         keep_ratio = np.random.uniform(low=0.95, high=0.99, size=len(split_indices))

#         # Adjust each client's data by keeping only a fraction of their dataset based on the keep_ratio
#         split_indices = [indices[:int(len(indices) * ratio)] 
#                          for indices, ratio in zip(split_indices, keep_ratio)]

#         # Construct a hash map (split_map) with the adjusted non-IID split
#         split_map = {k: split_indices[k] for k in range(args.K)}
        
#         logger.info(f'[SPLIT] Performed Non-IID split for {args.K} nodes.')
        
#         return split_map


