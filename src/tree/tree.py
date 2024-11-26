import logging
import random
from ..node import Node
from typing import List
import torch
from src import MetricManager
import os
import json

logger = logging.getLogger(__name__)

class BinaryTree:
    """
    Class to manage the binary tree structure in V-Tree Learning.
    """
    def __init__(self, args, nodes: List[Node], test_dataset):
        self.nodes = nodes
        self.root: Node = None
        self.test_dataset = test_dataset
        self.args = args

    def build_tree(self):
        """
        Build a binary tree structure from the given nodes.
        """
        logger.info("Building the binary tree...")
        if not self.nodes:
            return
        self.root = self.nodes[0]
        queue = [self.root]
        idx = 1
        while queue and idx < len(self.nodes):
            current = queue.pop(0)
            if idx < len(self.nodes):
                current.left = self.nodes[idx]
                current.left.parent = current
                queue.append(current.left)
                idx += 1
            if idx < len(self.nodes):
                current.right = self.nodes[idx]
                current.right.parent = current
                queue.append(current.right)
                idx += 1

    def reorganize_tree(self, new_root):
        """
        Reorganize the tree to make the new root the server for this round.
        """
        logger.info(f"Reorganizing tree to make Node {new_root.node_id} the new root.")
        self.root = new_root

    def print_tree(self, node=None, level=0):
        """
        Helper function to visualize the tree structure.
        """
        if node is None:
            node = self.root
        if node.right:
            self.print_tree(node.right, level + 1)
        print(" " * 4 * level + f"-> Node {node.node_id} (Data size: {node.data_size})")
        if node.left:
            self.print_tree(node.left, level + 1)
            
    def select_server(self):
        """
        Randomly select a server node from the tree.
        """
        return random.choice(self.nodes)

    def broadcast_model(self):
        """
        Broadcast the root's model to all nodes.
        """
        print(f"Broadcasting model from Node {self.root.node_id} to all nodes.")
        for node in self.nodes:
            node.model = self.root.model  # Simulating broadcast by copying the root's model

    @torch.no_grad()
    def evaluate(self):
        mm = MetricManager(self.args.eval_metrics)
        self.root.global_model.to(self.args.device)
        self.root.global_model.eval()

        for inputs, targets in torch.utils.data.DataLoader(dataset=self.test_dataset, batch_size=self.args.B, shuffle=False):
            inputs, targets = inputs.to(
                self.args.device), targets.to(self.args.device)

            outputs = self.root.global_model(inputs)
            loss = torch.nn.__dict__[self.args.criterion]()(outputs, targets)

            mm.track(loss.item(), outputs, targets)
        else:
            self.root.global_model.to('cpu')
            mm.aggregate(len(self.test_dataset))

        result = mm.results
        server_log_string = f'[{self.args.dataset.upper()}] [EVALUATE] [SERVER] '

        loss = result['loss']
        server_log_string += f'| loss: {loss:.4f} '

        for metric, value in result['metrics'].items():
            server_log_string += f'| {metric}: {value:.4f} '
        logger.info(server_log_string)

        # self.writer.add_scalar('Server Loss', loss, self.round)
        # for name, value in result['metrics'].items():
        #     self.writer.add_scalar(f'Server {name.title()}', value, self.round)
        # else:
        #     self.writer.flush()
        # self.results[self.round]['server_evaluated'] = result
           
    def finalize(self):
       """
       Finalize the training process by saving results and model weights.
       Creates a JSON file to store training results and saves the model weights in PyTorch format.
       """
       # Log the saving process
       logger.info(f"[{self.args.dataset.upper()}] Saving the model and results...")

       # Ensure the result path exists
    #    os.makedirs(self.args.result_path, exist_ok=True)

       # Save results to a JSON file
    #    json_path = os.path.join(self.args.result_path, f'{self.args.exp_name}.json')
    #    with open(json_path, 'w', encoding='utf8') as result_file:
    #        results = {key: value for key, value in self.results.items()}
    #        json.dump(results, result_file, indent=4)
    #    logger.info(f"Results saved to {json_path}")

       # Save model weights
       model_path = os.path.join(self.args.result_path, f'{self.args.exp_name}.pt')
       torch.save(self.root.global_model.state_dict(), model_path)
       logger.info(f"Model weights saved to {model_path}")

       # Optional: Close the writer if using TensorBoard or similar
       # self.writer.close()

       logger.info(f"[{self.args.dataset.upper()}] Finalization complete!")
