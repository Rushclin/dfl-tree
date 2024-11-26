import random
import numpy as np

class Node:
    """Class representing a node in the binary tree."""
    def __init__(self, node_id, data_size, model=None):
        self.node_id = node_id  # Unique ID for the node
        self.data_size = data_size  # Size of local data
        self.model = model  # Model associated with this node
        self.left = None  # Left child
        self.right = None  # Right child
        self.parent = None  # Parent node

    def local_training(self, epochs, batch_size, lr):
        """
        Simulate local training.
        For simplicity, this just updates the model as a dummy process.
        """
        print(f"Node {self.node_id} is training locally for {epochs} epochs.")
        self.model = np.random.rand(10)  # Dummy model update (random weights)
    
    def aggregate_models(self):
        """
        Aggregate models from the children and combine with the local model.
        Weighted aggregation based on data size.
        """
        if self.left is None and self.right is None:
            return self.model  # Leaf node only returns its model
        
        total_data_size = self.data_size
        aggregated_model = self.model * self.data_size
        
        if self.left:
            left_model = self.left.aggregate_models()
            total_data_size += self.left.data_size
            aggregated_model += left_model * self.left.data_size
        
        if self.right:
            right_model = self.right.aggregate_models()
            total_data_size += self.right.data_size
            aggregated_model += right_model * self.right.data_size
        
        self.model = aggregated_model / total_data_size
        return self.model


class BinaryTree:
    """Class representing the binary tree structure."""
    def __init__(self, nodes):
        self.nodes = nodes  # List of all nodes
        # self.root = None  # Root node (server node)
        self.root = self.select_server()  # Root node (server node)

    def build_tree(self):
        """Organize nodes into a binary tree."""
        if not self.nodes:
            return None
        # self.root = self.nodes[0]  # Set the first node as the initial root
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

    def select_server(self):
        """
        Randomly select a server node from the tree.
        """
        return random.choice(self.nodes)

    def reorganize_tree(self, new_root):
        """
        Reorganize the tree to make the selected server the root.
        """
        print(f"Reorganizing the tree with Node {new_root.node_id} as the new root.")
        self.root = new_root  # For simplicity, we directly set the new root

    def broadcast_model(self):
        """
        Broadcast the root's model to all nodes.
        """
        print(f"Broadcasting model from Node {self.root.node_id} to all nodes.")
        for node in self.nodes:
            node.model = self.root.model  # Simulating broadcast by copying the root's model

    def print_tree(self, node=None, level=0):
        """Helper function to display the tree structure."""
        if node is None:
            node = self.root
        if node.right:
            self.print_tree(node.right, level + 1)
        print(" " * 4 * level + f"-> Node {node.node_id} (Data size: {node.data_size})")
        if node.left:
            self.print_tree(node.left, level + 1)


class VTreeLearning:
    """Orchestrates the V-Tree Learning process."""
    def __init__(self, tree, global_model):
        self.tree = tree
        self.global_model = global_model

    def train(self, rounds, epochs, batch_size, lr):
        """
        Main training loop for V-Tree Learning.
        """
        for t in range(rounds):
            print(f"\nRound {t + 1}/{rounds}...")
            
            # Step 1: Select a server node
            server = self.tree.select_server()
            print(f"Node {server.node_id} elected as server.")
            
            # Step 2: Reorganize tree with server as root
            self.tree.reorganize_tree(server)
            
            # Step 3: Broadcast global model
            self.tree.broadcast_model()
            
            # Step 4: Local training
            for node in self.tree.nodes:
                node.local_training(epochs=epochs, batch_size=batch_size, lr=lr)
            
            # Step 5: Aggregation
            print("Performing aggregation...")
            self.global_model = self.tree.root.aggregate_models()
            print(f"Global model updated: {self.global_model}")


# Main Execution
if __name__ == "__main__":
    # Create dummy nodes with random data sizes
    nodes = [Node(node_id=i, data_size=random.randint(10, 100)) for i in range(7)]
    
    # Initialize and build the binary tree
    tree = BinaryTree(nodes)
    tree.build_tree()
    print("Initial Tree Structure:")
    tree.print_tree()
    
    # Initialize V-Tree Learning
    initial_model = np.zeros(10)  # Dummy initial global model
    vtree = VTreeLearning(tree, initial_model)
    
    # Start training
    vtree.train(rounds=5, epochs=2, batch_size=32, lr=0.01)
    print("\nFinal Tree Structure:")
    tree.print_tree()
