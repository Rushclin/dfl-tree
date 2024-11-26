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
