import torch
import logging
import inspect
from collections import defaultdict
from src import MetricManager

logger = logging.getLogger(__name__)

class Node:
    """
    Class representing a node in the binary tree for V-Tree Learning.
    Each node can act as a client or server during a training round.
    """
    def __init__(self, node_id, data_size, args, node_dataset, model, test_dataset):
        self.node_id = node_id  # Unique ID for the node
        self.data_size = data_size  # Size of local data
        self.model = model  # Local model on the node
        self.left = None  # Left child
        self.right = None  # Right child
        self.parent = None  # Parent node

        # Additional attributes for training and logging
        self.args = args
        # self.writer = writer
        self.node_dataset = node_dataset
        self.test_dataset = test_dataset
        # self.opt_kwargs = dict(lr=self.args.lr, momentum=self.args.beta1)
        # self.curr_lr = self.args.lr
        self.results = defaultdict(dict)
        self.global_model = model
        
        self.train_loader = self._create_dataloader(self.node_dataset, shuffle=self.args.shuffle)
        self.test_loader = self._create_dataloader(self.test_dataset, shuffle=False)

        self.optim = torch.optim.__dict__[self.args.optimizer]
        self.criterion = torch.nn.__dict__[self.args.criterion]
        
    def update(self):
        update_sizes = self.train()
        _ = self.evaluate()

    def train(self):
        """
        Perform local training on the node using its dataset.
        Simulates model updates on local data.
        """
                
        mm = MetricManager(self.args.eval_metrics)  # Track metrics such as loss and accuracy
        self.model.train()
        self.model.to(self.args.device)

        # Initialize optimizer with refined arguments
        optimizer = self.optim(self.model.parameters(), **self._refine_optim_args(self.args))
        
        for e in range(self.args.E):  # Loop through epochs
            logger.info(f'[EPOCH] Round {e} / {self.args.E}')            
            
            for inputs, targets in self.train_loader:  # Iterate through training data
                inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)

                outputs = self.model(inputs)  # Forward pass
                loss = self.criterion()(outputs, targets)  # Compute loss

                for param in self.model.parameters():
                    param.grad = None  # Zero gradients manually for each parameter
                loss.backward()  # Backward pass
                if self.args.max_grad_norm > 0:
                    # Clip gradients to avoid exploding gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                optimizer.step()  # Update the model's parameters

                mm.track(loss.item(), outputs, targets)  # Track loss and output metrics

            mm.aggregate(len(self.node_dataset), e + 1)  # Aggregate metrics for the epoch

        self.model.to('cpu')  # Move model back to CPU after training
        return mm.results
        
    @torch.inference_mode()
    def evaluation(self):
        """
        Evaluates the client's local model on the test dataset.
        This method runs in inference mode (no gradient calculation).
        Returns:
            A dictionary containing the evaluation results (loss and other metrics).
        """
        
        mm = MetricManager(self.args.eval_metrics)
        self.model.eval()  # Set the model to evaluation mode
        self.model.to(self.args.device)

        # Evaluate the model on the test set
        for inputs, targets in self.test_loader:
            inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)

            outputs = self.model(inputs)
            loss = self.criterion()(outputs, targets)  # Compute loss

            mm.track(loss.item(), outputs, targets)  # Track evaluation metrics
        else:
            self.model.to('cpu')  # Move model back to CPU after evaluation
            mm.aggregate(len(self.test_set))  # Aggregate evaluation metrics
        result = mm.results
        
        node_log_string = f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Tour: {str(self.round).zfill(4)}] [EVALUATE] [NODE] '
        loss = result['loss']
        node_log_string += f'| loss: {loss:.4f} '
        for metric, value in result['metrics'].items():
            node_log_string += f'| {metric}: {value:.4f} '
        logger.info(node_log_string)


    def aggregate_models(self):
        """
        Aggregate models from child nodes and combine with the local model.
        Weighted aggregation is performed based on the data sizes.
        """
        logger.info(f"Aggregating models on Node {self.node_id}...")
        total_data_points = self.data_size
        aggregated_model = {key: torch.zeros_like(param) for key, param in self.global_model.state_dict().items()}

        # Aggregate contributions from child nodes
        for child in [self.left, self.right]:
            if child is not None: 
                child_model = child.aggregate_models()
                weight = child.data_size / (total_data_points + child.data_size)
                if child_model is not None:
                    for key, param in child_model.items():
                       aggregated_model[key] += weight * param
                    total_data_points += child.data_size

        # Combine with local model
        local_model = self.model.state_dict()
        weight = self.data_size / total_data_points
        for key, param in local_model.items():
            aggregated_model[key] += weight * param

        self.model.load_state_dict(aggregated_model)
        logger.info(f"Node {self.node_id} completed aggregation.")
        # return aggregated_model

    @torch.no_grad()
    def evaluate(self):
        """
        Centrally evaluate the global model on the server dataset.
        Tracks evaluation metrics such as loss and accuracy.
        """
        logger.info(f"Evaluating the model on Node {self.node_id}...")
        
        mm = MetricManager(self.args.eval_metrics)
        self.global_model.to(self.args.device)
        self.global_model.eval()
        
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)
                outputs = self.global_model(inputs)
                loss = torch.nn.__dict__[self.args.criterion]()(outputs, targets)
                # metrics['loss'] += loss.item()
                # metrics['accuracy'] += (outputs.argmax(dim=1) == targets).float().mean().item()
                mm.track(loss.item(), outputs, targets)
            else:
                self.global_model.to('cpu')
                mm.aggregate(len(self.test_dataset))
        
            result = mm.results
            server_log_string = f'[{self.args.dataset.upper()}] [EVALUATE] [NODE] '

            loss = result['loss']
            server_log_string += f'| loss: {loss:.4f} '

            for metric, value in result['metrics'].items():
                server_log_string += f'| {metric}: {value:.4f} '
            logger.info(server_log_string)
            # logger.info(f"Evaluation Results (Node {self.node_id}): Loss: {metrics['loss']:.4f}, Accuracy: {metrics['accuracy']:.4f}")
        return mm
    
    def _create_dataloader(self, dataset, shuffle):
        """
        Creates a PyTorch DataLoader for the client's dataset.
        Args:
            dataset: The dataset to create the DataLoader from (training or test set).
            shuffle: Whether to shuffle the dataset before creating the DataLoader.
        Returns:
            A PyTorch DataLoader object.
        """
        if self.args.B == 0:  # If batch size is 0, set it to the size of the dataset
            self.args.B = len(dataset)
        return torch.utils.data.DataLoader(dataset=dataset, batch_size=self.args.B, shuffle=shuffle)

    def _refine_optim_args(self, args):
        """
        Refines the optimizer arguments based on the available attributes in args.
        Only includes the arguments required by the optimizer.
        Args:
            args: Configuration and arguments passed to the optimizer.
        Returns:
            refined_args: A dictionary containing only the relevant arguments for the optimizer.
        """
        required_args = inspect.getfullargspec(self.optim)[0]  # Get required arguments for the optimizer

        refined_args = {}
        for argument in required_args:
            if hasattr(args, argument): 
                refined_args[argument] = getattr(args, argument)
        return refined_args
    
    
def create_nodes(args, node_dataset,  model): 
    
    train_node_dataset, test_node_dataset = node_dataset[1] 
    
    nodes = [Node(node_id=i, data_size=len(node_dataset), 
                  args=args, node_dataset=train_node_dataset, 
                  model=model, test_dataset=test_node_dataset) for i in range(args.K)]
    
    return nodes
   