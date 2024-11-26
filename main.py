import hydra
import logging 
from omegaconf import DictConfig
from src import set_seed, check_args, load_model, load_dataset, create_nodes, BinaryTree

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(args : DictConfig) -> None:
    
    set_seed(args.seed)
    
    check_args(args)
    
    model = load_model(args)
        
    test_dataset, node_dataset = load_dataset(args)
    
    nodes = create_nodes(args, node_dataset, model)
    
    tree = BinaryTree(args, nodes, test_dataset)
    tree.build_tree()
    tree.print_tree()
    
    for curr_round in range(1, args.R + 1):
        logger.info(f"Round {curr_round}")
        server = tree.select_server()
        logger.info(f"Node {server.node_id} are selected as serveur for round {curr_round}")
        tree.reorganize_tree(server)
        tree.broadcast_model()
        
        for node in tree.nodes:
            node.update()
        
        logger.info("Performing aggregation...")
        tree.root.aggregate_models()
        
        if (curr_round % args.eval_every == 0) or (curr_round == args.R):
            tree.evaluate()
        
    tree.finalize()

if __name__ == "__main__":
    main()
