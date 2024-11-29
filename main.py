import os
import hydra
import time
import logging 
import concurrent.futures
from omegaconf import DictConfig
from src import set_seed, check_args, load_model, load_dataset, create_nodes, BinaryTree, TqdmToLogger, set_logger

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(args : DictConfig) -> None:
    
    curr_time = time.strftime("%y%m%d_%H%M%S", time.localtime())
    args.result_path = os.path.join(
        args.result_path, f'{args.exp_name}_{curr_time}')
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)

    set_logger(f'{args.log_path}/{args.exp_name}_{curr_time}.log', args)
    
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
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(args.K, os.cpu_count() -1 )) as workhorse:
            for node in TqdmToLogger(
                tree.nodes,
                logger=logger,
                desc=f'[{args.dataset.upper()}]  ...Train node ... ',
                total=args.K
            ):
                workhorse.submit(node.update).result()
        
        logger.info("Performing aggregation...")
        tree.root.aggregate_models()
        
        if (curr_round % args.eval_every == 0) or (curr_round == args.R):
            tree.evaluate(curr_round=curr_round)
        
    tree.finalize()

if __name__ == "__main__":
    main()
