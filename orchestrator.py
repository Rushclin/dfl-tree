# import requests
# import time

# NODES = ["http://node1:5000", "http://node2:5000"]

# def get_node_status():
#     statuses = []
#     for node in NODES:
#         try:
#             response = requests.get(f"{node}/status").json()
#             statuses.append(response)
#         except Exception as ex:
#             print(f"Could not reach {ex}")
#     return statuses

# def elect_server(nodes):
#     # Cost function example: prioritize lower latency and higher CPU
#     sorted_nodes = sorted(nodes, key=lambda x: x["resources"]["latency"] / x["resources"]["CPU"])
#     return sorted_nodes[0]["node_id"]

# def main():
#     for round_num in range(1, 10000000):  # Simulate 3 rounds
#         print(f"Starting round {round_num}")
#         nodes = get_node_status()
#         server_id = elect_server(nodes)
#         print(f"Node {server_id} elected as server for round {round_num}")
#         time.sleep(20)

# if __name__ == "__main__":
#     main()


import os 
import hydra
import time
import logging
from omegaconf import DictConfig
from src import set_seed, check_args, load_model, load_dataset, create_nodes, BinaryTree, TqdmToLogger, set_logger

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(args: DictConfig) -> None:
    
    curr_time = time.strftime("%y%m%d_%H%M%S", time.localtime())
    args.result_path = os.path.join(
        args.result_path, f'{args.exp_name}_{curr_time}')
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
        
        
        
    test_dataset, node_dataset = load_dataset(args)
    



if __name__ == "__main__":
    main()
