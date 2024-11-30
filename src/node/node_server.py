# from fastapi import FastAPI
# import os
# import random

# import os
# import uvicorn

# app = FastAPI()

# # Node-specific configurations
# node_id = os.getenv("NODE_ID", "unknown")
# resources = {
#     "CPU": random.randint(2, 8),  # Simulate CPU cores
#     "GPU": random.randint(1, 4),  # Simulate GPU availability
#     "latency": random.uniform(10, 100),  # Simulate network latency
# }

# @app.get("/status")
# def get_status():
#     return {"node_id": node_id, "resources": resources}

# @app.post("/train")
# def train_model(data: dict):
#     # Simulated local training process
#     epochs = data.get("epochs", 1)
#     return {"node_id": node_id, "epochs_completed": epochs, "status": "success"}


# config = uvicorn.Config(app, host='0.0.0.0', port=5000, log_level="debug")
# server = uvicorn.Server(config)
# server.run()


from fastapi import FastAPI
from node import NodeLogic
import torch
import json

app = FastAPI()

# Charger la configuration pour le nœud
with open("configs/args.json") as f:
    args = json.load(f)

# Créer un modèle PyTorch
from models.model import MyModel  # Exemple de modèle
model = MyModel()

# Exemple de datasets simulés
train_dataset = torch.utils.data.TensorDataset(torch.randn(100, 10), torch.randint(0, 2, (100,)))
test_dataset = torch.utils.data.TensorDataset(torch.randn(20, 10), torch.randint(0, 2, (20,)))

# Créer une instance de la logique de nœud
node = NodeLogic(node_id=args["node_id"], args=args, node_dataset=train_dataset, model=model, test_dataset=test_dataset)

@app.post("/train")
def train():
    """Endpoint pour entraîner le modèle localement."""
    results = node.train()
    return {"status": "success", "results": results}

@app.post("/evaluate")
def evaluate():
    """Endpoint pour évaluer le modèle local."""
    results = node.evaluate()
    return {"status": "success", "results": results}

@app.get("/status")
def status():
    """Endpoint pour vérifier l'état du nœud."""
    return {
        "node_id": node.node_id,
        "data_size": node.data_size,
        "left_child": node.left,
        "right_child": node.right,
        "parent": node.parent,
    }
