import hydra
import logging 
from omegaconf import DictConfig
from src import set_seed, check_args, load_model, load_dataset, create_nodes, BinaryTree, create_dataloaders, show_plot, load_experiment_model, show_evaluate_plot
import torch
from torch.nn.functional import cross_entropy


logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(args : DictConfig) -> None:
    set_seed(args.seed)
    
    check_args(args)
    
    model = load_experiment_model(args)
    
    
    # Plot an images
    
    test_dataset, node_dataset = load_dataset(args)
    test_loader = create_dataloaders(test_dataset, 10)
    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    
    show_plot(example_data, example_targets)
    evaluate_model(model, test_loader)
    show_evaluate_plot(model, test_loader)
    
    # End plot an images
    

def evaluate_model(model, test_loader):
    total_correct = 0
    total_samples = 0
    total_loss = 0

    with torch.no_grad():  # Désactive la rétropropagation pour l'évaluation
        for inputs, labels in test_loader:
            inputs, labels = inputs.to("cpu"), labels.to("cpu")  # Remplacez "cpu" par "cuda" si applicable

            # Prédictions du modèle
            outputs = model(inputs)
            loss = cross_entropy(outputs, labels)
            total_loss += loss.item()

            # Calcul des prédictions correctes
            _, predictions = torch.max(outputs, dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = total_correct / total_samples * 100
    avg_loss = total_loss / len(test_loader)

    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Average Loss: {avg_loss:.4f}")


if __name__ == "__main__":
    my_app()
