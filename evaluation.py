import torch
import hydra
import logging 
from omegaconf import DictConfig
from torch.nn.functional import cross_entropy
from src import set_seed, check_args, load_dataset, create_dataloaders, show_plot, load_experiment_model, show_evaluate_plot

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(args : DictConfig) -> None:
    set_seed(args.seed)
    
    check_args(args)
    
    model = load_experiment_model(args)
        
    test_dataset, _ = load_dataset(args)
    test_loader = create_dataloaders(args, test_dataset)
    examples = enumerate(test_loader)
    _, (example_data, example_targets) = next(examples)
    
    show_plot(example_data, example_targets)
    evaluate_model(args, model, test_loader)
    show_evaluate_plot(model, test_loader)
        
@torch.no_grad() # Deactivate retropropagation
def evaluate_model(args, model, test_loader):
    total_correct = 0
    total_samples = 0
    total_loss = 0

    for inputs, labels in test_loader:
        inputs, labels = inputs.to(args.device), labels.to(args.device)

        outputs = model(inputs)
        loss = cross_entropy(outputs, labels)
        total_loss += loss.item()

        _, predictions = torch.max(outputs, dim=1)
        total_correct += (predictions == labels).sum().item()
        total_samples += labels.size(0)

    accuracy = total_correct / total_samples * 100
    avg_loss = total_loss / len(test_loader)

    logger.info(f"Accuracy: {accuracy:.2f}%")
    logger.info(f"Average Loss: {avg_loss:.4f}")
    
    
    
@hydra.main(version_base=None, config_path="conf", config_name="config")
def test(args : DictConfig) -> None:
    model = load_experiment_model(args)
    
    validation_img_paths = [
        "./test/1/img_1755.jpg",
    ]
     
     

if __name__ == "__main__":
    test()
   
    
    
    
    
