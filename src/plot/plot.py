import matplotlib.pyplot as plt 
import torch

def show_plot(datas, targets):
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.tight_layout()
        plt.imshow(datas[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(targets[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()
    
    
def show_evaluate_plot(model, test_loader):
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predictions = torch.max(outputs, dim=1)
        
        plt.figure()
        for i in range(6):
            plt.subplot(2, 3, i+1)
            plt.tight_layout()
            plt.imshow(inputs[i][0], cmap="gray", interpolation='none')
            plt.title(f"Prediction {predictions[i]}, True {labels[i]}")
            plt.xticks([])
            plt.yticks([])
            
        plt.show()