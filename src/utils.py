import torch
import matplotlib.pyplot as plt


def save_model(model, path="best_model.pth"):
    torch.save(model.state_dict(), path)
    print("Model saved successfully!")


def plot_graph(train_losses, val_accuracies):

    epochs = range(1, len(train_losses)+1)

    plt.figure()
    plt.plot(epochs, train_losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

    plt.figure()
    plt.plot(epochs, val_accuracies)
    plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.show()