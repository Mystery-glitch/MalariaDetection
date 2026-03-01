import torch


def evaluate(model, loader):

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total