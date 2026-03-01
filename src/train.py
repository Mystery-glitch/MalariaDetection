import copy
from .evaluate import evaluate


def train_model(model, train_loader, val_loader, optimizer, criterion, epochs):

    best_weights = copy.deepcopy(model.state_dict())
    best_acc = 0

    train_losses = []
    val_accuracies = []

    for epoch in range(epochs):

        model.train()
        running_loss = 0

        for images, labels in train_loader:

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        val_acc = evaluate(model, val_loader)

        train_losses.append(epoch_loss)
        val_accuracies.append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())

        print(f"Epoch {epoch+1} | Loss: {epoch_loss:.4f} | Val Acc: {val_acc:.4f}")

    model.load_state_dict(best_weights)

    return model, best_acc, train_losses, val_accuracies