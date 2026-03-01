import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from src.data_loader import get_dataloaders
from src.model_builder import get_model
from src.train import train_model
from src.evaluate import evaluate
from src.metrics import get_metrics
from src.utils import save_model
from src.config import MODELS, LEARNING_RATES, OPTIMIZERS, EPOCHS


def main():

    train_loader, val_loader, test_loader = get_dataloaders()

    results = []
    best_global_acc = 0
    best_global_model = None

    for model_name in MODELS:
        for lr in LEARNING_RATES:
            for opt_name in OPTIMIZERS:

                print(f"\nTraining {model_name} | LR={lr} | OPT={opt_name}")

                model = get_model(model_name)
                criterion = nn.CrossEntropyLoss()

                if opt_name == "adam":
                    optimizer = optim.Adam(model.parameters(), lr=lr)
                else:
                    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

                trained_model, val_acc, train_losses, val_accuracies = train_model(
                    model, train_loader, val_loader,
                    optimizer, criterion, EPOCHS
                )

                test_acc = evaluate(trained_model, test_loader)

                results.append([model_name, lr, opt_name, test_acc])

                if test_acc > best_global_acc:
                    best_global_acc = test_acc
                    best_global_model = trained_model

    df = pd.DataFrame(results,
                      columns=["Model", "LR", "Optimizer", "Test Accuracy"])

    print("\nFINAL RESULTS:\n")
    print(df.sort_values("Test Accuracy", ascending=False))

    print("\nBest Test Accuracy:", best_global_acc)

    get_metrics(best_global_model, test_loader)

    save_model(best_global_model, "best_model.pth")


if __name__ == "__main__":
    main()