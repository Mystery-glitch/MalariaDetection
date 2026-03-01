import torch.nn as nn
import torch.optim as optim
from .train import train_model
from .evaluate import evaluate
from .config import LEARNING_RATES, OPTIMIZERS


def tune_model(model, train_loader, val_loader, test_loader, epochs):

    results = []
    best_acc = 0
    best_model = None

    for lr in LEARNING_RATES:
        for opt_name in OPTIMIZERS:

            print(f"\nLR={lr} | OPT={opt_name}")

            if opt_name == "adam":
                optimizer = optim.Adam(model.parameters(), lr=lr)
            else:
                optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

            criterion = nn.CrossEntropyLoss()

            trained_model, val_acc, _, _ = train_model(
                model, train_loader, val_loader,
                optimizer, criterion, epochs
            )

            test_acc = evaluate(trained_model, test_loader)

            results.append([lr, opt_name, test_acc])

            if test_acc > best_acc:
                best_acc = test_acc
                best_model = trained_model

    return best_model, best_acc, results