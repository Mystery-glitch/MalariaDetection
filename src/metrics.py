import torch
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


def get_metrics(model, test_loader):

    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:

            outputs = model(images)

            probs = torch.softmax(outputs, dim=1)[:, 1]
            _, preds = torch.max(outputs, 1)

            all_probs.extend(probs.numpy())
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    print("\nClassification Report:\n")
    print(classification_report(all_labels, all_preds))

    print("\nConfusion Matrix:\n")
    print(confusion_matrix(all_labels, all_preds))

    print("\nROC-AUC Score:\n")
    print(roc_auc_score(all_labels, all_probs))