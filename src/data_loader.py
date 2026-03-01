from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from .config import DATA_DIR, BATCH_SIZE


def get_transforms():

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ])

    return train_transform, eval_transform


def split_dataset(dataset_path):

    base_dataset = datasets.ImageFolder(dataset_path)

    indices = list(range(len(base_dataset)))
    targets = base_dataset.targets

    train_idx, temp_idx = train_test_split(
        indices, test_size=0.3,
        stratify=targets, random_state=42)

    temp_targets = [targets[i] for i in temp_idx]

    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5,
        stratify=temp_targets, random_state=42)

    return train_idx, val_idx, test_idx


def get_dataloaders():

    train_transform, eval_transform = get_transforms()
    train_idx, val_idx, test_idx = split_dataset(DATA_DIR)

    train_full = datasets.ImageFolder(DATA_DIR, transform=train_transform)
    eval_full = datasets.ImageFolder(DATA_DIR, transform=eval_transform)

    train_ds = Subset(train_full, train_idx)
    val_ds = Subset(eval_full, val_idx)
    test_ds = Subset(eval_full, test_idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader