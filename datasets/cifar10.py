from torchvision import datasets, transforms
import torch
import numpy as np

def get_data_scaler(config):
  """Data normalizer. Assume data are always in [0, 1]."""
  if config.data.centered:
    # Rescale to [-1, 1]
    return lambda x: x * 2. - 1.
  else:
    return lambda x: x


def getLoader(name, batch, test_batch, augment=False, hasGPU=False, conditional=-1):
    if name == 'cifar10':
        # 1) train/validation split fraction
        val_frac = 1.0 / 6.0
        random_seed = 0
        
        # 2) CIFAR-10 normalization constants (RGB)
        normalize = transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616)
        )

        # 3) Compose transforms
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
        if augment:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
        else:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                normalize
            ])

        # 4) DataLoader kwargs
        kwargs = {'num_workers': 4, 'pin_memory': True} if hasGPU else {}

        # 5) Download/train dataset
        full_train = datasets.CIFAR10(
            root='../data',
            train=True,
            download=True,
            transform=train_transform
        )
        test_data = datasets.CIFAR10(
            root='../data',
            train=False,
            download=True,
            transform=val_transform
        )

        # 6) (Optional) filter by class label
        if 0 <= conditional <= 9:
            # targets is a list for CIFAR10
            targets = np.array(full_train.targets)
            idx = targets == conditional
            # Subset the data and targets
            full_train.data = full_train.data[idx]
            full_train.targets = targets[idx].tolist()
            # Same for test set
            test_targets = np.array(test_data.targets)
            test_idx = test_targets == conditional
            test_data.data = test_data.data[test_idx]
            test_data.targets = test_targets[test_idx].tolist()

        # 7) Split into train / val
        n_total = len(full_train)
        n_val = int(val_frac * n_total)
        n_train = n_total - n_val
        generator = torch.Generator().manual_seed(random_seed)
        train_data, valid_data = torch.utils.data.random_split(
            full_train,
            [n_train, n_val],
            generator=generator
        )

        # 8) Build loaders
        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=batch,
            shuffle=True,
            **kwargs
        )
        val_loader = torch.utils.data.DataLoader(
            valid_data,
            batch_size=test_batch,
            shuffle=False,
            **kwargs
        )
        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=test_batch,
            shuffle=False,
            **kwargs
        )

        return train_loader, val_loader, test_loader

    else:
        raise ValueError(f"Unknown dataset: {name}")
