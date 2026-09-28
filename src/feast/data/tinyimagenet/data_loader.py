"""TinyImageNet-200 data loader for federated training.

Layout expected (after running scripts/data_setup/prepare_tinyimagenet.py):
    data/tinyimagenet/
        train/<wnid>/*.JPEG    # 100k images, 200 classes, 500/class
        val/<wnid>/*.JPEG      # 10k images, used as the test set

TinyImageNet has no separate validation folder (test/ has no labels). We follow
the CIFAR-100 protocol: hold out a fraction of train/ for server validation +
BN calibration, and use val/ as the held-out test set.

Image size: 64x64. Use stem_stride=2 in the supernet so feature maps after the
stem match CIFAR (32x32) and body MACs are comparable across datasets.
"""

import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms

from .datasets import ImageFolderTruncated

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


# Standard TinyImageNet statistics (commonly cited).
TINYIMAGENET_MEAN = [0.4802, 0.4481, 0.3975]
TINYIMAGENET_STD = [0.2770, 0.2691, 0.2821]


class Cutout:
    """Randomly mask out a square patch from the image (after ToTensor + Normalize).

    Identical to the implementation in cifar100/data_loader.py. Cutout(32) on 64×64
    images gives the same 25% coverage ratio as Cutout(16) on 32×32 CIFAR images.
    """
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)
        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)
        mask[y1:y2, x1:x2] = 0.0
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}
    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        net_cls_counts[net_i] = {unq[i]: unq_cnt[i] for i in range(len(unq))}
    logging.debug("Data statistics: %s" % str(net_cls_counts))
    return net_cls_counts


def _data_transforms_tinyimagenet():
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(
            lambda x: F.pad(x.unsqueeze(0), (8, 8, 8, 8), mode='reflect').data.squeeze()
        ),
        transforms.ToPILImage(),
        transforms.RandomCrop(64),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=TINYIMAGENET_MEAN, std=TINYIMAGENET_STD),
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=TINYIMAGENET_MEAN, std=TINYIMAGENET_STD),
    ])
    return train_transform, eval_transform


def _data_transforms_tinyimagenet_mixaug():
    """Mix-aug: RandAugment(2,6) only. No Cutout; CutMix/Mixup is applied in trainer."""
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(
            lambda x: F.pad(x.unsqueeze(0), (8, 8, 8, 8), mode='reflect').data.squeeze()
        ),
        transforms.ToPILImage(),
        transforms.RandomCrop(64),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=6),
        transforms.ToTensor(),
        transforms.Normalize(mean=TINYIMAGENET_MEAN, std=TINYIMAGENET_STD),
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=TINYIMAGENET_MEAN, std=TINYIMAGENET_STD),
    ])
    return train_transform, eval_transform


def _data_transforms_tinyimagenet_strong():
    """Strong augmentation: Cutout(32) + RandAugment(2,9) + ColorJitter."""
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(
            lambda x: F.pad(x.unsqueeze(0), (8, 8, 8, 8), mode='reflect').data.squeeze()
        ),
        transforms.ToPILImage(),
        transforms.RandomCrop(64),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.ToTensor(),
        transforms.Normalize(mean=TINYIMAGENET_MEAN, std=TINYIMAGENET_STD),
        Cutout(32),
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=TINYIMAGENET_MEAN, std=TINYIMAGENET_STD),
    ])
    return train_transform, eval_transform


def load_tinyimagenet_data(datadir):
    """Load TinyImageNet (path, label) tuples from the on-disk ImageFolder layout."""
    train_dir = os.path.join(datadir, "train")
    test_dir = os.path.join(datadir, "val")  # val/ is our held-out test set

    logging.info("train_dir = %s", train_dir)
    logging.info("test_dir  = %s (used as test set)", test_dir)

    train_transform, eval_transform = _data_transforms_tinyimagenet()
    trainset = ImageFolderTruncated(train_dir, transform=train_transform)
    testset = ImageFolderTruncated(test_dir, transform=eval_transform)

    X_train, y_train = trainset.imgs, trainset.targets
    X_test, y_test = testset.imgs, testset.targets
    return X_train, y_train, X_test, y_test


def partition_data(dataset, datadir, partition, n_nets, alpha, validation_split=0.0,
                   client_budgets=None, corr_gamma=1.0, max_training_mac=600_000_000):
    """Partition TinyImageNet train data among clients with optional Dirichlet heterogeneity.

    Optionally holds out validation_split fraction of train for server-side
    validation and BN calibration (val/ is reserved as the held-out test set).

    Args:
        client_budgets: dict {client_id: mac_budget} for γ-correlated allocation.
        corr_gamma: exponent for budget-proportional data allocation.
        max_training_mac: cap for budget before applying gamma.
        validation_split: fraction of train to hold out for server val (default 0.0).
    """
    logging.info("********* TinyImageNet partition data ***************")
    pil_logger = logging.getLogger("PIL")
    pil_logger.setLevel(logging.INFO)

    X_train_full, y_train_full, X_test, y_test = load_tinyimagenet_data(datadir)
    X_train_full = np.array(X_train_full, dtype=object)
    X_test = np.array(X_test, dtype=object)
    y_train_full = np.array(y_train_full)
    y_test = np.array(y_test)
    n_train_full = len(X_train_full)

    # --- Server validation split (before client partitioning) ---
    X_val, y_val = None, None
    if validation_split > 0:
        n_val = int(n_train_full * validation_split)
        np.random.seed(42)
        all_indices = np.random.permutation(n_train_full)
        val_indices = all_indices[:n_val]
        train_indices = all_indices[n_val:]
        X_val = X_train_full[val_indices]
        y_val = y_train_full[val_indices]
        X_train = X_train_full[train_indices]
        y_train = y_train_full[train_indices]
        logging.info(f"Validation split: {n_val} samples held out for server validation.")
        logging.info(f"Remaining client training samples: {len(train_indices)}")
    else:
        X_train = X_train_full
        y_train = y_train_full

    n_train = len(X_train)

    if partition == "homo":
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, n_nets)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}

    elif partition == "hetero":
        min_size = 0
        K = 200
        N = y_train.shape[0]
        logging.info("N = " + str(N))
        net_dataidx_map = {}

        # Per-client balance cap and Dirichlet concentration base, per Supplementary
        # Eq. gamma_alloc: q_i(gamma) = b_i_alloc^gamma / sum_j b_j_alloc^gamma, with
        # b_i_alloc = min(b_i, max_training_mac).
        if client_budgets is not None:
            budgets_arr = np.array([client_budgets[i] for i in range(n_nets)], dtype=float)
            capped_budgets = np.minimum(budgets_arr, max_training_mac)
            q = np.power(capped_budgets, corr_gamma)
            q = q / q.sum()
            balance_thresholds = q * N
            concentration_base = alpha * q
        else:
            balance_thresholds = np.full(n_nets, N / n_nets)
            concentration_base = None

        # Budget-aware Dirichlet partition (Supplementary Sec. A): draw p_k ~
        # Dir(c_k) per class (Eq. asym_dirichlet), mask clients past their balance
        # cap and renormalize, then form integer split points (Eq. integer_split_points).
        while min_size < 10:
            idx_batch = [[] for _ in range(n_nets)]
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                if client_budgets is not None:
                    concentration = np.maximum(concentration_base * len(idx_k), 1e-6)
                else:
                    concentration = np.repeat(alpha, n_nets)
                proportions = np.random.dirichlet(concentration)
                proportions = np.array([
                    p * (len(idx_j) < thresh)
                    for p, idx_j, thresh in zip(proportions, idx_batch, balance_thresholds)
                ])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

        if client_budgets is not None:
            sizes = [len(idx_batch[j]) for j in range(n_nets)]
            logging.info(
                f"[Correlated Partition] gamma={corr_gamma:.2f} | "
                f"client sample range: [{min(sizes)}, {max(sizes)}] | "
                f"total: {sum(sizes)}"
            )
    else:
        raise ValueError(f"Unsupported partition method for TinyImageNet: {partition}")

    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)
    return X_train, y_train, X_val, y_val, X_test, y_test, net_dataidx_map, traindata_cls_counts


def get_dataloader(dataset, datadir, train_bs, test_bs=256, dataidxs=None, augmentation="basic"):
    return get_dataloader_tinyimagenet(datadir, train_bs, test_bs, dataidxs, augmentation=augmentation)


def get_dataloader_tinyimagenet(datadir, train_bs, test_bs, dataidxs=None, augmentation="basic"):
    dl_obj = ImageFolderTruncated
    if augmentation == "strong":
        transform_train, transform_test = _data_transforms_tinyimagenet_strong()
    elif augmentation == "mixaug":
        transform_train, transform_test = _data_transforms_tinyimagenet_mixaug()
    else:
        transform_train, transform_test = _data_transforms_tinyimagenet()

    traindir = os.path.join(datadir, "train")
    testdir = os.path.join(datadir, "val")

    train_ds = dl_obj(traindir, dataidxs=dataidxs, transform=transform_train)
    test_ds = dl_obj(testdir, transform=transform_test)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True,
                               drop_last=False, num_workers=4, pin_memory=True)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False,
                              drop_last=False, num_workers=4, pin_memory=True)
    return train_dl, test_dl


def _build_subset_loader_from_train(datadir, indices, transform, batch_size, shuffle):
    """Build a DataLoader over a fixed subset of train/ with the given transform.

    Used to construct val_data_global and bn_calibration_global from the
    held-out validation indices (since TinyImageNet has no separate val folder).
    """
    traindir = os.path.join(datadir, "train")
    base_ds = ImageFolderTruncated(traindir, transform=transform)
    subset = data.Subset(base_ds, indices.tolist() if hasattr(indices, "tolist") else list(indices))
    return data.DataLoader(subset, batch_size=batch_size, shuffle=shuffle,
                           drop_last=False, num_workers=4, pin_memory=True)


def load_partition_data_tinyimagenet(dataset, data_dir, partition_method, partition_alpha,
                                     client_number, batch_size, val_batch_size=256,
                                     validation_split=0.1, bn_calibration_split=0.0,
                                     client_budgets=None, corr_gamma=1.0,
                                     max_training_mac=600_000_000, augmentation="basic"):
    """Top-level loader for federated TinyImageNet training.

    Args:
        validation_split: fraction of train held out for server val (default 0.1).
        bn_calibration_split: within the held-out val, fraction reserved for BN
            calibration with training augmentations (default 0.0 = no BN cal loader).
        augmentation: 'basic' (pad+crop+flip), 'strong' (+ Cutout + RandAugment(2,9) + ColorJitter),
            or 'mixaug' (+ RandAugment(2,6), no Cutout; CutMix/Mixup applied at batch level).

    Returns the same 12-tuple as load_partition_data_cinic10 for drop-in compatibility.
    """
    X_train, y_train, X_val, y_val, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        dataset,
        data_dir,
        partition_method,
        client_number,
        partition_alpha,
        validation_split=validation_split,
        client_budgets=client_budgets,
        corr_gamma=corr_gamma,
        max_training_mac=max_training_mac,
    )
    class_num = len(np.unique(y_train))
    logging.info(f"class_num = {class_num} (expected 200 for TinyImageNet)")
    train_data_num = sum([len(net_dataidx_map[r]) for r in range(client_number)])

    # --- Global loaders (full train + test) ---
    train_data_global, test_data_global = get_dataloader(dataset, data_dir, batch_size, val_batch_size,
                                                         augmentation=augmentation)
    test_data_num = len(test_data_global.dataset)
    logging.info(f"test_dl_global samples = {test_data_num}")

    # --- Validation + BN calibration loaders (from the held-out val subset of train) ---
    val_data_global = None
    val_data_num = 0
    bn_calibration_global = None
    bn_cal_data_num = 0

    if X_val is not None and len(X_val) > 0:
        if augmentation == "strong":
            train_transform, eval_transform = _data_transforms_tinyimagenet_strong()
        elif augmentation == "mixaug":
            train_transform, eval_transform = _data_transforms_tinyimagenet_mixaug()
        else:
            train_transform, eval_transform = _data_transforms_tinyimagenet()
        # Recover the original train-set indices held out as val. partition_data shuffled
        # with seed 42 and took the first n_val; reproduce that index set here.
        n_train_full = len(X_train) + len(X_val)
        n_val = len(X_val)
        np.random.seed(42)
        all_indices = np.random.permutation(n_train_full)
        val_indices = all_indices[:n_val]

        if bn_calibration_split > 0:
            bn_cal_size = int(n_val * bn_calibration_split)
            bn_cal_size = max(bn_cal_size, batch_size)
            g = torch.Generator().manual_seed(42)
            shuffled = torch.randperm(n_val, generator=g).tolist()
            bn_cal_local = shuffled[:bn_cal_size]
            eval_local = shuffled[bn_cal_size:]
            bn_cal_global_indices = val_indices[bn_cal_local]
            eval_global_indices = val_indices[eval_local]

            bn_calibration_global = _build_subset_loader_from_train(
                data_dir, bn_cal_global_indices, train_transform, batch_size, shuffle=True
            )
            val_data_global = _build_subset_loader_from_train(
                data_dir, eval_global_indices, eval_transform, val_batch_size, shuffle=False
            )
            val_data_num = len(val_data_global.dataset)
            bn_cal_data_num = len(bn_calibration_global.dataset)
            logging.info(f"val_dl_global (eval): {val_data_num} samples (no augmentation)")
            logging.info(f"bn_calibration_global: {bn_cal_data_num} samples (train augmentations)")
        else:
            val_data_global = _build_subset_loader_from_train(
                data_dir, val_indices, eval_transform, val_batch_size, shuffle=False
            )
            val_data_num = len(val_data_global.dataset)
            logging.info(f"val_dl_global (no BN split): {val_data_num} samples")

    # --- Per-client loaders ---
    data_local_num_dict = {}
    train_data_local_dict = {}
    test_data_local_dict = {}

    for client_idx in range(client_number):
        dataidxs = net_dataidx_map[client_idx]
        local_data_num = len(dataidxs)
        data_local_num_dict[client_idx] = local_data_num
        logging.info("client_idx = %d, local_sample_number = %d" % (client_idx, local_data_num))
        train_data_local, test_data_local = get_dataloader(
            dataset, data_dir, batch_size, val_batch_size, dataidxs, augmentation=augmentation
        )
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local

    return (train_data_num, val_data_num, test_data_num,
            train_data_global, val_data_global, test_data_global,
            data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num,
            bn_calibration_global, bn_cal_data_num)
