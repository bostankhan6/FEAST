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


# generate the non-IID distribution for all methods
def read_data_distribution(filename='./data_preprocessing/non-iid-distribution/CIFAR10/distribution.txt'):
    distribution = {}
    with open(filename, 'r') as data:
        for x in data.readlines():
            if '{' != x[0] and '}' != x[0]:
                tmp = x.split(':')
                if '{' == tmp[1].strip():
                    first_level_key = int(tmp[0])
                    distribution[first_level_key] = {}
                else:
                    second_level_key = int(tmp[0])
                    distribution[first_level_key][second_level_key] = int(tmp[1].strip().replace(',', ''))
    return distribution


def read_net_dataidx_map(filename='./data_preprocessing/non-iid-distribution/CIFAR10/net_dataidx_map.txt'):
    net_dataidx_map = {}
    with open(filename, 'r') as data:
        for x in data.readlines():
            if '{' != x[0] and '}' != x[0] and ']' != x[0]:
                tmp = x.split(':')
                if '[' == tmp[-1].strip():
                    key = int(tmp[0])
                    net_dataidx_map[key] = []
                else:
                    tmp_array = x.split(',')
                    net_dataidx_map[key] = [int(i.strip()) for i in tmp_array]
    return net_dataidx_map


def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    logging.debug('Data statistics: %s' % str(net_cls_counts))
    return net_cls_counts


class Cutout(object):
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

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cinic10():
    cinic_mean = [0.47889522, 0.47227842, 0.43047404]
    cinic_std = [0.24205776, 0.23828046, 0.25874835]
    # Transformer for train set: random crops and horizontal flip
    train_transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Lambda(
                                              lambda x: F.pad(x.unsqueeze(0),
                                                              (4, 4, 4, 4),
                                                              mode='reflect').data.squeeze()),
                                          transforms.ToPILImage(),
                                          transforms.RandomCrop(32),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=cinic_mean,
                                                               std=cinic_std),
                                          ])

    # Transformer for test/validation set (no augmentation)
    valid_transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize(mean=cinic_mean,
                                                               std=cinic_std),
                                          ])
    return train_transform, valid_transform


def _data_transforms_cinic10_mixaug():
    """Mix-aug: RandAugment(2,6) only. No Cutout; CutMix/Mixup applied at batch level in trainer."""
    cinic_mean = [0.47889522, 0.47227842, 0.43047404]
    cinic_std = [0.24205776, 0.23828046, 0.25874835]
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: F.pad(x.unsqueeze(0), (4, 4, 4, 4), mode='reflect').data.squeeze()),
        transforms.ToPILImage(),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=6),
        transforms.ToTensor(),
        transforms.Normalize(mean=cinic_mean, std=cinic_std),
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cinic_mean, std=cinic_std),
    ])
    return train_transform, eval_transform


def load_cinic10_data(datadir):
    """Load CINIC-10 data from train, valid, and test folders."""
    _train_dir = datadir + str('/train')
    _valid_dir = datadir + str('/valid')
    _test_dir = datadir + str('/test')
    
    logging.info("_train_dir = " + str(_train_dir))
    logging.info("_valid_dir = " + str(_valid_dir))
    logging.info("_test_dir = " + str(_test_dir))
    
    cinic_mean = [0.47889522, 0.47227842, 0.43047404]
    cinic_std = [0.24205776, 0.23828046, 0.25874835]
    
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: F.pad(x.unsqueeze(0), (4, 4, 4, 4), mode='reflect').data.squeeze()),
        transforms.ToPILImage(),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=cinic_mean, std=cinic_std),
    ])
    
    # No augmentation for val/test
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cinic_mean, std=cinic_std),
    ])
    
    trainset = ImageFolderTruncated(_train_dir, transform=train_transform)
    validset = ImageFolderTruncated(_valid_dir, transform=eval_transform)
    testset = ImageFolderTruncated(_test_dir, transform=eval_transform)
    
    X_train, y_train = trainset.imgs, trainset.targets
    X_val, y_val = validset.imgs, validset.targets
    X_test, y_test = testset.imgs, testset.targets
    
    return (X_train, y_train, X_val, y_val, X_test, y_test)


def partition_data(dataset, datadir, partition, n_nets, alpha,
                   client_budgets=None, corr_gamma=1.0, max_training_mac=600_000_000):
    """Partition CINIC-10 train data. Validation/test data passed through unchanged.

    Args:
        client_budgets: dict {client_id: mac_budget} for γ-correlated allocation.
                        If None, uses standard equal-share Dirichlet partitioning.
        corr_gamma: exponent for budget-proportional data allocation (default 1.0).
        max_training_mac: cap for budget before applying gamma (default 600M).
    """
    logging.info("*********partition data***************")
    pil_logger = logging.getLogger('PIL')
    pil_logger.setLevel(logging.INFO)

    X_train, y_train, X_val, y_val, X_test, y_test = load_cinic10_data(datadir)
    X_train = np.array(X_train)
    X_val = np.array(X_val)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_val = np.array(y_val)
    y_test = np.array(y_test)
    n_train = len(X_train)

    if partition == "homo":
        total_num = n_train
        idxs = np.random.permutation(total_num)
        batch_idxs = np.array_split(idxs, n_nets)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}

    elif partition == "hetero":
        min_size = 0
        K = 10
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

    elif partition == "hetero-fix":
        dataidx_map_file_path = './data_preprocessing/non-iid-distribution/CINIC10/net_dataidx_map.txt'
        net_dataidx_map = read_net_dataidx_map(dataidx_map_file_path)

    if partition == "hetero-fix":
        distribution_file_path = './data_preprocessing/non-iid-distribution/CINIC10/distribution.txt'
        traindata_cls_counts = read_data_distribution(distribution_file_path)
    else:
        traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)

    return X_train, y_train, X_val, y_val, X_test, y_test, net_dataidx_map, traindata_cls_counts


# for centralized training
def get_dataloader(dataset, datadir, train_bs, test_bs=256, dataidxs=None, augmentation="basic"):
    return get_dataloader_cinic10(datadir, train_bs, test_bs, dataidxs, augmentation=augmentation)


# for local devices
def get_dataloader_test(dataset, datadir, train_bs, test_bs, dataidxs_train, dataidxs_test):
    return get_dataloader_test_cinic10(datadir, train_bs, test_bs, dataidxs_train, dataidxs_test)


def get_dataloader_cinic10(datadir, train_bs, test_bs, dataidxs=None, augmentation="basic"):
    dl_obj = ImageFolderTruncated

    if augmentation == "mixaug":
        transform_train, transform_test = _data_transforms_cinic10_mixaug()
    else:
        transform_train, transform_test = _data_transforms_cinic10()

    traindir = os.path.join(datadir, 'train')
    valdir = os.path.join(datadir, 'test')

    train_ds = dl_obj(traindir, dataidxs=dataidxs, transform=transform_train)
    test_ds = dl_obj(valdir, transform=transform_test)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=False, num_workers=4)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=False, num_workers=4)

    return train_dl, test_dl

def get_dataloader_test_cinic10(datadir, train_bs, test_bs, dataidxs_train=None, dataidxs_test=None):
    dl_obj = ImageFolderTruncated

    transform_train, transform_test = _data_transforms_cinic10()

    traindir = os.path.join(datadir, 'train')
    valdir = os.path.join(datadir, 'test')

    train_ds = dl_obj(traindir, dataidxs=dataidxs_train, transform=transform_train)
    test_ds = dl_obj(valdir, dataidxs=dataidxs_test, transform=transform_test)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=False)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=False)

    return train_dl, test_dl


def load_partition_data_distributed_cinic10(process_id, dataset, data_dir, partition_method, partition_alpha,
                                            client_number, batch_size):
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(dataset,
                                                                                             data_dir,
                                                                                             partition_method,
                                                                                             client_number,
                                                                                             partition_alpha)
    class_num = len(np.unique(y_train))
    logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
    train_data_num = sum([len(net_dataidx_map[r]) for r in range(client_number)])

    # get global test data
    if process_id == 0:
        train_data_global, test_data_global = get_dataloader(dataset, data_dir, batch_size, batch_size)
        logging.info("train_dl_global number = " + str(len(train_data_global)))
        logging.info("test_dl_global number = " + str(len(train_data_global)))
        test_data_num = len(test_data_global)
        train_data_local = None
        test_data_local = None
        local_data_num = 0
    else:
        # get local dataset
        dataidxs = net_dataidx_map[process_id - 1]
        local_data_num = len(dataidxs)
        logging.info("rank = %d, local_sample_number = %d" % (process_id, local_data_num))
        # training batch size = 64; algorithms batch size = 32
        train_data_local, test_data_local = get_dataloader(dataset, data_dir, batch_size, batch_size,
                                                           dataidxs)
        logging.info("process_id = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
            process_id, len(train_data_local), len(test_data_local)))
        test_data_num = 0
        train_data_global = None
        test_data_global = None

    return train_data_num, test_data_num, train_data_global, test_data_global, local_data_num, train_data_local, test_data_local, class_num


def load_partition_data_cinic10(dataset, data_dir, partition_method, partition_alpha, client_number, batch_size, val_batch_size=256, validation_split=None, bn_calibration_split=0.0,
                                client_budgets=None, corr_gamma=1.0, max_training_mac=600_000_000, augmentation="basic"):
    """
    Loads and partitions CINIC-10 data for federated learning.
    
    CINIC-10 already comes with pre-split train/valid/test folders.
    The validation_split parameter is ignored (kept for API compatibility).
    
    Args:
        bn_calibration_split: Fraction of validation set for BN calibration (default 0.0 = no split).
    
    Returns:
        train_data_num: Total training samples across all clients
        val_data_num: Number of validation samples (from existing valid folder)
        test_data_num: Number of test samples
        train_data_global: DataLoader for all training data (for BN reset, etc.)
        val_data_global: DataLoader for validation set (for periodic eval during training)
        test_data_global: DataLoader for test set (use ONLY for final evaluation!)
        data_local_num_dict: {client_id: local_sample_count}
        train_data_local_dict: {client_id: local_train_loader}
        test_data_local_dict: {client_id: local_test_loader}
        class_num: Number of classes (10 for CINIC-10)
        bn_calibration_global: DataLoader for BN calibration (if bn_calibration_split > 0)
        bn_cal_data_num: Number of BN calibration samples
    """
    X_train, y_train, X_val, y_val, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        dataset,
        data_dir,
        partition_method,
        client_number,
        partition_alpha,
        client_budgets=client_budgets,
        corr_gamma=corr_gamma,
        max_training_mac=max_training_mac,
    )
    class_num = len(np.unique(y_train))
    logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
    train_data_num = sum([len(net_dataidx_map[r]) for r in range(client_number)])

    # --- Global Loaders ---
    train_data_global, test_data_global = get_dataloader(dataset, data_dir, batch_size, val_batch_size,
                                                         augmentation=augmentation)
    logging.info("train_dl_global number = " + str(len(train_data_global)))
    test_data_num = len(test_data_global.dataset)
    logging.info(f"test_dl_global samples = {test_data_num}")
    
    # --- Validation Loader (from existing valid folder) ---
    # Split into: Calibration (for BN reset) and Evaluation (for accuracy)
    # BN calibration must use the SAME transform as local client training
    # to match the distribution the model saw during training.
    if augmentation == "mixaug":
        train_transform, eval_transform = _data_transforms_cinic10_mixaug()
    else:
        train_transform, eval_transform = _data_transforms_cinic10()

    import os
    valid_dir = os.path.join(data_dir, 'valid')
    
    # --- Split validation set using bn_calibration_split ---
    total_val_samples = len(ImageFolderTruncated(valid_dir, transform=None))
    bn_calibration_global = None
    bn_cal_data_num = 0
    
    if bn_calibration_split > 0:
        bn_cal_size = int(total_val_samples * bn_calibration_split)
        bn_cal_size = max(bn_cal_size, batch_size)
        
        # Use fixed seed for reproducibility
        g = torch.Generator().manual_seed(42)
        indices = torch.randperm(total_val_samples, generator=g).tolist()
        bn_cal_indices = indices[:bn_cal_size]
        eval_indices = indices[bn_cal_size:]
        
        # BN Calibration: WITH training augmentations
        bn_cal_full = ImageFolderTruncated(valid_dir, transform=train_transform)
        bn_calibration_dataset = data.Subset(bn_cal_full, bn_cal_indices)
        
        # Evaluation: no augmentation
        eval_full = ImageFolderTruncated(valid_dir, transform=eval_transform)
        val_eval_dataset = data.Subset(eval_full, eval_indices)
        
        bn_calibration_global = data.DataLoader(
            bn_calibration_dataset, 
            batch_size=batch_size,
            shuffle=True, 
            drop_last=False, 
            num_workers=4,
            pin_memory=True
        )
        val_data_global = data.DataLoader(
            val_eval_dataset, 
            batch_size=val_batch_size, 
            shuffle=False, 
            drop_last=False, 
            num_workers=4
        )
        
        val_data_num = len(val_eval_dataset)
        bn_cal_data_num = len(bn_calibration_dataset)
        logging.info(f"val_dl_global (for eval): {val_data_num} samples (no augmentation)")
        logging.info(f"bn_calibration_global (for BN reset): {bn_cal_data_num} samples (WITH training augmentations)")
    else:
        full_val_dataset = ImageFolderTruncated(valid_dir, transform=eval_transform)
        val_data_global = data.DataLoader(
            full_val_dataset, 
            batch_size=val_batch_size, 
            shuffle=False, 
            drop_last=False, 
            num_workers=4
        )
        val_data_num = len(full_val_dataset)
        logging.info(f"val_dl_global (no BN split): {val_data_num} samples")

    # --- Local Client Loaders ---
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()

    for client_idx in range(client_number):
        dataidxs = net_dataidx_map[client_idx]
        local_data_num = len(dataidxs)
        data_local_num_dict[client_idx] = local_data_num
        logging.info("client_idx = %d, local_sample_number = %d" % (client_idx, local_data_num))

        train_data_local, test_data_local = get_dataloader(dataset, data_dir, batch_size, val_batch_size,
                                                           dataidxs, augmentation=augmentation)
        logging.info("client_idx = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
            client_idx, len(train_data_local), len(test_data_local)))
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local
    
    return (train_data_num, val_data_num, test_data_num, 
            train_data_global, val_data_global, test_data_global, 
            data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num,
            bn_calibration_global, bn_cal_data_num)

