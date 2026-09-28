import logging

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import pickle
from .datasets import CIFAR100_truncated

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


def _data_transforms_cifar100():
    CIFAR_MEAN = [0.5071, 0.4865, 0.4409]
    CIFAR_STD = [0.2673, 0.2564, 0.2762]

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    train_transform.transforms.append(Cutout(16))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    return train_transform, valid_transform


def _data_transforms_cifar100_mixaug():
    """Mix-aug: RandAugment(2,6) pixel-space. No Cutout; CutMix/Mixup applied at batch level in trainer."""
    CIFAR_MEAN = [0.5071, 0.4865, 0.4409]
    CIFAR_STD = [0.2673, 0.2564, 0.2762]

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=6),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    return train_transform, valid_transform

def load_cifar100_data(datadir):
    train_transform, test_transform = _data_transforms_cifar100()

    cifar10_train_ds = CIFAR100_truncated(datadir, train=True, download=True, transform=train_transform)
    cifar10_test_ds = CIFAR100_truncated(datadir, train=False, download=True, transform=test_transform)

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

    return (X_train, y_train, X_test, y_test)


def partition_data(dataset, datadir, partition, n_nets, alpha, load_from_pkl, validation_split=0.0,
                   client_budgets=None, corr_gamma=1.0, max_training_mac=600_000_000):
    """
    Partitions CIFAR-100 training data across clients.
    
    Args:
        validation_split: Fraction of training data to hold out for server validation (0.0-0.5).
                         This split happens BEFORE client partitioning.
    
    Returns:
        X_train, y_train: Client training data (after validation split)
        X_val, y_val: Server validation data (if validation_split > 0)
        X_test, y_test: Official test set (untouched)
        net_dataidx_map: Client partition indices
        traindata_cls_counts: Class distribution per client
    """
    logging.debug("*********partition data***************")
    X_train_full, y_train_full, X_test, y_test = load_cifar100_data(datadir)
    
    if load_from_pkl:
        with open(datadir+'/train.pkl', 'rb') as file:
            train_tuple = pickle.load(file)
            X_train_full, y_train_full = train_tuple[0], train_tuple[1]
    
    n_train_full = X_train_full.shape[0]
    
    # --- Server Validation Split (before client partitioning) ---
    X_val, y_val = None, None
    if validation_split > 0:
        n_val = int(n_train_full * validation_split)
        # Shuffle and split
        np.random.seed(42)  # Fixed seed for reproducibility
        all_indices = np.random.permutation(n_train_full)
        val_indices = all_indices[:n_val]
        train_indices = all_indices[n_val:]
        
        X_val = X_train_full[val_indices]
        y_val = y_train_full[val_indices]
        X_train = X_train_full[train_indices]
        y_train = y_train_full[train_indices]
        
        logging.info(f"Validation Split: {n_val} samples held out for server validation.")
        logging.info(f"Remaining Training Samples: {len(train_indices)}")
    else:
        X_train = X_train_full
        y_train = y_train_full
    
    n_train = X_train.shape[0]

    if partition == "homo":
        total_num = n_train
        idxs = np.random.permutation(total_num)
        batch_idxs = np.array_split(idxs, n_nets)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}

    elif partition == "hetero":
        min_size = 0
        K = 100
        N = y_train.shape[0]
        logging.debug("N = " + str(N))
        net_dataidx_map = {}

        # Pre-compute per-client balance thresholds.
        # Standard: every client is capped at N/n_nets (equal share).
        # Correlated: cap at q_i * N so high-budget clients can accumulate more,
        # per Supplementary Eq. gamma_alloc: q_i(gamma) = b_i_alloc^gamma / sum_j b_j_alloc^gamma.
        if client_budgets is not None:
            budgets_arr = np.array([client_budgets[i] for i in range(n_nets)], dtype=float)
            # b_i_alloc = min(b_i, B_cap) (Eq. gamma_alloc). Client budgets can exceed
            # max_training_mac (e.g., budget=1500M, max_training_mac=600M); without this
            # cap, high-budget clients would get disproportionately more data even though
            # they only train subnets up to max_training_mac. This ensures fairness.
            capped_budgets = np.minimum(budgets_arr, max_training_mac)
            q = np.power(capped_budgets, corr_gamma)
            q = q / q.sum()
            balance_thresholds = q * N          # shape (n_nets,), per-client cap
            concentration_base = alpha * q      # will be scaled per class below
        else:
            balance_thresholds = np.full(n_nets, N / n_nets)
            concentration_base = None

        # The following loop implements Supplementary Sec. A's budget-aware Dirichlet
        # partition: draw p_k ~ Dir(c_k) per class (Eq. asym_dirichlet, c_k,i =
        # alpha_data * N_k * q_i(gamma)), mask out clients that already reached their
        # per-class balance cap and renormalize, then form integer split points
        # (Eq. integer_split_points) via the cumulative-proportions cut below.
        while min_size < 10:
            idx_batch = [[] for _ in range(n_nets)]
            # for each class in the dataset
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                if client_budgets is not None:
                    concentration = np.maximum(concentration_base * len(idx_k), 1e-6)
                else:
                    concentration = np.repeat(alpha, n_nets)
                proportions = np.random.dirichlet(concentration)
                ## Balance: zero out clients that have already reached their target share
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
        dataidx_map_file_path = './data_preprocessing/non-iid-distribution/CIFAR100/net_dataidx_map.txt'
        net_dataidx_map = read_net_dataidx_map(dataidx_map_file_path)

    if partition == "hetero-fix":
        distribution_file_path = './data_preprocessing/non-iid-distribution/CIFAR100/distribution.txt'
        traindata_cls_counts = read_data_distribution(distribution_file_path)
    else:
        traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)

    return X_train, y_train, X_val, y_val, X_test, y_test, net_dataidx_map, traindata_cls_counts


# for centralized training
def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None, load_from_pkl=False, augmentation="basic"):
    return get_dataloader_CIFAR100(datadir, train_bs, test_bs, dataidxs, load_from_pkl, augmentation=augmentation)


# for local devices
def get_dataloader_test(dataset, datadir, train_bs, test_bs, dataidxs_train, dataidxs_test):
    return get_dataloader_test_CIFAR100(datadir, train_bs, test_bs, dataidxs_train, dataidxs_test)


def get_dataloader_CIFAR100(datadir, train_bs, test_bs, dataidxs=None, load_from_pkl=False, augmentation="basic"):
    dl_obj = CIFAR100_truncated

    if augmentation == "mixaug":
        transform_train, transform_test = _data_transforms_cifar100_mixaug()
    else:
        transform_train, transform_test = _data_transforms_cifar100()

    train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=True)
    test_ds = dl_obj(datadir, train=False, transform=transform_test, download=True)
    if load_from_pkl:
        with open(datadir+'/train.pkl', 'rb') as file:
            train_tuple = pickle.load(file)
            train_ds.data, train_ds.target = train_tuple[0], train_tuple[1]
    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=False)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=False)

    return train_dl, test_dl


def get_dataloader_test_CIFAR100(datadir, train_bs, test_bs, dataidxs_train=None, dataidxs_test=None):
    dl_obj = CIFAR100_truncated

    transform_train, transform_test = _data_transforms_cifar100()

    train_ds = dl_obj(datadir, dataidxs=dataidxs_train, train=True, transform=transform_train, download=True)
    test_ds = dl_obj(datadir, dataidxs=dataidxs_test, train=False, transform=transform_test, download=True)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=False)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=False)

    return train_dl, test_dl


def load_partition_data_distributed_cifar100(process_id, dataset, data_dir, partition_method, partition_alpha,
                                            client_number, batch_size):
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(dataset,
                                                                                             data_dir,
                                                                                             partition_method,
                                                                                             client_number,
                                                                                             partition_alpha)
    class_num = len(np.unique(y_train))
    logging.debug("traindata_cls_counts = " + str(traindata_cls_counts))
    train_data_num = sum([len(net_dataidx_map[r]) for r in range(client_number)])

    # get global test data
    if process_id == 0:
        train_data_global, test_data_global = get_dataloader(dataset, data_dir, batch_size, batch_size)
        logging.info("train_dl_global number = " + str(len(train_data_global)))
        logging.info("test_dl_global number = " + str(len(train_data_global)))
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
        train_data_global = None
        test_data_global = None

    return train_data_num, train_data_global, test_data_global, local_data_num, train_data_local, test_data_local, class_num


def load_partition_data_cifar100(dataset, data_dir, partition_method, partition_alpha, client_number, batch_size, val_batch_size=256, load_from_pkl=False, validation_split=0.0, bn_calibration_split=0.0,
                                 client_budgets=None, corr_gamma=1.0, max_training_mac=600_000_000, augmentation="basic"):
    """
    Loads and partitions CIFAR-100 data for federated learning.
    
    Args:
        validation_split: Fraction of training data to hold out for server validation (default 0.0).
        bn_calibration_split: Fraction of validation set for BN calibration (default 0.0 = no split).
    
    Returns:
        train_data_num, val_data_num, test_data_num, train_data_global, val_data_global, 
        test_data_global, data_local_num_dict, train_data_local_dict, test_data_local_dict, 
        class_num, bn_calibration_global, bn_cal_data_num
    """
    X_train, y_train, X_val, y_val, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        dataset,
        data_dir,
        partition_method,
        client_number,
        partition_alpha,
        load_from_pkl,
        validation_split=validation_split,
        client_budgets=client_budgets,
        corr_gamma=corr_gamma,
        max_training_mac=max_training_mac,
    )
    class_num = len(np.unique(y_train))
    logging.debug("traindata_cls_counts = " + str(traindata_cls_counts))
    train_data_num = sum([len(net_dataidx_map[r]) for r in range(client_number)])

    # --- Global Loaders ---
    train_data_global, test_data_global = get_dataloader(dataset, data_dir, batch_size, val_batch_size, dataidxs=None, load_from_pkl=load_from_pkl, augmentation=augmentation)
    logging.info("train_dl_global number = " + str(len(train_data_global)))
    test_data_num = len(test_data_global.dataset)
    logging.info(f"test_dl_global samples = {test_data_num}")
    
    # --- Validation Loader (from held-out training split) ---
    val_data_global = None
    val_data_num = 0
    bn_calibration_global = None
    bn_cal_data_num = 0
    
    if X_val is not None and len(X_val) > 0:
        if augmentation == "mixaug":
            transform_train, transform_test = _data_transforms_cifar100_mixaug()
        else:
            transform_train, transform_test = _data_transforms_cifar100()
        CIFAR100_MEAN = torch.tensor([0.5071, 0.4865, 0.4409]).view(1, 3, 1, 1)
        CIFAR100_STD = torch.tensor([0.2673, 0.2564, 0.2762]).view(1, 3, 1, 1)
        
        # Dataset for evaluation (no augmentation, pre-normalized)
        class ValidationDataset(torch.utils.data.Dataset):
            def __init__(self, X, y, mean, std):
                self.X = (torch.from_numpy(X).permute(0, 3, 1, 2).float() / 255.0 - mean) / std
                self.y = torch.from_numpy(y).long()
            def __len__(self):
                return len(self.y)
            def __getitem__(self, idx):
                return self.X[idx], self.y[idx]
        
        # Dataset for BN calibration (WITH training augmentations)
        class BNCalDataset(torch.utils.data.Dataset):
            def __init__(self, X, y, transform):
                self.X = X
                self.y = torch.from_numpy(y).long()
                self.transform = transform
            def __len__(self):
                return len(self.y)
            def __getitem__(self, idx):
                img = self.X[idx]
                if self.transform:
                    img = self.transform(img)
                return img, self.y[idx]
        
        total_val_samples = len(X_val)
        
        if bn_calibration_split > 0:
            bn_cal_size = int(total_val_samples * bn_calibration_split)
            bn_cal_size = max(bn_cal_size, batch_size)
            
            g = torch.Generator().manual_seed(42)
            indices = torch.randperm(total_val_samples, generator=g).tolist()
            bn_cal_indices = indices[:bn_cal_size]
            eval_indices = indices[bn_cal_size:]
            
            # BN Calibration: uses TRAINING augmentations
            bn_X = X_val[bn_cal_indices]
            bn_y = y_val[bn_cal_indices]
            bn_cal_dataset = BNCalDataset(bn_X, bn_y, transform_train)
            
            # Evaluation: no augmentation
            eval_X = X_val[eval_indices]
            eval_y = y_val[eval_indices]
            val_eval_dataset = ValidationDataset(eval_X, eval_y, CIFAR100_MEAN, CIFAR100_STD)
            
            bn_calibration_global = data.DataLoader(
                bn_cal_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True
            )
            val_data_global = data.DataLoader(val_eval_dataset, batch_size=val_batch_size, shuffle=False)
            
            val_data_num = len(val_eval_dataset)
            bn_cal_data_num = len(bn_cal_dataset)
            logging.info(f"val_dl_global (for eval): {val_data_num} samples (no augmentation)")
            logging.info(f"bn_calibration_global (for BN reset): {bn_cal_data_num} samples (WITH training augmentations)")
        else:
            full_val_dataset = ValidationDataset(X_val, y_val, CIFAR100_MEAN, CIFAR100_STD)
            val_data_global = data.DataLoader(full_val_dataset, batch_size=val_batch_size, shuffle=False)
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
        logging.debug("client_idx = %d, local_sample_number = %d" % (client_idx, local_data_num))

        train_data_local, test_data_local = get_dataloader(dataset, data_dir, batch_size, val_batch_size,
                                                 dataidxs, load_from_pkl, augmentation=augmentation)
        logging.debug("client_idx = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
            client_idx, len(train_data_local), len(test_data_local)))
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local
    
    return (train_data_num, val_data_num, test_data_num, 
            train_data_global, val_data_global, test_data_global, 
            data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num,
            bn_calibration_global, bn_cal_data_num)
