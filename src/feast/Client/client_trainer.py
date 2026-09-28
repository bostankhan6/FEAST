from abc import ABC, abstractmethod
from ofa.imagenet_classification.elastic_nn.utils import set_running_statistics
import torch
from torch import nn
import logging


class ClientTrainer(ABC):
    def __init__(self, model, device, args, teacher_model=None):
        self.client_model = model
        self.test_model = None
        self.device = device
        self.args = args
        self.client_idx = None
        self.local_training_data = None
        self.local_test_data = None
        self.local_sample_number = None
        self.seed = 0
        self.teacher_model = teacher_model

    def cross_entropy_loss_with_soft_target(self, pred, soft_target):
        """Eq. kd_loss (Supplementary Sec. D): cross-entropy against a soft
        (already-softmaxed, stop-gradient) teacher distribution, no temperature
        scaling."""
        logsoftmax = nn.LogSoftmax(dim=1)
        return torch.mean(torch.sum(-soft_target * logsoftmax(pred), 1))

    def get_sample_number(self):
        return self.local_sample_number

    def update_local_dataset(
        self, client_idx, local_training_data, local_test_data, local_sample_number,
    ):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number

    def set_model(self, model, arch_bundle=None, client_budget=None, per_step_data=None):
        self.client_model = model
        self.arch_bundle = arch_bundle
        # Sandwich mode only for strategies that train multiple subnets
        sandwich_strategies = ['inverse_kd_sandwich', 'feast']
        self.is_sandwich_mode = (arch_bundle is not None and 
                                  getattr(self.args, 'training_strategy', '') in sandwich_strategies)
        self.client_budget = client_budget
        # Per-step random data: {filtered_cache, filtered_macs, coverage_map, min_macs, max_macs}
        self.per_step_data = per_step_data

    def set_test_model(self, model):
        self.test_model = model

    def local_test(self, use_test_set, skip_bn_reset=False):
        if use_test_set:
            test_data = self.local_test_data
        else:
            test_data = self.local_training_data

        # Skip BN reset if already done globally (efficient evaluation mode)
        if not skip_bn_reset and (self.args.reset_bn_stats or self.args.reset_bn_stats_test):
            n_samples = len(self.local_training_data.dataset)
            subset_size = int(self.args.reset_bn_sample_size * n_samples)
            if self.args.verbose:
                logging.info(f"[BN Reset] Re-calibrating BN stats for model on {subset_size} samples...")
            data_loader = self.random_sub_train_loader(
                subset_size,
                self.args.batch_size,
            )
            # Reset the model that will actually be tested: test_model if set
            # (Subnet), otherwise client_model (Supernet) -- resetting the wrong
            # one silently no-ops BN calibration for whichever model gets evaluated.
            target_model = self.test_model if self.test_model is not None else self.client_model
            model_to_reset = target_model.get_model() if hasattr(target_model, 'get_model') else target_model
            set_running_statistics(model_to_reset, data_loader)
        metrics = self.test(test_data, self.args)
        return metrics

    def random_sub_train_loader(self, subset_size, subset_batch_size):
        n_samples = len(self.local_training_data.dataset)
        g = torch.Generator()
        g.manual_seed(self.seed)
        self.seed += 1
        rand_indexes = torch.randperm(n_samples, generator=g).tolist()
        sub_sampler = torch.utils.data.sampler.SubsetRandomSampler(
            rand_indexes[:subset_size]
        )
        subset = []
        sub_data_loader = torch.utils.data.DataLoader(
            self.local_training_data.dataset,
            batch_size=subset_batch_size,
            sampler=sub_sampler,
            pin_memory=True,
        )
        for images, labels in sub_data_loader:
            subset.append((images, labels))
        return subset

    @abstractmethod
    def test(self, dataset, device, args, **kwargs):
        pass

    @abstractmethod
    def train(self, lr, local_ep, **kwargs):
        pass
