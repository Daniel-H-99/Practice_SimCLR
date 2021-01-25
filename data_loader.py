import numpy as np
import os
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
from gaussian_blur import GaussianBlur
from torchvision import datasets

class DataSetWrapper(object):
    def __init__(self, args, batch_size, num_workers, valid_size, input_shape):
        self.args = args
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.input_shape = input_shape
        
    def get_data_loaders(self):
        data_augment = self._simclr_transform()

        train_dataset = datasets.STL10(os.path.join(self.args.path, self.args.data_dir), split='train+unlabeled', download=True,
                                       transform=SimCLRDataTransform(data_augment))
        train_loader, valid_loader = self.get_train_validation_data_loaders(train_dataset)
        return train_loader, valid_loader
    
    def get_finetune_data_loaders(self):
        finetune_augment = transforms.Compose([
                                    transforms.Resize(self.args.resize),
                                    transforms.RandomResizedCrop(self.args.imgsize),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        test_augment = transforms.Compose([
                                    transforms.Resize(self.args.resize),
                                    transforms.CenterCrop(self.args.imgsize),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        finetune_dataset = datasets.STL10(os.path.join(self.args.path, self.args.data_dir), split='train', download=True, transform=finetune_augment)       
        test_dataset = datasets.STL10(os.path.join(self.args.path, self.args.data_dir), split='test', download=True, transform=test_augment)
        
        finetune_loader = DataLoader(finetune_dataset, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=False, shuffle=False)
        
        return finetune_loader, test_loader

    def _simclr_transform(self):

        data_transforms = transforms.Compose([
                                  transforms.Resize(self.args.resize),
                                  transforms.RandomResizedCrop(self.args.imgsize),
                                  transforms.ColorJitter(0.5, 0.5, 0.5, 0.5),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        gaussian_blur = GaussianBlur(kernel_size = int(0.1* self.input_shape[0]))

        return data_transforms

    def get_train_validation_data_loaders(self, train_dataset):
        # obtain training indices that will be used for validation
        num_train = len(train_dataset)
        indices = list(range(num_train))
        np.random.shuffle(indices)

        split = int(np.floor(self.valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]

        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler,
                                  num_workers=self.num_workers, drop_last=True, shuffle=False)

        valid_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=valid_sampler,
                                  num_workers=self.num_workers, drop_last=True)
        return train_loader, valid_loader


class SimCLRDataTransform(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        xi = self.transform(sample)
        xj = self.transform(sample)
        return xi, xj
