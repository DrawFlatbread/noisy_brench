import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import dataset

class dividemix_dataloader():
    def __init__(self, batch_size, num_workers, task=1, sub_task=5):
        self.batch_size = batch_size
        self.num_workers = num_workers

        normalize = transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],
                                         std=[0.2302, 0.2265, 0.2262])
        self.transform_train=transforms.Compose([transforms.Resize(64),
                                      transforms.RandomResizedCrop(56),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      normalize,
                                      ])

        self.transform_val=transforms.Compose([transforms.Resize(64),
                                      transforms.RandomResizedCrop(56),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      normalize,
                                      ])

        self.transform_test=transforms.Compose([transforms.Resize(64),
                                      transforms.CenterCrop(56),
                                      transforms.ToTensor(),
                                      normalize,
                                      ])
        self.task = task
        self.sub_task = sub_task
        if task == 1 or task == 2 or task == 3:
            self.dataset = dataset.custom.TaskDataset
        else:
            self.dataset = dataset.custom.WebDataset


    def run(self, mode, pred=[],prob=[]):
        if mode == 'warmup':
            train_dataset = self.dataset(transform=self.transform_train, task=self.task, sub_task=self.sub_task, mode="train")
            trainloader = DataLoader(
                dataset=train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True)
            return trainloader

        elif mode == 'train':
            labeled_dataset = self.dataset(self.transform_train, mode="labeled", pred=pred, probability=prob, task=self.task, sub_task=self.sub_task)
            labeled_loader = DataLoader(
                dataset=labeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True)
            unlabeled_dataset = self.dataset(transform=self.transform_val, mode="unlabeled", pred=pred, probability=prob, task=self.task, sub_task=self.sub_task)
            unlabeled_loader = DataLoader(
                dataset=unlabeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True)
            return labeled_loader, unlabeled_loader

        elif mode == 'eval_train':
            labeled_dataset = self.dataset(transform=self.transform_train, mode="train", task=self.task, sub_task=self.sub_task)
            labeled_loader = DataLoader(
                dataset=labeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True)
            return labeled_loader

        elif mode == 'val':
            val_dataset = self.dataset(transform=self.transform_val,mode='val', task=self.task, sub_task=self.sub_task)
            val_loader = DataLoader(
                dataset=val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True)
            return val_loader  

        elif mode == 'test':
            test_dataset = self.dataset(transform=self.transform_test, mode='test', task=self.task, sub_task=self.sub_task)
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True)
            return test_loader
        elif mode == 'web_test':
            web_test_dataset = self.dataset(transform=self.transform_test, mode='web_test', task=self.task, sub_task=self.sub_task)
            web_test_loader = DataLoader(
                dataset=web_test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True)
            return web_test_loader
