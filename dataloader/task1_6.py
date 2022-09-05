import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os, sys


class timyimagenet_dataset(Dataset):
    def __init__(self, transform, mode, pred=[], probability=[]):
        self.transform = transform
        self.mode = mode

        self.val_dir = 'data/tiny-imagenet-200/val'

        if self.mode == 'test':
            self.test_imgs = []
            with open('data/task1/6/data/test_label1.txt','r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    entry = l.split()           
                    img_path = 'data/task1/6/data/test/%s' % (entry[0])
                    self.test_imgs.append(img_path)

        else:
            train_imgs = []
            noise_label = []

            with open('data/task1/6/data/label.txt', 'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    entry = l.split()           
                    img_path = 'data/task1/6/data/train/%s' % (entry[0])
                    train_imgs.append(img_path)
                    noise_label.append( int(entry[1]) ) 

            if self.mode == 'train' or self.mode == 'val':
                self.train_imgs = train_imgs
                self.noise_label = noise_label
            else:
                if self.mode == 'labeled':
                    pred_idx = pred.nonzero()[0]
                    self.probability = [probability[i] for i in pred_idx]   
                    
                    clean = (np.array(noise_label)==np.array(train_label))                                                       
                    auc_meter = AUCMeter()
                    auc_meter.reset()
                    auc_meter.add(probability,clean)        
                    auc,_,_ = auc_meter.value()               
                    sys.stdout.write('Numer of labeled samples:%d   AUC:%.3f\n'%(pred.sum(),auc))
                elif self.mode == "unlabeled":
                    pred_idx = (1-pred).nonzero()[0]                                               
                
                self.train_imgs = train_imgs[pred_idx]
                self.noise_label = [noise_label[i] for i in pred_idx]                          
                print("%s data has a size of %d"%(self.mode,len(self.noise_label)))     


    def __getitem__(self, index):
        if self.mode=='labeled':
            img, target, prob = self.train_imgs[index], self.noise_label[index], self.probability[index]
            img = Image.fromarray(img)
            img1 = self.transform(img) 
            img2 = self.transform(img) 
            return img1, img2, target, prob            
        elif self.mode=='unlabeled':
            img = self.train_imgs[index]
            img = Image.fromarray(img)
            img1 = self.transform(img) 
            img2 = self.transform(img) 
            return img1, img2

        if self.mode == 'train':
            img_path = self.train_imgs[index]
            target = self.noise_label[index]
            image = Image.open( img_path ).convert('RGB')
            img = self.transform(image)
            return img, target, index

        elif self.mode == 'val':
            img_path = self.train_imgs[index]  # 此处和train一致（dividemix）
            target = self.noise_label[index]
            image = Image.open( img_path ).convert('RGB')
            img = self.transform(image)
            return img, target

        elif self.mode == 'test':
            img_path = self.test_imgs[index]
            image = Image.open( img_path ).convert('RGB')
            img = self.transform(image)
            return img

    def __len__(self):
        if self.mode == 'test':
            return len(self.test_imgs)
        else:
            return len(self.train_imgs)


class tinyimagenet_dataloader():
    def __init__(self, batch_size, num_workers):
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

    def run(self, mode, pred=[],prob=[]):
        if mode == 'warmup':
            train_dataset = timyimagenet_dataset(transform=self.transform_train, mode="train")
            trainloader = DataLoader(
                dataset=train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True)
            return trainloader
        elif mode == 'train':
            labeled_dataset = timyimagenet_dataset(transform=self.transform_train, mode="train", pred=pred, probability=prob)
            labeled_loader = DataLoader(
                dataset=labeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True)
            unlabeled_dataset = timyimagenet_dataset(transform=self.transform_val, mode="val", pred=pred, probability=prob)
            unlabeled_loader = DataLoader(
                dataset=unlabeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True)
            return labeled_loader, unlabeled_loader

        elif mode == 'eval_train':
            labeled_dataset = timyimagenet_dataset(transform=self.transform_train, mode="train")
            labeled_loader = DataLoader(
                dataset=labeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True)

        elif mode == 'val':
            val_dataset = timyimagenet_dataset(transform=self.transform_val,mode='val')
            val_loader = DataLoader(
                dataset=val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True)
            return val_loader  

        elif mode == 'test':
            test_dataset = timyimagenet_dataset(transform=self.transform_test, mode='test')
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True)
            return test_loader



