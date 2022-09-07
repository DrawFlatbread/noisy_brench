from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class TaskDataset(Dataset):
    def __init__(self, transform, mode, pred=[], probability=[], task=1, sub_task=5):
        self.transform = transform
        self.mode = mode

        self.annotation_path = 'data/task{}/{}/data/'.format(task, sub_task)

        print("| Training on datasts : " + self.annotation_path + " |")
        if self.mode == 'test':
            self.test_imgs = []
            with open(self.annotation_path + 'test_label1.txt','r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    entry = l.split()           
                    img_path = self.annotation_path + 'test/%s' % (entry[0])
                    self.test_imgs.append(img_path)

        else:
            train_imgs = []
            noise_label = []

            with open(self.annotation_path + 'label.txt', 'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    entry = l.split()           
                    img_path = self.annotation_path + 'train/%s' % (entry[0])
                    train_imgs.append(img_path)
                    noise_label.append( int(entry[1]) ) 

            if self.mode == 'train' or self.mode == 'val':
                self.train_imgs = train_imgs
                self.noise_label = noise_label
            else:
                if self.mode == 'labeled':
                    pred_idx = pred.nonzero()[0]
                    self.probability = [probability[i] for i in pred_idx]   
                    
                elif self.mode == "unlabeled":
                    pred_idx = (1-pred).nonzero()[0]                                               
                
                self.train_imgs = np.array(train_imgs)[pred_idx].tolist()
                self.noise_label = [noise_label[i] for i in pred_idx]                          
                print("%s data has a size of %d"%(self.mode,len(self.noise_label)))     


    def __getitem__(self, index):
        if self.mode=='labeled':
            img, target, prob = self.train_imgs[index], self.noise_label[index], self.probability[index]
            img = Image.open(img).convert('RGB')
            img1 = self.transform(img) 
            img2 = self.transform(img) 
            return img1, img2, target, prob            
        elif self.mode=='unlabeled':
            img = self.train_imgs[index]
            img = Image.open(img).convert('RGB')
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


