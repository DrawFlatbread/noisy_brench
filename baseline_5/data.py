import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os, sys


class timyimagenet_dataset(Dataset):
    def __init__(self, transform, mode):
        self.transform = transform
        self.mode = mode

        self.val_dir = 'data/tiny-imagenet-200/val'

        if self.mode == 'val':
            self._create_class_idx_dict_val()
            self._make_dataset()
            words_file = 'data/tiny-imagenet-200/words.txt'
            wnids_file = 'data/tiny-imagenet-200/wnids.txt'

            self.set_nids = set()

            with open(wnids_file, 'r') as fo:
                data = fo.readlines()
                for entry in data:
                    self.set_nids.add(entry.strip("\n"))

            self.class_to_label = {}
            with open(words_file, 'r') as fo:
                data = fo.readlines()
                for entry in data:
                    words = entry.split("\t")
                    if words[0] in self.set_nids:
                        self.class_to_label[words[0]] = (words[1].strip("\n").split(","))[0]
        elif self.mode == 'test':
            self.test_imgs = []
            with open('data/task1/5/data/test_label1.txt','r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    entry = l.split()           
                    img_path = 'data/task1/5/data/test/%s' % (entry[0])
                    self.test_imgs.append(img_path)

        else:
            train_imgs = []
            noise_label = []

            with open('data/task1/5/data/label.txt', 'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    entry = l.split()           
                    img_path = 'data/task1/5/data/train/%s' % (entry[0])
                    train_imgs.append(img_path)
                    noise_label.append( int(entry[1]) ) 

            if self.mode == 'all':
                self.train_imgs = train_imgs
                self.noise_label = noise_label

    def _create_class_idx_dict_val(self):
        val_image_dir = os.path.join(self.val_dir, "images")
        if sys.version_info >= (3, 5):
            images = [d.name for d in os.scandir(val_image_dir) if d.is_file()]
        val_annotations_file = os.path.join(self.val_dir, "val_annotations.txt")
        self.val_img_to_class = {}
        set_of_classes = set()
        with open(val_annotations_file, 'r') as fo:
            entry = fo.readlines()
            for data in entry:
                words = data.split("\t")
                self.val_img_to_class[words[0]] = words[1]
                set_of_classes.add(words[1])

        self.len_val_dataset = len(list(self.val_img_to_class.keys()))
        classes = sorted(list(set_of_classes))
        # self.idx_to_class = {i:self.val_img_to_class[images[i]] for i in range(len(images))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}
        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}

    def _make_dataset(self):
        self.val_imgs = []
        img_root_dir = self.val_dir
        list_of_dirs = ["images"]

        for tgt in list_of_dirs:
            dirs = os.path.join(img_root_dir, tgt)
            if not os.path.isdir(dirs):
                continue

            for root, _, files in sorted(os.walk(dirs)):
                for fname in sorted(files):
                    if (fname.endswith(".JPEG")):
                        path = os.path.join(root, fname)
                        item = (path, self.class_to_tgt_idx[self.val_img_to_class[fname]])
                        self.val_imgs.append(item)

    def __getitem__(self, index):
        if self.mode == 'all':
            img_path = self.train_imgs[index]
            target = self.noise_label[index]
            image = Image.open( img_path ).convert('RGB')
            img = self.transform(image)
            return img, target, index

        elif self.mode == 'val':
            img_path, tgt = self.val_imgs[index]
            image = Image.open( img_path ).convert('RGB')
            img = self.transform(image)
            return img, tgt

        elif self.mode == 'test':
            img_path = self.test_imgs[index]
            image = Image.open( img_path ).convert('RGB')
            img = self.transform(image)
            return img

    def __len__(self):
        if self.mode == 'val':
            return self.len_val_dataset
        if self.mode != 'test':
            return len(self.train_imgs)
        else:
            return len(self.test_imgs)


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

    def run(self, mode):
        if mode == 'warmup':
            all_dataset = timyimagenet_dataset(transform=self.transform_train, mode="all")
            trainloader = DataLoader(
                dataset=all_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True)
            return trainloader

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



