import os
from torch.utils.data import Dataset
from PIL import Image
import torch


class ImageDataset(Dataset):
    """
    Input:
    ImageDataset is a class that take a directory path (img_dir), a patch size (ps), a boolean value for saying if call
    train or valid test (train=True or False), a transformation of the output (i.g., T.ToTensor())
    Output:
    ImageDataset outputs a dictionary containing both exposed images (exp_images) and its corresponding ground truth
    (gt_images). If transform=None, then the value for each is a PIL object (PIL image in RGB).
    """

    def __init__(self, img_dir, ps, train, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.ps = ps
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        if self.train:
            path = os.path.join(self.img_dir, 'training', 'exp_images_PS%d' % self.ps)
        if not self.train:
            path = os.path.join(self.img_dir, 'validation', 'exp_images_PS%d' % self.ps)
        return len(os.listdir(path))

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.train:
            names_train = os.listdir(os.path.join(self.img_dir, 'training', 'exp_images_PS%d' % self.ps))
            exp_path = os.path.join(self.img_dir, 'training', 'exp_images_PS%d' % self.ps)
            exp_im = os.path.join(exp_path, sorted(names_train)[idx])
            gt_path = os.path.join(self.img_dir, 'training', 'gt_images_PS%d' % self.ps)
            gt_im = os.path.join(gt_path, sorted(names_train)[idx])

        if not self.train:
            names_valid = os.listdir(os.path.join(self.img_dir, 'validation', 'exp_images_PS%d' % self.ps))
            exp_path = os.path.join(self.img_dir, 'validation', 'exp_images_PS%d' % self.ps)
            exp_im = os.path.join(exp_path, sorted(names_valid)[idx])
            gt_path = os.path.join(self.img_dir, 'validation', 'gt_images_PS%d' % self.ps)
            gt_im = os.path.join(gt_path, sorted(names_valid)[idx])

        exp_image = Image.open(exp_im)
        gt_image = Image.open(gt_im)
        sample = {'exp_image': exp_image, 'gt_image': gt_image}

        if self.transform:
            # sample = self.transform(sample)
            sample['exp_image'] = self.transform(sample['exp_image'])
            sample['gt_image'] = self.transform(sample['gt_image'])

        return sample


