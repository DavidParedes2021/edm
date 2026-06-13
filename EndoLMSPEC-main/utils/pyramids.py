import cv2
import numpy as np
import torchvision.transforms as T
import torch.nn as nn
import torch


class LaplacianPyramid(nn.Module):

    def __init__(self, levels):
        """
        Class for creating a Laplacian Pyramid from a batch of patches with size 'ps'.
        level: number of levels in the laplacian pyramid.
        """
        super(LaplacianPyramid, self).__init__()

        self.levels = levels

    def forward(self, batch):
        """
        batch: inputs a batch of patches of size 'ps' (e.g., 128x128) from an image. This batch comes from DataLoader as a pytorch
               tensor, therefore first each image must be converted to PIL and then to cv2 format for using cv2 libraries
               for Laplacian Pyramid.
        """

        transform = T.ToTensor()
        trans_PIL = T.ToPILImage()

        pyramids_dir = {}
        L4, L3, L2, L1 = [], [], [], []

        if len(batch.size())==3:
            batch = batch[None,:,:,:]

        for patch in batch:
            patch = trans_PIL(patch)
            patch = patch.convert('RGB')
            patch = np.array(patch)

            # generate Gaussian pyramid for the patch.
            G = patch.copy()
            gp = [G]

            for i in range(1, self.levels):
                G = cv2.pyrDown(G)
                gp.append(G)

            # Laplacian Pyramid
            lp = [transform(gp[3])]
            for i in range(3, 0, -1):
                GE = cv2.pyrUp(gp[i])
                L = cv2.subtract(gp[i - 1], GE)
                lp.append(transform(L))

            lp.reverse()  # list with LP reverse sorted -> L4, L3, L2, L1
            # print(type(lp[0]))
            L4.append(lp[3])
            L3.append(lp[2])
            L2.append(lp[1])
            L1.append(lp[0])

        pyramids_dir["level4"], pyramids_dir["level3"] = torch.stack(L4, dim=0), torch.stack(L3, dim=0)
        pyramids_dir["level2"], pyramids_dir["level1"] = torch.stack(L2, dim=0), torch.stack(L1, dim=0)

        # form the batch of pyramids

        return pyramids_dir


class GaussianPyramid(nn.Module):

    def __init__(self, levels):
        """
        Class for creating a Gaussian Pyramid from a patch size 'ps'.
        level: number of levels in the laplacian pyramid.
        """
        super(GaussianPyramid, self).__init__()

        self.levels = levels

    def forward(self, batch):
        """
        patch: input patch size 'ps' (e.g., 128x128) from an image. This image comes from DataLoader as a pytorch
               tensor, therefore first must be converted to PIL and then to cv2 format for using cv2 libraries
               for Gaussian Pyramid. Output a list of images of Gaussian Pyramid G0 is the original image G3 is the
               smallest.
        """

        transform = T.ToTensor()
        trans_PIL = T.ToPILImage()

        pyramids_dir = {}
        G0, G1, G2, G3 = [], [], [], []
        for patch in batch:
            patch = trans_PIL(patch)
            patch = patch.convert('RGB')
            patch = np.array(patch)

            # generate Gaussian pyramid for the patch.
            G = patch.copy()
            gp = [transform(G)]

            for i in range(1, self.levels):
                G = cv2.pyrDown(G)
                gp.append(transform(G))

            G0.append(gp[0])
            G1.append(gp[1])
            G2.append(gp[2])
            G3.append(gp[3])

        pyramids_dir["level1"], pyramids_dir["level2"] = torch.stack(G0, dim=0), torch.stack(G1, dim=0)
        pyramids_dir["level3"], pyramids_dir["level4"] = torch.stack(G2, dim=0), torch.stack(G3, dim=0)

        return pyramids_dir