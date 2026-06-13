import torch
import torch.nn.functional as F
from utils.pyramids import *
from losses.discriminator_loss import DiscriminatorLoss
from losses.discriminator_loss import adversarial_loss
from tqdm import tqdm
from utils.pyramids import GaussianPyramid
from losses.losses import GeneratorLoss, DiscLoss
from losses.losses import SSIMLoss, RGBuvHistBlock


def evaluate(epoch, net, net_D, dataloader, device, ps, loss_weights):
    net.eval()
    num_val_batches = len(dataloader)
    val_loss_generator = 0

    alpha, beta, gamma, delta, epsilon = loss_weights[0], loss_weights[1], loss_weights[2], loss_weights[3], \
                                         loss_weights[4]

    if epsilon > 0:
        intensity_scale = True
        histogram_size = 128
        max_input_size = 512
        method = 'inverse-quadratic'

    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        exp_images, gt_images = batch['exp_image'], batch['gt_image']

        with torch.no_grad():
            # Predictions (forward propagation)
            _, y_pred = net(exp_images)

            GP = GaussianPyramid(4)  # for computing Pyramid Loss
            G_pyramid = GP(gt_images)

            for pyramid in G_pyramid:
                G_pyramid[pyramid] = G_pyramid[pyramid].to(device=device, dtype=torch.float32)

            laplacian_pyr, y_pred = net(exp_images)

            # Critertions for Losses:
            mae_loss = nn.L1Loss()
            bcelog_loss = nn.BCEWithLogitsLoss()  # This already includes sigmoid
            if gamma > 0:
                ssim_loss = SSIMLoss()

            if epsilon > 0:
                # create a histogram block
                histogram_block = RGBuvHistBlock(insz=max_input_size, h=histogram_size, intensity_scale=intensity_scale,
                                                 method=method,
                                                 device=device)

            if (epoch + 1 >= 15) and (ps == 256):

                # Adversarial Loss (only for 256 patches
                # _, y_pred = net(exp_images)
                y_pred_2 = [Y.detach() for Y in y_pred.values()]
                disc_fake = net_D(y_pred_2[-1])
                fake_loss = bcelog_loss(disc_fake, torch.zeros_like(disc_fake))
                disc_real = net_D(G_pyramid['level1'])
                real_loss = bcelog_loss(disc_real, torch.ones_like(disc_real))

                disc_adv = net_D(y_pred['subnet_16'])
                adv_loss = bcelog_loss(disc_adv, torch.ones_like(disc_adv))

            else:

                adv_loss = torch.tensor([[0]]).to(device=device, dtype=torch.float32)

            # COMPUTING LOSSES
            if gamma > 0:
                ssim = ssim_loss(y_pred['subnet_16'], G_pyramid['level1'])
            pyr_loss = 4 * mae_loss(y_pred['subnet_24_1'],
                                    F.interpolate(G_pyramid['level4'], (y_pred['subnet_24_1'].shape[2],
                                                                        y_pred['subnet_24_1'].shape[3]),
                                                  mode='bilinear', align_corners=True)) + \
                       2 * mae_loss(y_pred['subnet_24_2'],
                                    F.interpolate(G_pyramid['level3'], (y_pred['subnet_24_2'].shape[2],
                                                                        y_pred['subnet_24_2'].shape[3]),
                                                  mode='bilinear', align_corners=True)) + \
                       mae_loss(y_pred['subnet_24_3'],
                                F.interpolate(G_pyramid['level2'], (y_pred['subnet_24_3'].shape[2],
                                                                    y_pred['subnet_24_3'].shape[3]),
                                              mode='bilinear', align_corners=True))

            rec_loss = mae_loss(y_pred['subnet_16'], G_pyramid['level1'])

            if epsilon > 0:
                input_hist = histogram_block(y_pred['subnet_16'])
                target_hist = histogram_block(G_pyramid['level1'])
                histo_loss = (1 / np.sqrt(2.0) * (torch.sqrt(torch.sum(
                    torch.pow(torch.sqrt(target_hist) - torch.sqrt(input_hist), 2)))) / input_hist.shape[0])

            # Generator loss with weighted losses:
            loss_generator = alpha * pyr_loss + beta * rec_loss + delta*histo_loss + epsilon * adv_loss

        val_loss_generator += loss_generator

    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return val_loss_generator
    return val_loss_generator / num_val_batches
