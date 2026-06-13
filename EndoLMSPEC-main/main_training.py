# Main script for training.
import argparse
import sys
from train_network import *
from losses.discriminator import Discriminator
from generator import Generator


def get_args():
    # Adding arguments:
    parser = argparse.ArgumentParser(description='Train the Generator-Unet-like on images')
    parser.add_argument("--exposure_dataset", type=str, default="exposure_dataset/",
                        help="path to folder with patches for training and validation.")
    # parser.add_argument("--epochs_list", "-el", nargs='+', required=True, type=int, default=[40, 30],
    #                     help="list with the epochs for: 1)128x128 and 2)256x256, respectively.")
    parser.add_argument("--epochs_list", "-el", nargs='+', type=int, default=[40, 30],
                        help="list with the epochs for: 1)128x128 and 2)256x256, respectively.")
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4,
                                                 help='starting learning rate', dest='lr')
    parser.add_argument('--learning_rate_d', '-lrd', type=float, default=1e-5,
                        help='Starting learning rate for discriminator', dest='lrd')
    parser.add_argument('--load_model', type=str, default=None,
                        help='Directory to the generator model to be loaded.')
    parser.add_argument('--load_D_model', type=str, default=None,
                        help='Directory to the discriminator model to be loaded.')
    parser.add_argument('--chkpnt_period', type=int, default=5, help='Every certain epochs save the models.')
    parser.add_argument("--checkpoint_dir", type=str, default='checkpoint',
                        help="path for saving checkpoints or for loading model from a .pth file")
    parser.add_argument("--GPU", type=int, default=1,
                        help="Select the device")
    parser.add_argument("--patch_sizes","-ps", nargs='+', type=int, default=[128, 256],
                        help="list with the different size of the patches")
    parser.add_argument("--loss_weights", "-lw", nargs='+', type=float, default=[1.0, 1.0, 1.0, 1.0, 0.0],
                        help="list with the loss weights.")
    parser.add_argument("--batch_sizes", "-bs", nargs='+', type=int, default=[32, 8],
                        help="list with the different size of the batches")
    parser.add_argument("--with_discriminator", type=bool, default=True,
                        help="Include discriminator loss term?.")
    parser.parse_args()

    return parser.parse_args()


opt = get_args()

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Specify device either GPU or CPU
if torch.cuda.is_available():
    #device = torch.device('cuda')
    device = torch.device(opt.GPU)
else:
    device = torch.device('cpu')

logging.info(f'Using device {device}')

net = Generator(n_channels=3, device=device, bilinear=False)
net_D = Discriminator()

logging.info(f'Network:\n'
             f'\t{net.n_channels} input channels\n'
             f'\t{net.n_channels} output channels\n'
             f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

if opt.load_model:
    print('Loading the generator model...\n')
    net.load_state_dict(torch.load(os.path.join(opt.load_model), map_location=device))
    logging.info(f'Model loaded from {opt.load_model}')
else:
    print('Creating the generator model...\n')

if opt.with_discriminator:
    if opt.load_D_model:
        print('Loading the discriminator model...\n')
        net_D.load_state_dict(torch.load(os.path.join(opt.load_D_model), map_location=device))
        logging.info(f'Model loaded from {opt.load_D_model}')
    else:
        print('Creating the discriminator model...\n')

net.to(device=device)
net_D.to(device=device)

dataset_dir = opt.exposure_dataset
epochs_list = opt.epochs_list
patch_sizes = opt.patch_sizes
checkpoint_dir = opt.checkpoint_dir
batch_sizes = opt.batch_sizes
loss_weights = opt.loss_weights

print('using device:', device)

for ps in patch_sizes:
    # Here we are going to consider that we only have 128x128 and 256x256 patches. Thus he have two cases.
    if ps == 128:
        drop_rate = 20  # drop learning rate
        checkpoint_period = opt.chkpnt_period  # backup every checkpoint_period
        epochs = epochs_list[0]  # number of epochs for 128x128 case.
        minibatch = batch_sizes[0]   # mini-batch size.


    elif ps == 256:
        drop_rate = 10  # drop learning rate
        checkpoint_period = opt.chkpnt_period//2  # backup every checkpoint_period
        epochs = epochs_list[1]  # number of epochs for 256x256 case.
        minibatch = batch_sizes[1]  # mini-batch size.
        from_chkpoint = os.path.join(checkpoint_dir, 'main_net', 'model_128.pth')
        #D_from_chkpoint = os.path.join(checkpoint_dir, 'disc_net', 'D_model_128.pth')
        if opt.load_model:
            continue
        else:
            net.load_state_dict(torch.load(from_chkpoint, map_location=device))
        #    net_D.load_state_dict(torch.load(D_from_chkpoint, map_location=device))

        # if opt.with_discriminator:
        #     from_chkpoint_d = os.path.join(checkpoint_dir, 'disc_net', 'D_model_128.pth')
        #     net_D.load_state_dict(torch.load(from_chkpoint_d, map_location=device))

    else:
        raise ValueError('Wrong ps value')

    print('Starting training...\n')

    try:
        train_net(net=net,
                  net_D=net_D,
                  epochs=epochs,
                  drop_rate=drop_rate,
                  dir_patches=dataset_dir,
                  ps=ps,
                  batch_size=minibatch,
                  loss_weights=loss_weights,
                  learning_rate=opt.lr,
                  learning_rate_d=opt.lrd,
                  device=device,
                  dir_checkpoint=checkpoint_dir,
                  checkpoint_period=checkpoint_period)

    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED_{}.pth'.format(ps))
        torch.save(net_D.state_dict(), 'D_INTERRUPTED_{}.pth'.format(ps))
        logging.info('Saved interrupt')
        sys.exit(0)