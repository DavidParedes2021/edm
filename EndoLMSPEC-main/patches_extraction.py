import argparse
import os
import numpy as np
import cv2
from PIL import Image, ImageOps
from sklearn.feature_extraction import image

#For problems with cv2
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main_patch_extraction(dir_path, patches_size_pow):

    dir_path = opt.exposure_dataset
    exp_train = os.path.join(dir_path, 'training', 'exp_images')
    gt_train = os.path.join(dir_path, 'training', 'gt_images')
    exp_valid = os.path.join(dir_path, 'validation', 'exp_images')
    gt_valid = os.path.join(dir_path, 'validation', 'gt_images')

    for i in opt.patches_size_pow:
        # Random patch extraction. Patch size p_size=2^7, 2^8. Number of patches per image p_num_per_im
        ps = 2**i
        p_size = (ps,ps,3)  # shape of the patch ("3" means three channels)
        tr_p_num_per_im = 2**(10 - i) + 2  # Num. of patches per image for Training
        val_p_num_per_im = 2**(10 - i)//2 + 2  # Num. of patches per image for Validation

        # For training:
        rand_patch_extraction(exp_train, gt_train, p_size, tr_p_num_per_im)
        # For validation:
        rand_patch_extraction(exp_valid, gt_valid, p_size, val_p_num_per_im)

    return "All patches done!"


"""Now random patch extraction"""


def rand_patch_extraction(exp_dir, gt_dir, p_size, p_num_per_im):
    # Making new directories for exposed and gt images.
    gt_patch_dir = os.path.join(os.path.split(gt_dir)[0], 'gt_images_PS%d' % p_size[0])
    exp_patch_dir = os.path.join(os.path.split(exp_dir)[0], 'exp_images_PS%d' % p_size[0])

    print(gt_patch_dir)
    print(exp_patch_dir)
    print()

    if not os.path.isdir(exp_patch_dir):
        os.mkdir(exp_patch_dir)
    if not os.path.isdir(gt_patch_dir):
        os.mkdir(gt_patch_dir)

    exp_ims, gt_ims = os.listdir(exp_dir), os.listdir(gt_dir)  # takes all files jpg in gt_dir

    print(set(exp_ims)-set(gt_ims))

    if len(exp_ims) != len(gt_ims):
        raise OSError('Datasets are different size!')

    for i in range(len(gt_ims)):
        print('[%dx%d]: Processing %s (%d/%d)... \n' % (p_size[0], p_size[1], gt_ims[i], i + 1, len(gt_ims)))
        name, ext = gt_ims[i].split('.')

        gt_im_read = cv2.imread(os.path.join(gt_dir, gt_ims[i]))
        exp_im_read = cv2.imread(os.path.join(exp_dir, gt_ims[i]))

        print('Image shape: {}'.format(gt_im_read.shape))

        # Generate the patch. Crop the image.
        gt_patches = image.extract_patches_2d(gt_im_read, (p_size[0], p_size[1]), max_patches=p_num_per_im,
                                              random_state=7)
        exp_patches = image.extract_patches_2d(exp_im_read, (p_size[0], p_size[1]), max_patches=p_num_per_im,
                                               random_state=7)
        zip_patches = zip(gt_patches, exp_patches)
        # Filtering of patches according to their intensity and image gradient.
        count, count1 = 1, 1
        for gt_patch, exp_patch in zip_patches:
            intensity = round(np.sum(exp_patch)/(exp_patch.shape[0]*exp_patch.shape[1]*exp_patch.shape[2]*255), 2)
            patch_RGB = cv2.cvtColor(exp_patch, cv2.COLOR_BGR2RGB)
            patch_pil = Image.fromarray(patch_RGB)
            patch_grayscale = ImageOps.grayscale(patch_pil)


            # Computing x and y gradients for each patch
            grad_x = cv2.Sobel(np.array(patch_grayscale), cv2.CV_16S, 1, 0, ksize=3, scale=1,
                               delta=0, borderType=cv2.BORDER_DEFAULT)
            grad_y = cv2.Sobel(np.array(patch_grayscale), cv2.CV_16S, 0, 1, ksize=3, scale=1,
                               delta=0, borderType=cv2.BORDER_DEFAULT)
            magnitude = np.sqrt(grad_x ** 2.0 + grad_y ** 2.0)
            avg_magnitude = np.sum(magnitude) / (magnitude.shape[0] * magnitude.shape[0] * np.max(magnitude))

            grad_x_ = Image.fromarray(grad_x)
            grad_y_ = Image.fromarray(grad_y)

            # patch_pil.show()
            # patch_grayscale.show()
            # grad_x_.show()
            # grad_y_.show()
            print('intensity is {}'.format(intensity))
            print("magnitude is {}".format(avg_magnitude))

            if (intensity < 0.02) or (intensity > 0.98) or (avg_magnitude < 0.06):
                continue
            else:
                cv2.imwrite(os.path.join(gt_patch_dir, '%s_%d.%s' % (name, count, ext)), gt_patch)
                cv2.imwrite(os.path.join(exp_patch_dir, '%s_%d.%s' % (name, count, ext)), exp_patch)
                count += 1

        print('GT patches shape: {}'.format(gt_patches.shape))
        print('Exp patches shape: {} \n'.format(exp_patches.shape))


if __name__ == '__main__':
    # Adding arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exposure_dataset", type=str, default="exposure_dataset/",
                        help="path to folder with training and valid folders")
    parser.add_argument("--patches_size_pow", nargs='+', required=True, type=int,
                        help="list with the power for the size of the patches")
    opt = parser.parse_args()
    print(opt)

    patches_size_pow = opt.patches_size_pow
    dir_path = opt.exposure_dataset
    main_patch_extraction(dir_path, patches_size_pow)






