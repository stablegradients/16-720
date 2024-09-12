import os, multiprocessing
from os.path import join, isfile

import numpy as np
from PIL import Image
import scipy.ndimage
import skimage.color
from torch import gradient


def extract_filter_responses(opts, img):
    '''
    Extracts the filter responses for the given image.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    [output]
    * filter_responses: numpy.ndarray of shape (H,W,3F)
    '''

    def scale_channels_to_255(tensor):
        """
        Scales each channel in a 3D tensor to the range [0, 255].

        Parameters:
        tensor (numpy.ndarray): Input 3D tensor to be scaled.

        Returns:
        numpy.ndarray: Scaled tensor with values in the range [0, 255].
        """
        # Initialize an empty array with the same shape as the input tensor
        scaled_tensor = np.zeros_like(tensor, dtype=np.float32)
        
        # Scale each channel independently
        for c in range(tensor.shape[-1]):
            channel = tensor[:, :, c]
            min_val = channel.min()
            max_val = channel.max()
            # Scale the channel to the range [0, 255]
            scaled_tensor[:, :, c] = 255 * (channel - min_val) / (max_val - min_val)
        
        # Convert the scaled tensor to uint8 type
        return scaled_tensor.astype(np.uint8)

    # If the image is grayscale, convert it to RGB by stacking the same channel three times
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)

    # Convert the image from RGB to LAB color space
    img_lab = skimage.color.rgb2lab(img)
    filter_responses = []

    # Apply Gaussian filters at multiple scales
    for scale in opts.filter_scales:
        # Apply Gaussian filter to each channel
        filters = [scipy.ndimage.gaussian_filter(img_lab[:, :, c], sigma=scale) for c in range(3)]
        # Apply Laplacian of Gaussian filter to each channel
        laplace = [scipy.ndimage.gaussian_laplace(img_lab[:, :, c], sigma=scale) for c in range(3)]
        # Apply Gaussian derivative filters to each channel
        grad_x = [scipy.ndimage.gaussian_filter(img_lab[:, :, c], sigma=scale, order=[0, 1]) for c in range(3)]
        grad_y = [scipy.ndimage.gaussian_filter(img_lab[:, :, c], sigma=scale, order=[1, 0]) for c in range(3)]
        
        # Stack the filter responses along the last axis and add to the list
        filter_responses.extend([
            np.stack(filters, axis=-1),
            np.stack(laplace, axis=-1),
            np.stack(grad_x, axis=-1),
            np.stack(grad_y, axis=-1)
        ])

    # Concatenate all filter responses along the last axis
    filter_responses = np.concatenate(filter_responses, axis=-1)
    
    # Scale the filter responses to the range [0, 255]
    return scale_channels_to_255(filter_responses)


def compute_dictionary_one_image(args):
    '''
    Extracts a random subset of filter responses of an image and save it to disk
    This is a worker function called by compute_dictionary

    Your are free to make your own interface based on how you implement compute_dictionary
    '''

    # ----- TODO -----
    pass

def compute_dictionary(opts, n_worker=1):
    '''
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * opts         : options
    * n_worker     : number of workers to process in parallel
    
    [saved]
    * dictionary : numpy.ndarray of shape (K,3F)
    '''

    data_dir = opts.data_dir
    feat_dir = opts.feat_dir
    out_dir = opts.out_dir
    K = opts.K

    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    # ----- TODO -----
    pass

    ## example code snippet to save the dictionary
    # np.save(join(out_dir, 'dictionary.npy'), dictionary)

def get_visual_words(opts, img, dictionary):
    '''
    Compute visual words mapping for the given img using the dictionary of visual words.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    
    [output]
    * wordmap: numpy.ndarray of shape (H,W)
    '''
    
    # ----- TODO -----
    pass

