import os, math, multiprocessing
from os.path import join
from copy import copy

import numpy as np
from PIL import Image

import visual_words
from tqdm import tqdm


def get_feature_from_wordmap(opts, wordmap):
    '''
    Compute histogram of visual words.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist: numpy.ndarray of shape (K)
    '''

    K = opts.K
    # ----- TODO -----
    hist = np.zeros(K)
    for i in range(K):
        hist[i] = np.sum(wordmap == i)
    hist = hist / np.sum(hist)
    return hist

def get_feature_from_wordmap_SPM(opts, wordmap):
    '''
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist_all: numpy.ndarray of shape (K*(4^L-1)/3)
    '''
        
    K = opts.K
    L = opts.L
    # ----- TODO -----
    import concurrent.futures

    def compute_histogram(wordmap_sub, K):
        hist = np.zeros(K)
        for k in range(K):
            hist[k] = np.sum(wordmap_sub == k)
        return hist

    hist_all = np.empty(0)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for l in range(L):
            w = wordmap.shape[1] // (2**(-l))
            h = wordmap.shape[0] // (2**(-l))
            for i in range(2**l):
                for j in range(2**l):
                    wordmap_sub = wordmap[int(i*h):int((i+1)*h), int(j*w):int((j+1)*w)]
                    futures.append(executor.submit(compute_histogram, wordmap_sub, K))

        for future in concurrent.futures.as_completed(futures):
            hist_all = np.concatenate((hist_all, future.result()), 0)

    hist_all = hist_all / np.sum(hist_all)
    return hist_all
    
def get_image_feature(opts, img_path, dictionary):
    '''
    Extracts the spatial pyramid matching feature.

    [input]
    * opts      : options
    * img_path  : path of image file to read
    * dictionary: numpy.ndarray of shape (K, 3F)


    [output]
    * feature: numpy.ndarray of shape (K*(4^L-1)/3)
    '''

    # ----- TODO -----
    img = Image.open(img_path)
    img = np.array(img).astype(np.float32)/255
    wordmap = visual_words.get_visual_words(opts, img, dictionary)
    feature = get_feature_from_wordmap_SPM(opts, wordmap)
    return feature
    
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from os.path import join
from tqdm import tqdm

# External function to process a single image
def process_image(file_idx, opts, train_files, dictionary, data_dir):
    return get_image_feature(opts, join(data_dir, train_files[file_idx]), dictionary)

def build_recognition_system(opts, n_worker=1):
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N,M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K,3F)
    '''
    data_dir = opts.data_dir
    out_dir = opts.out_dir
    SPM_layer_num = opts.L

    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    train_labels = np.loadtxt(join(data_dir, 'train_labels.txt'), np.int32)
    dictionary = np.load(join(out_dir, 'dictionary.npy'))

    # Initialize arrays to store features and labels
    features = []
    
    # Use ProcessPoolExecutor to parallelize the processing of images
    with ProcessPoolExecutor(max_workers=n_worker) as executor:
        results = list(tqdm(executor.map(process_image, range(len(train_files)), [opts]*len(train_files), [train_files]*len(train_files), [dictionary]*len(train_files), [data_dir]*len(train_files)), total=len(train_files), desc="Processing images"))

    # Collect features and labels from the results
    for feature in results:
        features.append(feature)
        
    # Convert lists to numpy arrays
    features = np.array(features)
    np.savez_compressed(join(out_dir, 'trained_system.npz'),
        features=features,
        labels=train_labels,
        dictionary=dictionary,
        SPM_layer_num=SPM_layer_num,
    )
    print("Recognition system built and saved.")

# Example usage
# opts = ...  # Define your options object with necessary attributes
# build_recognition_system(opts, n_worker=4)


def distance_to_set(word_hist, histograms):
    '''
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N,K)

    [output]
    * hist_dist: numpy.ndarray of shape (N)
    '''

    # ----- TODO -----
    # computes the distance between 2 histograms. This function computes the histogram
    # intersection similarity between word_hist and each training sample
    # euclidean distance between word_hist and each training sample
    hist_dist = np.sum(np.minimum(word_hist, histograms), axis=1)
    return hist_dist

    pass    
    

from concurrent.futures import ProcessPoolExecutor
from os.path import join
import numpy as np
from PIL import Image
import copy
from tqdm import tqdm

# External function to process a single test image
def process_test_image(params):
    i, test_file, test_opts, dictionary, features, labels, test_labels, data_dir = params
    img_path = join(data_dir, test_file)
    img = Image.open(img_path)
    img = np.array(img).astype(np.float32) / 255
    wordmap = visual_words.get_visual_words(test_opts, img, dictionary)
    test_feature = get_feature_from_wordmap_SPM(test_opts, wordmap)
    
    distances = distance_to_set(test_feature, features)
    predicted_label = labels[np.argmax(distances)]
    true_label = test_labels[i]
    
    return predicted_label, true_label

def evaluate_recognition_system(opts, n_worker=1):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8,8)
    * accuracy: accuracy of the evaluated system
    '''

    data_dir = opts.data_dir
    out_dir = opts.out_dir

    trained_system = np.load(join(out_dir, 'trained_system.npz'))
    dictionary = trained_system['dictionary']

    # using the stored options in the trained system instead of opts.py
    test_opts = copy.deepcopy(opts)
    test_opts.K = dictionary.shape[0]
    test_opts.L = trained_system['SPM_layer_num']

    test_files = open(join(data_dir, 'test_files.txt')).read().splitlines()
    test_labels = np.loadtxt(join(data_dir, 'test_labels.txt'), np.int32)

    # Load trained system information
    features = trained_system['features']
    labels = trained_system['labels']
    SPM_layer_num = trained_system['SPM_layer_num']

    # Initialize confusion matrix
    num_classes = len(np.unique(test_labels))
    conf = np.zeros((num_classes, num_classes))

    # Prepare input parameters for each image processing task
    params = [(i, test_files[i], test_opts, dictionary, features, labels, test_labels, data_dir) 
              for i in range(len(test_files))]

    # Parallel processing of image evaluation
    with ProcessPoolExecutor(max_workers=n_worker) as executor:
        results = list(tqdm(executor.map(process_test_image, params),
                            total=len(test_files),
                            desc="Evaluating images"))

    # Aggregate results and update confusion matrix
    correct_predictions = 0
    for predicted_label, true_label in results:
        conf[true_label, predicted_label] += 1
        if predicted_label == true_label:
            correct_predictions += 1

    # Calculate accuracy
    accuracy = correct_predictions / len(test_labels)

    return conf, accuracy
