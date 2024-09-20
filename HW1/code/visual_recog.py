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
    H, W = wordmap.shape
    hist_all = []

    for l in range(L + 1):
        num_cells = 2 ** l
        cell_H = H // num_cells
        cell_W = W // num_cells
        hist_level = []

        for i in range(num_cells):
            for j in range(num_cells):
                cell_wordmap = wordmap[i * cell_H:(i + 1) * cell_H, j * cell_W:(j + 1) * cell_W]
                hist = get_feature_from_wordmap(opts, cell_wordmap)
                hist_level.append(hist)

        hist_level = np.concatenate(hist_level)
        weight = 2 ** (l - L) if l != 0 else 2 ** (-L)
        hist_all.append(hist_level * weight)

    hist_all = np.concatenate(hist_all)
    hist_all = hist_all / np.sum(hist_all)  # Normalize the histogram

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
    try:
        return get_image_feature(opts, join(data_dir, train_files[file_idx]), dictionary)
    except Exception as e:
        print(f"Error processing image {file_idx}: {e}")
        return None

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
        futures = [executor.submit(process_image, i, opts, train_files, dictionary, data_dir) for i in range(len(train_files))]
        for future in futures:
            result = future.result()
            if result is not None:
                features.append(result)

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

    # dum predicted labels, true labels and test_image names to a single text file
    with open(join(out_dir, 'predicted_labels.txt'), 'w') as f:
        for i in range(len(results)):
            f.write(f"{test_files[i]} {results[i][0]} {results[i][1]}\n")
    # Calculate accuracy
    accuracy = correct_predictions / len(test_labels)

    return conf, accuracy
