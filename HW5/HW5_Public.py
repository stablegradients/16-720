#!/usr/bin/env python
# coding: utf-8

# # 16-720 HW6: Photometric Stereo

# #### **For each question please refer to the handout for more details.**
# 
# Programming questions begin at **Q1**. **Remember to run all cells** and save the notebook to your local machine as a pdf for gradescope submission.

# # Collaborators
# **List your collaborators for all questions here**:
# 
# 
# ---

# # Utils and Imports

# Importing all necessary libraries.
# 

# In[1]:


import numpy as np
from matplotlib import pyplot as plt
from skimage.color import rgb2xyz
import warnings
from scipy.ndimage import gaussian_filter
from matplotlib import cm
from skimage.io import imread
from scipy.sparse import kron as spkron
from scipy.sparse import eye as speye
from scipy.sparse.linalg import lsqr as splsqr
import os
import shutil


# Downloading the data

# In[ ]:


if os.path.exists('./content/data'):
  shutil.rmtree('./content/data')

os.mkdir('./content/data')
os.system("rm ./content/data/data.zip")



# In[9]:


get_ipython().system("wget 'https://docs.google.com/uc?export=download&id=13nA1Haq6bJz0-h_7NmovvSRrRD76qiF0' -O ./content/data/data.zip")
get_ipython().system('unzip "./content/data/data.zip" -d "./content/"')


# Utils Functions.

# In[10]:


def integrateFrankot(zx, zy, pad = 512):

    """
    Question 1 (j)

    Implement the Frankot-Chellappa algorithm for enforcing integrability
    and normal integration

    Parameters
    ----------
    zx : numpy.ndarray
        The image of derivatives of the depth along the x image dimension

    zy : tuple
        The image of derivatives of the depth along the y image dimension

    pad : float
        The size of the full FFT used for the reconstruction

    Returns
    ----------
    z: numpy.ndarray
        The image, of the same size as the derivatives, of estimated depths
        at each point

    """

    # Raise error if the shapes of the gradients don't match
    if not zx.shape == zy.shape:
        raise ValueError('Sizes of both gradients must match!')

    # Pad the array FFT with a size we specify
    h, w = 512, 512

    # Fourier transform of gradients for projection
    Zx = np.fft.fftshift(np.fft.fft2(zx, (h, w)))
    Zy = np.fft.fftshift(np.fft.fft2(zy, (h, w)))
    j = 1j

    # Frequency grid
    [wx, wy] = np.meshgrid(np.linspace(-np.pi, np.pi, w),
                           np.linspace(-np.pi, np.pi, h))
    absFreq = wx**2 + wy**2

    # Perform the actual projection
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        z = (-j*wx*Zx-j*wy*Zy)/absFreq

    # Set (undefined) mean value of the surface depth to 0
    z[0, 0] = 0.
    z = np.fft.ifftshift(z)

    # Invert the Fourier transform for the depth
    z = np.real(np.fft.ifft2(z))
    z = z[:zx.shape[0], :zx.shape[1]]

    return z


def enforceIntegrability(N, s, sig = 3):

    """
    Question 2 (e)

    Find a transform Q that makes the normals integrable and transform them
    by it

    Parameters
    ----------
    N : numpy.ndarray
        The 3 x P matrix of (possibly) non-integrable normals

    s : tuple
        Image shape

    Returns
    -------
    Nt : numpy.ndarray
        The 3 x P matrix of transformed, integrable normals
    """

    N1 = N[0, :].reshape(s)
    N2 = N[1, :].reshape(s)
    N3 = N[2, :].reshape(s)

    N1y, N1x = np.gradient(gaussian_filter(N1, sig), edge_order = 2)
    N2y, N2x = np.gradient(gaussian_filter(N2, sig), edge_order = 2)
    N3y, N3x = np.gradient(gaussian_filter(N3, sig), edge_order = 2)

    A1 = N1*N2x-N2*N1x
    A2 = N1*N3x-N3*N1x
    A3 = N2*N3x-N3*N2x
    A4 = N2*N1y-N1*N2y
    A5 = N3*N1y-N1*N3y
    A6 = N3*N2y-N2*N3y

    A = np.hstack((A1.reshape(-1, 1),
                   A2.reshape(-1, 1),
                   A3.reshape(-1, 1),
                   A4.reshape(-1, 1),
                   A5.reshape(-1, 1),
                   A6.reshape(-1, 1)))

    AtA = A.T.dot(A)
    W, V = np.linalg.eig(AtA)
    h = V[:, np.argmin(np.abs(W))]

    delta = np.asarray([[-h[2],  h[5], 1],
                        [ h[1], -h[4], 0],
                        [-h[0],  h[3], 0]])
    Nt = np.linalg.inv(delta).dot(N)

    return Nt

def plotSurface(surface, suffix=''):

    """
    Plot the depth map as a surface

    Parameters
    ----------
    surface : numpy.ndarray
        The depth map to be plotted

    suffix: str
        suffix for save file

    Returns
    -------
        None

    """
    x, y = np.meshgrid(np.arange(surface.shape[1]),
                       np.arange(surface.shape[0]))
    fig = plt.figure()
    #ax = fig.gca(projection='3d')
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(x, y, -surface, cmap = cm.coolwarm,
                           linewidth = 0, antialiased = False)
    ax.view_init(elev = 60., azim = 75.)
    plt.savefig(f'faceCalibrated{suffix}.png')
    plt.show()

def loadData(path = "../data/"):

    """
    Question 1 (c)

    Load data from the path given. The images are stored as input_n.tif
    for n = {1...7}. The source lighting directions are stored in
    sources.mat.

    Paramters
    ---------
    path: str
        Path of the data directory

    Returns
    -------
    I : numpy.ndarray
        The 7 x P matrix of vectorized images

    L : numpy.ndarray
        The 3 x 7 matrix of lighting directions

    s: tuple
        Image shape

    """
    I = None
    L = None
    s = None

    L = np.load(path + 'sources.npy').T

    im = imread(path + 'input_1.tif')
    P = im[:, :, 0].size
    s = im[:, :, 0].shape

    I = np.zeros((7, P))
    for i in range(1, 8):
        im = imread(path + 'input_' + str(i) + '.tif')
        im = rgb2xyz(im)[:, :, 1]
        I[i-1, :] = im.reshape(-1,)

    return I, L, s

def displayAlbedosNormals(albedos, normals, s):

    """
    Question 1 (e)

    From the estimated pseudonormals, display the albedo and normal maps

    Please make sure to use the `coolwarm` colormap for the albedo image
    and the `rainbow` colormap for the normals.

    Parameters
    ----------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    -------
    albedoIm : numpy.ndarray
        Albedo image of shape s

    normalIm : numpy.ndarray
        Normals reshaped as an s x 3 image

    """
    albedoIm = None
    normalIm = None

    albedoIm = albedos.reshape(s)
    normalIm = (normals.T.reshape((s[0], s[1], 3))+1)/2

    plt.figure()
    plt.imshow(albedoIm, cmap = 'gray')

    plt.figure()
    plt.imshow(normalIm, cmap = 'rainbow')

    plt.show()

    return albedoIm, normalIm


# # Q1: Calibrated photometric stereo (75 points)
# 

# ### Q 1 (a): Understanding n-dot-l lighting (5 points)
# 
# ---
# 
# Understanding the n·l Lighting Model
# 
# The n·l lighting model is a fundamental concept in computer graphics used to simulate diffuse reflection on surfaces. It calculates the intensity of light that a surface point reflects based on the angle between the surface normal and the light source direction.
# 
# Where Does the Dot Product Come From?
# 
# The dot product between the normal vector n and the light direction vector l computes the cosine of the angle θ between these two vectors:
# 
# ￼
# 
# Since we usually normalize these vectors (so that ￼), the dot product simplifies to:
# 
# ￼
# 
# This cosine term is crucial because it represents how much the surface is facing the light source. A larger cosine value (closer to 1) means the surface is directly facing the light, resulting in maximum illumination. A smaller cosine value (closer to 0) means the surface is oriented away from the light, resulting in less illumination.
# 
# Role of Projected Area (Fig. 2b)
# 
# The concept of projected area helps explain why the cosine of the angle affects the lighting. Imagine a small patch of the surface with area dA. When light hits this patch at an angle θ, the effective area that “receives” the light is the projection of dA onto a plane perpendicular to the light direction, which is:
# 
# ￼
# 
# This projected area determines how much light the surface patch intercepts. Therefore, the intensity of the diffuse reflection is proportional to ￼, which is directly computed by the dot product ￼.
# 
# Why Does the Viewing Direction Not Matter?
# 
# In the diffuse (Lambertian) reflection model, the surface scatters light uniformly in all directions. This means that the intensity of the reflected light is the same regardless of the viewer’s position. Therefore, the viewing direction does not affect the calculation of diffuse lighting in the n·l model.
# 
# 
# ---
# 

# ### Q 1 (b): Rendering the n-dot-l lighting (10 points)

# In[14]:


data_dir = './content/data/'


# In[15]:


def renderNDotLSphere(center, rad, light, pxSize, res):
    """
    Question 1 (b)

    Render a hemispherical bowl with a given center and radius. Assume that
    the hollow end of the bowl faces in the positive z direction, and the
    camera looks towards the hollow end in the negative z direction. The
    camera's sensor axes are aligned with the x- and y-axes.

    Parameters
    ----------
    center : numpy.ndarray
        The center of the hemispherical bowl in an array of size (3,)

    rad : float
        The radius of the bowl

    light : numpy.ndarray
        The direction of incoming light

    pxSize : float
        Pixel size

    res : numpy.ndarray
        The resolution of the camera frame

    Returns
    -------
    image : numpy.ndarray
        The rendered image of the hemispherical bowl
    """
    
    [X, Y] = np.meshgrid(np.arange(res[0]), np.arange(res[1]))
    X = (X - res[0]/2) * pxSize * 1.e-4
    Y = (Y - res[1]/2) * pxSize * 1.e-4
    Z = np.sqrt(rad**2 + 0j - X**2 - Y**2)

    # Mask out invalid points where Z is zero or imaginary
    valid = np.isreal(Z)
    Z = np.real(Z)
    Z[~valid] = 0
    X[~valid] = 0
    Y[~valid] = 0

    ### YOUR CODE HERE
    # Stack the coordinates to form position vectors
    normals = np.dstack((X, Y, Z))

    # Normalize the normals
    normals_norm = np.linalg.norm(normals, axis=2, keepdims=True)
    # Avoid division by zero
    normals_norm[normals_norm == 0] = 1
    normals = normals / normals_norm

    # Normalize the light direction
    light_dir = light / np.linalg.norm(light)

    # Compute the dot product between normals and light direction
    dot_product = np.dot(normals, light_dir)

    # Clamp the values to [0, 1]
    dot_product = np.clip(dot_product, 0, 1)

    # Initialize the image and assign the computed intensities
    image = dot_product.squeeze()
    ### END YOUR CODE

    return image
# Part 1(b)
radius = 0.75 # cm
center = np.asarray([0, 0, 0]) # cm
pxSize = 7 # um
res = (3840, 2160)

light = np.asarray([1, 1, 1])/np.sqrt(3)
image = renderNDotLSphere(center, radius, light, pxSize, res)
plt.figure()
plt.imshow(image, cmap = 'gray')
plt.imsave('1b-a.png', image, cmap = 'gray')

light = np.asarray([1, -1, 1])/np.sqrt(3)
image = renderNDotLSphere(center, radius, light, pxSize, res)
plt.figure()
plt.imshow(image, cmap = 'gray')
plt.imsave('1b-b.png', image, cmap = 'gray')

light = np.asarray([-1, -1, 1])/np.sqrt(3)
image = renderNDotLSphere(center, radius, light, pxSize, res)
plt.figure()
plt.imshow(image, cmap = 'gray')
plt.imsave('1b-c.png', image, cmap = 'gray')

I, L, s = loadData(data_dir)


# ### Q 1 (c): Initials (10 points)
# 

# In[16]:


### YOUR CODE HERE
# Perform Singular Value Decomposition
U, S, Vt = np.linalg.svd(I, full_matrices=False)

# Report the singular values
print("Singular values of I:")
print(S)

# Plot the singular values
plt.figure(figsize=(8, 5))
plt.plot(S, 'o-', linewidth=2)
plt.title('Singular Values of I')
plt.xlabel('Index')
plt.ylabel('Singular Value')
plt.grid(True)
plt.show()
### END YOUR CODE


# 
# ---
# Discussion:
# 
# The singular value decomposition (SVD) of the matrix ￼ reveals that the first three singular values are orders of magnitude larger than the remaining ones. This indicates that the majority of the information contained in ￼ can be effectively captured using just three components, aligning with the theoretical expectation of a rank-3 matrix in photometric stereo. However, the smaller singular values are not exactly zero, which can be attributed to several practical factors. These include noise in the data due to sensor imperfections, deviations from the idealized model assumptions—such as surfaces not being perfectly Lambertian or lighting conditions not being ideal—and quantization errors inherent in digital image acquisition. In conclusion, while the singular values support the theoretical rank-3 requirement, the presence of small but non-zero singular values in practice highlights minor variations and inaccuracies not accounted for by the idealized model.
# 
# ---

# ### Q 1 (d) Estimating pseudonormals (20 points)

# In[17]:


def estimatePseudonormalsCalibrated(I, L):
    """
    Question 1 (d)

    In calibrated photometric stereo, estimate pseudonormals from the
    light direction and image matrices

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P array of vectorized images

    L : numpy.ndarray
        The 3 x 7 array of lighting directions

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pseudonormals
    """

    ### YOUR CODE HERE
    # Since L.T is of size (7 x 3) and I is of size (7 x P),
    # we can solve for B using least squares for all pixels at once.

    # Solve the linear system A x = y in a least-squares sense
    B, residuals, rank, s = np.linalg.lstsq(L.T, I, rcond=None)

    ### END YOUR CODE

    return B

# Part 1(e)
B = estimatePseudonormalsCalibrated(I, L)


# ---
# 
# YOUR ANSWER HERE...
# 
# ---
# 

# ### Q 1 (e) Albedos and normals (10 points)

# ---
# 
# YOUR ANSWER HERE...
# 
# ---

# In[19]:


def estimateAlbedosNormals(B):
    '''
    Question 1 (e)

    From the estimated pseudonormals, estimate the albedos and normals

    Parameters
    ----------
    B : numpy.ndarray
        The 3 x P matrix of estimated pseudonormals

    Returns
    -------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals
    '''

    ### YOUR CODE HERE
    # Compute the albedos as the magnitude of the pseudonormals
    albedos = np.linalg.norm(B, axis=0)

    # Avoid division by zero
    albedos_nonzero = albedos.copy()
    albedos_nonzero[albedos_nonzero == 0] = 1

    # Compute the normals by normalizing the pseudonormals
    normals = B / albedos_nonzero
    ### END YOUR CODE

    return albedos, normals


# Part 1(e)
albedos, normals = estimateAlbedosNormals(B)
albedoIm, normalIm = displayAlbedosNormals(albedos, normals, s)
plt.imsave('1f-a.png', albedoIm, cmap = 'gray')
plt.imsave('1f-b.png', normalIm, cmap = 'rainbow')


# ### Q 1 (f): Normals and depth (5 points)
# 
# ---
# 
# YOUR ANSWER HERE...
# 
# ---

# ### Q 1 (g): Understanding integrability of gradients (5 points)
# 
# ---
# 
# YOUR ANSWER HERE...
# 
# ---

# ### Q 1 (h): Shape estimation (10 points)

# In[20]:


def estimateShape(normals, s):
    """
    Question 1 (h)

    Integrate the estimated normals to get an estimate of the depth map
    of the surface.

    Parameters
    ----------
    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    ----------
    surface: numpy.ndarray
        The image, of size s, of estimated depths at each point

    """

    ### YOUR CODE HERE
    # Reshape normals into images
    n_x = normals[0, :].reshape(s)
    n_y = normals[1, :].reshape(s)
    n_z = normals[2, :].reshape(s)

    # Avoid division by zero
    n_z_safe = n_z.copy()
    n_z_safe[n_z_safe == 0] = 1e-8  # Small epsilon to prevent division by zero

    # Compute gradients of the depth map
    z_x = -n_x / n_z_safe
    z_y = -n_y / n_z_safe

    # Integrate gradients to obtain depth map
    surface = integrateFrankot(z_x, z_y)
    ### END YOUR CODE

    return surface


# Part 1(h)
surface = estimateShape(normals, s)
plotSurface(surface)




# # Q2: Uncalibrated photometric stereo (50 points)

# ### Q 2 (a): Uncalibrated normal estimation (10 points)
# 
# ---
# 
# YOUR ANSWER HERE...
# 
# ---

# ### Q 2 (b): Calculation and visualization (10 points)

# In[21]:


def estimatePseudonormalsUncalibrated(I):
    """
    Question 2 (b)

    Estimate pseudonormals without the help of light source directions.

    Parameters
    ----------
    I : numpy.ndarray
        The N x P matrix of loaded images (N images, P pixels)

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pseudonormals

    L : numpy.ndarray
        The 3 x N array of lighting directions (up to an unknown transformation)
    """

    ### YOUR CODE HERE
    # Perform SVD on the intensity matrix I
    U, S, Vt = np.linalg.svd(I, full_matrices=False)
    
    # Retain only the top 3 singular values and corresponding vectors
    U3 = U[:, :3]      # N x 3
    S3 = np.diag(S[:3])  # 3 x 3
    V3 = Vt[:3, :]     # 3 x P

    # Compute the square root of S3
    sqrt_S3 = np.sqrt(S3)

    # Estimate L' and B'
    L_est = U3 @ sqrt_S3    # N x 3
    B_est = sqrt_S3 @ V3    # 3 x P

    # Transpose L_est to match the expected output dimensions (3 x N)
    L = L_est.T
    B = B_est

    ### END YOUR CODE
    return B, L


# Part 2 (b)
I, L, s = loadData(data_dir)
B, LEst = estimatePseudonormalsUncalibrated(I)
albedos, normals = estimateAlbedosNormals(B)
albedoIm, normalIm = displayAlbedosNormals(albedos, normals, s)
plt.imsave('2b-a.png', albedoIm, cmap = 'gray')
plt.imsave('2b-b.png', normalIm, cmap = 'rainbow')


# ### Q 2 (c): Comparing to ground truth lighting
# 
# ---
# 
# YOUR ANSWER HERE...
# 
# ---
# 

# ### Q 2 (d): Reconstructing the shape, attempt 1 (5 points)
# 
# ---
# 
# YOUR ANSWER HERE...
# 
# ---

# In[22]:


# Part 2 (d)
### YOUR CODE HERE
# Load data
I, L_true, s = loadData(data_dir)

# Estimate pseudonormals and lighting directions (Uncalibrated)
B, L_est = estimatePseudonormalsUncalibrated(I)

# Estimate albedos and normals from the estimated pseudonormals
albedos, normals = estimateAlbedosNormals(B)

# Display and save albedo and normal maps
albedoIm, normalIm = displayAlbedosNormals(albedos, normals, s)
plt.imsave('2b-a.png', albedoIm, cmap='gray')
plt.imsave('2b-b.png', normalIm, cmap='rainbow')

# Part 2(d): Reconstruct and plot the surface
# Use the normals estimated from uncalibrated photometric stereo
surface_uncalibrated = estimateShape(normals, s)

# Plot the reconstructed surface
plotSurface(surface_uncalibrated, suffix='_uncalibrated')
### END YOUR CODE


# ### Q 2 (e): Reconstructing the shape, attempt 2 (5 points)
# 
# ---
# 
# YOUR ANSWER HERE...
# 
# ---
# 

# In[ ]:


# Part 2 (e)
# Your code here
### YOUR CODE HERE
### END YOUR CODE


# ### Q 2 (f): Why low relief? (5 points)

# ---
# 
# YOUR ANSWER HERE...
# 
# ---

# In[ ]:


def plotBasRelief(B, mu, nu, lam):

    """
    Question 2 (f)

    Make a 3D plot of of a bas-relief transformation with the given parameters.

    Parameters
    ----------
    B : numpy.ndarray
        The 3 x P matrix of pseudonormals

    mu : float
        bas-relief parameter

    nu : float
        bas-relief parameter

    lambda : float
        bas-relief parameter

    Returns
    -------
        None

    """
    P = np.asarray([[1, 0, -mu/lam],
					[0, 1, -nu/lam],
					[0, 0,   1/lam]])
    Bp = P.dot(B)
    surface = estimateShape(Bp, s)
    plotSurface(surface, suffix=f'br_{mu}_{nu}_{lam}')

# keep all outputs visible
from IPython.display import Javascript
display(Javascript('''google.colab.output.setIframeHeight(0, true, {maxHeight: 5000})'''))

# Part 2 (f)
### YOUR CODE HERE
### END YOUR CODE



# ### Q 2 (g): Flattest surface possible (5 points)
# 
# ---
# 
# YOUR ANSWER HERE...
# 
# ---

# ### Q 2 (h): More measurements
# 
# ---
# 
# YOUR ANSWER HERE...
# 
# ---
