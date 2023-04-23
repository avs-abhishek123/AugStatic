import os
import json
import random
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import pprint as pp
import numpy as np
import matplotlib.pyplot as plt

import cv2
import PIL
from PIL import Image, ImageDraw

import torch
from torchvision import transforms

import albumentations as A

from functools import wraps


# Functools module is for higher-order functions that work on other functions.
# It provides functions for working with other functions and callable objects to use or extend them without completely rewriting them.



# ### Dependencies
# 
# #### All dependencies are in requirements.txt
# 
# (Use this command to generate the requirement.txt 
# **conda list -e > requirements.txt**)
# 
# Install albumentations using the following commands
# * pip install -U albumentations
# * pip install -U git+https://github.com/albumentations-team/albumentations
# 
# Why albumentations?
# https://docs.google.com/spreadsheets/d/1rmaGngJXj3X0_ugVLWVW7h4lvayWiIJO_o2dfRNQ380/edit?usp=sharing


# ### Transform Functions 
# * Blur
# * CLAHE
# * ChannelDropout
# * ChannelShuffle
# * ColorJitter
# * Downscale
# * Emboss
# * FancyPCA
# * FromFloat (Not used while evaluating model accuracies)
# * GaussNoise
# * GaussianBlur
# * GlassBlur
# * HueSaturationValue
# * ISONoise
# * InvertImg
# * MedianBlur
# * MotionBlur
# * MultiplicativeNoise
# * Normalize (Not used while evaluating model accuracies)
# * Posterize
# * RGBShift
# * Sharpen
# * Solarize
# * Superpixels
# * ToFloat (Not used while evaluating model accuracies)
# * ToGray
# * ToSepia
# * VerticalFlip
# * HorizontalFlip
# * Flip (Not used while evaluating model accuracies)
# * Transpose
# * OpticalDistortion
# * GridDistortion
# * JpegCompression
# * Cutout
# * CoarseDropout
# * MaskDropout (Not used while evaluating model accuracies)
# * GridDropout
# * FDA (Non-Functional)
# * Equalize (Non-Functional)
# * HistogramMatching (Non-Functional)
# * ImageCompression (Non-Functional)
# * PixelDistributionAdaptation (Non-Functional)
# * PadIfNeeded (Non-Functional)
# * Lambda (Non-Functional)
# * TemplateTransform (Non-Functional)
# * RingingOvershoot (Non-Functional)
# * UnsharpMask (Non-Functional)



# Blur
def Blur(image, blur_limit=(3, 7), always_apply=False, p=1.0):
    """
    Blur the input image using a random-sized kernel.

    Args:
        image (numpy.ndarray): Input image to be blurred.
        blur_limit (tuple or int): Maximum kernel size for blurring the input image.
            Should be in range [3, inf). Default: (3, 7).
        always_apply (bool): Whether to always apply the transform. Default: False.
        p (float): Probability of applying the transform. Default: 1.0.

    Returns:
        numpy.ndarray: Blurred image.

    Targets:
        image

    Image types:
        uint8, float32
    """
    transform = A.Compose([A.augmentations.transforms.Blur(blur_limit, always_apply, p)])
    transformed = transform(image=image)
    transformed_image = transformed["image"]
    return transformed_image


# CLAHE ( Contrast Limited Adaptive Histogram Equalization )
def CLAHE(image, clip_limit=(1, 4), tile_grid_size=(8, 8), always_apply=False, p=1.0):
    """
    Apply Contrast Limited Adaptive Histogram Equalization to the input image.

    Args:
        image (numpy.ndarray): Input image to be processed.
        clip_limit (float or tuple of float): Upper threshold value for contrast limiting.
            If clip_limit is a single float value, the range will be (1, clip_limit). Default: (1, 4).
        tile_grid_size (tuple of int): Size of grid for histogram equalization. Default: (8, 8).
        always_apply (bool): Whether to always apply the transform. Default: False.
        p (float): Probability of applying the transform. Default: 1.0.

    Returns:
        numpy.ndarray: Processed image.

    Targets:
        image

    Image types:
        uint8
    """
    transform = A.Compose([A.augmentations.transforms.CLAHE(clip_limit, tile_grid_size, always_apply, p)])
    transformed = transform(image=image)
    transformed_image = transformed["image"]
    return transformed_image

# ChannelDropout
def ChannelDropout(image,channel_drop_range=(1, 1), fill_value=0, always_apply=False, p=1.0):

    """
    Randomly drop channels in the input image.

    Args:
        image (numpy.ndarray): Input image to randomly drop channels.
        channel_drop_range (tuple of int): Range from which we choose the number of channels to drop. Default: (1, 1).
        fill_value (int or float): Pixel value for the dropped channel. Default: 0.
        always_apply (bool): Whether to always apply the transform. Default: False.
        p (float): Probability of applying the transform. Default: 1.0.

    Returns:
        numpy.ndarray: Image with randomly dropped channels.

    Targets:
        image

    Image types:
        uint8, uint16, uint32, float32
    """
    transform = A.Compose([A.augmentations.ChannelDropout(channel_drop_range, fill_value, always_apply, p)])
    transformed = transform(image=image)
    transformed_image = transformed["image"]
    return transformed_image

# ChannelShuffle
def ChannelShuffle(image, p=1.0):
    """
    Randomly rearrange channels of the input RGB image.

    Args:
        p (float): Probability of applying the transform. Default: 1.0.

    Returns:
        numpy.ndarray: Image with shuffled channels.

    Targets:
        image

    Image types:
        uint8, float32
    """
    transform = A.Compose([A.augmentations.transforms.ChannelShuffle(p)])
    transformed = transform(image=image)
    transformed_image = transformed["image"]
    return transformed_image

# ColorJitter


def ColorJitter (image,brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=False, p=1.0):
    """
    Randomly changes the brightness, contrast, and saturation of an image. Compared to ColorJitter from torchvision,
    this transform gives a little bit different results because Pillow (used in torchvision) and OpenCV (used in
    Albumentations) transform an image to HSV format by different formulas. Another difference - Pillow uses uint8
    overflow, but we use value saturation.

    Args:
        image (numpy.ndarray): The image to be transformed.
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0 <= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
        always_apply (bool): Whether to always apply the transform.
        p (float): The probability of applying the transform.

    Returns:
        numpy.ndarray: The transformed image.
    """
    transform = A.Compose([A.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue, always_apply=always_apply, p=p)])
    transformed = transform(image=image)
    transformed_image = transformed["image"]
    return transformed_image

# Downscale
def Downscale (image,scale_min=0.25, scale_max=0.25, interpolation=0, always_apply=False, p=1.0):
    """
    Decreases image quality by downscaling and upscaling back.

    Args:
        image (numpy.ndarray): The image to be downscaled and upscaled back.
        scale_min (float): The lower bound on the image scale. Should be less than 1.
        scale_max (float): The upper bound on the image scale. Should be less than 1.
        interpolation (int): The cv2 interpolation method. cv2.INTER_NEAREST by default.
        always_apply (bool): Whether to always apply the transform.
        p (float): The probability of applying the transform.

    Targets:
        image

    Image types:
        numpy.ndarray of type uint8 or float32.
        
    Returns:
        numpy.ndarray: The transformed image.
    """
    transform = A.Compose([A.augmentations.transforms.Downscale(scale_min=scale_min, scale_max=scale_max, interpolation=interpolation, always_apply=always_apply, p=p)])
    transformed = transform(image=image)
    transformed_image = transformed["image"]
    return transformed_image


# Emboss
def Emboss (image,alpha=(0.2, 0.5), strength=(0.2, 0.7), always_apply=False, p=1.0):
    """Emboss the input image and overlays the result with the original image.
    Args:
        alpha ((float, float)): range to choose the visibility of the embossed image. At 0, only the original image is
            visible,at 1.0 only its embossed version is visible. Default: (0.2, 0.5).
        strength ((float, float)): strength range of the embossing. Default: (0.2, 0.7).
        p (float): probability of applying the transform. Default: 1.0.
    Targets:
        image
    """
    transform = A.Compose([A.augmentations.transforms.Emboss (alpha, strength, always_apply, p)])
    transformed = transform(image=image)
    transformed_image = transformed["image"]
    return transformed_image


# Equalize
"""
    Embosses the input image and overlays the result with the original image.

    Args:
        image (numpy.ndarray): The input image to be embossed.
        alpha (tuple of float): The range to choose the visibility of the embossed image. At 0, only the original image is
            visible; at 1.0 only its embossed version is visible. Default: (0.2, 0.5).
        strength (tuple of float): The strength range of the embossing. Default: (0.2, 0.7).
        always_apply (bool): Whether to always apply the transform.
        p (float): The probability of applying the transform.

    Targets:
        image

    Returns:
        numpy.ndarray: The transformed image.

    transform = A.Compose([A.augmentations.transforms.Emboss(alpha=alpha, strength=strength, always_apply=always_apply, p=p)])
    transformed = transform(image=image)
    transformed_image = transformed["image"]
    return transformed_image
"""


# FDA (Fourier Domain Adaptation

"""
def FDA (image, reference_images, beta_limit=0.1, read_fn='', always_apply=False, p=1.0):
    Apply Contrast Limited Adaptive Histogram Equalization to the input image.

    Args:
        image (numpy.ndarray): Input image to be transformed.
        reference_images (numpy.ndarray): Reference images for the transformation.
        beta_limit (float): Parameter to control contrast amplification. Default: 0.1.
        read_fn (str): Path to the file containing reference images. Default: ''.
        always_apply (bool): Whether to always apply the transformation. Default: False.
        p (float): Probability of applying the transform. Default: 1.0.

    Returns:
        numpy.ndarray: The transformed image.

    Targets:
        image

    Image types:
        uint8

    Refer:
        https://github.com/YanchaoYang/FDA

    transform = A.Compose([A.augmentations.FDAFDA (reference_images, beta_limit, read_fn, always_apply, p)])
    transformed = transform(image=image)
    transformed_image = transformed["image"]
    return transformed_image
"""


# FancyPCA
def FancyPCA (image,alpha=0.1, always_apply=False, p=1.0):
    """Augment RGB image using FancyPCA from Krizhevsky's paper "ImageNet Classification with Deep Convolutional 
    Neural Networks".

    Args:
        image (ndarray): RGB image to be augmented.
        alpha (float): Scale factor for perturbation of eigenvalues and eigenvectors. Scale is sampled from a 
            Gaussian distribution with mu=0 and sigma=alpha. Default is 0.1.
        always_apply (bool): Indicates whether the transform should always be applied, regardless of the given 
            probability. Default is False.
        p (float): Probability of applying the transform. Default is 1.0.

    Returns:
        ndarray: Augmented image.

    Targets:
        image

    Image types:
        3-channel uint8 images only

    Credit:
        - Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet classification with deep convolutional
            neural networks. Advances in neural information processing systems, 25, 1097-1105.
        - Deshanadesai.github.io. (2022). Fancy PCA with Scikit-Image. [online]
            Available at: https://deshanadesai.github.io/notes/Fancy-PCA-with-Scikit-Image [Accessed 23 Apr. 2023].
        - Pixelatedbrian.github.io. (2022). Fancy PCA. [online]
            Available at: https://pixelatedbrian.github.io/2018-04-29-fancy_pca/ [Accessed 23 Apr. 2023].
    """

    transform = A.Compose([A.augmentations.transforms.FancyPCA (alpha, always_apply, p)])
    transformed = transform(image=image)
    transformed_image = transformed["image"]
    return transformed_image


# FromFloat
def FromFloat (image,dtype='uint16', max_value=None, always_apply=False, p=1.0):
    """Converts input array where all values should lie in the range [0, 1.0] to values in the range [0, max_value] and
    cast the resulted value to a type specified by `dtype`. If `max_value` is None the transform will try to infer
    the maximum value for the data type from the `dtype` argument. This is the inverse transform for
    `albumentations.augmentations.transforms.ToFloat`.

    Args:
        image: Input image.
        max_value (float): Maximum possible input value. Default: None.
        dtype (str or np.dtype): Data type of the output. Default: 'uint16'.
        always_apply (bool): If True, apply the transform to all images, otherwise only apply to a random subset.
            Default: False.
        p (float): Probability of applying the transform. Default: 1.0.

    Targets:
        image

    Image types:
        float32

    Returns:
        The transformed image.
    """
    transform = A.Compose([A.augmentations.transforms.FromFloat(dtype=dtype, max_value=max_value,
                                                                always_apply=always_apply, p=p)])
    transformed = transform(image=image)
    transformed_image = transformed["image"]
    return transformed_image

# GaussNoise
def GaussNoise (image,var_limit=(10.0, 50.0), mean=0, per_channel=True, always_apply=False, p=1.0):
    """Apply Gaussian noise to the input image.

    Args:
        var_limit (float or tuple of floats): variance range for noise. If var_limit is a single float, the range
            will be (0, var_limit). Default: (10.0, 50.0).
        mean (float): mean of the noise. Default: 0
        per_channel (bool): if set to True, noise will be sampled for each channel independently.
            Otherwise, the noise will be sampled once for all channels. Default: True
        always_apply (bool): apply the transform to every input. Default: False
        p (float): probability of applying the transform. Default: 1.0.

    Targets:
        image

    Image types:
        uint8, float32
    """
    transform = A.Compose([A.augmentations.transforms.GaussNoise(var_limit=var_limit, mean=mean, per_channel=per_channel,
                                                                    always_apply=always_apply, p=p)])
    transformed = transform(image=image)
    transformed_image = transformed["image"]
    return transformed_image


# GaussianBlur
def GaussianBlur (image, blur_limit=(3, 7), sigma_limit=0, always_apply=False, p=1.0):
    """
    Blur the input image using a Gaussian filter with a random kernel size.
    Args:
        image (numpy.ndarray): Input image to be blurred.
        blur_limit (int or tuple(int, int)): Maximum Gaussian kernel size for blurring the input image.
            Must be zero or odd and in range [0, inf). If set to 0 it will be computed from sigma
            as `round(sigma * (3 if img.dtype == np.uint8 else 4) * 2 + 1) + 1`.
            If set to a single value, `blur_limit` will be in range (0, blur_limit).
            Default: (3, 7).
        sigma_limit (float or tuple(float, float)): Gaussian kernel standard deviation. Must be greater in range [0, inf).
            If set to a single value, `sigma_limit` will be in range (0, sigma_limit).
            If set to 0 sigma will be computed as `sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8`. Default: 0.
        always_apply (bool): Whether to always apply the transform. Default: False.
        p (float): Probability of applying the transform. Default: 1.0.
    Targets:
        image
    Image types:
        uint8, float32
    """
    transform = A.Compose([A.augmentations.transforms.GaussianBlur(blur_limit, sigma_limit, always_apply, p)])
    transformed = transform(image=image)
    transformed_image = transformed["image"]
    return transformed_image

# GlassBlur


def GlassBlur (image,sigma=0.7, max_delta=4, iterations=2, always_apply=False, mode='fast', p=1.0):
    """Apply glass noise to the input image.
    Args:
        sigma (float): standard deviation for Gaussian kernel.
        max_delta (int): max distance between pixels which are swapped.
        iterations (int): number of repeats.
            Should be in range [1, inf). Default: (2).
        mode (str): mode of computation: fast or exact. Default: "fast".
        p (float): probability of applying the transform. Default: 1.0.
    Targets:
        image
    Image types:
        uint8, float32
    Reference:
    |  https://arxiv.org/abs/1903.12261
    |  https://github.com/hendrycks/robustness/blob/master/ImageNet-C/create_c/make_imagenet_c.py
    """
    transform = A.Compose([A.augmentations.transforms.GlassBlur(sigma, max_delta, iterations, always_apply, mode, p)])
    transformed = transform(image=image)
    transformed_image = transformed["image"]
    return transformed_image


# HistogramMatching
"""
def HistogramMatching (image,reference_images, blend_ratio=(0.5, 1.0), read_fn="", always_apply=False, p=1.0):
    Apply histogram matching to an input image. This manipulates the pixels of the input image so that its
    histogram matches the histogram of the reference image. If the images have multiple channels, the matching is
    done independently for each channel, as long as the number of channels is equal in the input image and the
    reference image.

    Histogram matching can be used as a lightweight normalization for image processing, such as feature matching,
    especially in circumstances where the images have been taken from different sources or in different conditions
    (i.e. lighting).

    See: https://scikit-image.org/docs/dev/auto_examples/color_exposure/plot_histogram_matching.html

    Args:
        image (numpy.ndarray): Input image.
        reference_images (List[str] or List[numpy.ndarray]): List of file paths for reference images or list of
            reference images.
        blend_ratio (Tuple[float, float]): Tuple of min and max blend ratio. Matched image will be blended with
            original with random blend factor for increased diversity of generated images.
        read_fn (Callable): User-defined function to read image. Function should get image path and return numpy
            array of image pixels.
        always_apply (bool): If True, the transform is always applied to the image.
        p (float): Probability of applying the transform. Default: 1.0.

    Targets:
        image

    Image types:
        uint8, uint16, float32

    Returns:
        numpy.ndarray: Transformed image.

    transform = A.Compose([A.augmentations.domain_adaptation.HistogramMatching(reference_images, blend_ratio, read_fn, always_apply, p)])
    transformed = transform(image=image)
    transformed_image = transformed["image"]
    return transformed_image
"""


# HueSaturationValue
def HueSaturationValue (image,hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, always_apply=False, p=1.0):
    """Randomly change hue, saturation and value of the input image.

    Args:
        image (numpy.ndarray): Input image.
        hue_shift_limit (int or tuple of ints): Range for changing hue. If hue_shift_limit is a single int, the range
            will be (-hue_shift_limit, hue_shift_limit). Default: 20.
        sat_shift_limit (int or tuple of ints): Range for changing saturation. If sat_shift_limit is a single int,
            the range will be (-sat_shift_limit, sat_shift_limit). Default: 30.
        val_shift_limit (int or tuple of ints): Range for changing value. If val_shift_limit is a single int, the range
            will be (-val_shift_limit, val_shift_limit). Default: 20.
        always_apply (bool): If True, apply the transform to all input images. If False, apply the transform to
            randomly selected images. Default: False.
        p (float): Probability of applying the transform. Default: 1.0.

    Targets:
        image

    Image types:
        uint8, float32
    """
    transform = A.Compose([
        A.augmentations.transforms.HueSaturationValue(hue_shift_limit=hue_shift_limit,
                                                        sat_shift_limit=sat_shift_limit,
                                                        val_shift_limit=val_shift_limit,
                                                        always_apply=always_apply,
                                                        p=p)
    ])
    transformed = transform(image=image)
    transformed_image = transformed["image"]
    return transformed_image


# ISONoise
def ISONoise (image,color_shift=(0.01, 0.05), intensity=(0.1, 0.5), always_apply=False, p=1.0):
    """Apply camera sensor noise to the input image.

    Args:
        color_shift (float, float): Variance range for color hue change.
            Measured as a fraction of 360 degree Hue angle in HLS colorspace. Default: (0.01, 0.05).
        intensity ((float, float): Multiplicative factor that controls the strength
            of color and luminance noise. Default: (0.1, 0.5).
        always_apply (bool): Whether to always apply the transform. Default: False.
        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8

    Returns:
        numpy.ndarray: Transformed image.
    """
    transform = A.Compose([A.augmentations.transforms.ISONoise(color_shift, intensity, always_apply, p)])
    transformed = transform(image=image)
    transformed_image = transformed["image"]
    return transformed_image

# ImageCompression
"""
def ImageCompression (image,quality_lower=99, quality_upper=100, compression_type=None, always_apply=False, p=1.0):
    Decrease Jpeg, WebP compression of an image.

    Args:
        image (numpy.ndarray): Input image.
        quality_lower (float): Lower bound on the image quality.
                                Should be in [0, 100] range for jpeg and [1, 100] for webp.
        quality_upper (float): Upper bound on the image quality.
                                Should be in [0, 100] range for jpeg and [1, 100] for webp.
        compression_type (ImageCompressionType): Should be ImageCompressionType.JPEG or ImageCompressionType.WEBP.
            Default: ImageCompressionType.JPEG
        always_apply (bool): Whether to always apply the transformation.
        p (float): Probability of applying the transform. Default: 1.0.

    Returns:
        numpy.ndarray: Transformed image.

    Targets:
        image
    Image types:
        uint8, float32

    transform = A.Compose([A.augmentations.transforms.ImageCompression(quality_lower, quality_upper, compression_type, always_apply, p)])
    transformed = transform(image=image)
    transformed_image = transformed["image"]
    return transformed_image
"""

# InvertImg
def InvertImg(image,p=1.0):
    """
    Invert the input image by subtracting pixel values from 255.

    Args:
        p (float): probability of applying the transform. Default: 1.0.

    Targets:
        image

    Image types:
        uint8
    """

    transform = A.Compose([A.augmentations.transforms.InvertImg(p)])
    transformed = transform(image=image)
    transformed_image = transformed["image"]
    return transformed_image


# MedianBlur
def MedianBlur (image,blur_limit=7, always_apply=False, p=1.0):
    """
    Apply median blur to the input image using a random aperture linear size.

    Args:
        image: Input image.
        blur_limit (tuple[int]): Maximum aperture linear size for blurring the input image.
            Must be odd and in range [3, inf). Default: (3, 7).
        p (float): Probability of applying the transform. Default: 1.0.

    Targets:
        image

    Image types:
        uint8, float32
    """
    transform = A.Compose([A.augmentations.transforms.MedianBlur(blur_limit, always_apply=False, p=p)])
    transformed = transform(image=image)
    transformed_image = transformed["image"]
    return transformed_image

# MotionBlur
def MotionBlur(image,blur_limit=7,p=1.0):
    """Apply motion blur to the input image using a random-sized kernel.

    Args:
        image (numpy.ndarray): input image.
        blur_limit (tuple[int]): maximum kernel size for blurring the input image. Should be in range [3, inf).
            Default: (3, 7).
        p (float): probability of applying the transform. Default: 1.0.

    Returns:
        numpy.ndarray: transformed image.

    Targets:
        image
    Image types:
        uint8, float32
    """
    transform = A.Compose([A.augmentations.transforms.MotionBlur(blur_limit, p)])
    transformed = transform(image=image)
    transformed_image = transformed["image"]
    return transformed_image

# MultiplicativeNoise
def MultiplicativeNoise (image,multiplier=(0.9, 1.1), per_channel=False, elementwise=False, always_apply=False, p=1.0):
    """
    Multiply image to random number or array of numbers.

    Args:
        image: Input image.
        multiplier (float or tuple of floats): If a single float is provided, the image will be multiplied by this number.
            If a tuple of floats is provided, the multiplier will be in the range [multiplier[0], multiplier[1]). 
            Default: (0.9, 1.1).
        per_channel (bool): If False, the same values for all channels will be used.
            If True, sample values for each channel will be used. Default: False.
        elementwise (bool): If False, all pixels in an image will be multiplied with a random value sampled once.
            If True, image pixels will be multiplied with values that are pixelwise randomly sampled. Default: False.
        always_apply (bool): If True, apply the transform to all images. If False, apply the transform to some images.
            Default: False.
        p (float): Probability of applying the transform. Default: 1.0.

    Targets:
        image

    Image types:
        Any
    """
    transform = A.Compose([A.augmentations.transforms.MultiplicativeNoise(multiplier, per_channel, elementwise, always_apply, p)])
    transformed = transform(image=image)
    transformed_image = transformed["image"]
    return transformed_image

# Normalize
def Normalize (image,mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0):
    """
    Normalize the input image by applying the formula: `img = (img - mean * max_pixel_value) / (std * max_pixel_value)`
    Args:
        image (numpy.ndarray): input image
        mean (float or list of float): mean values
        std (float or list of float): standard deviation values
        max_pixel_value (float): maximum possible pixel value
        always_apply (bool): whether to apply the transform always or not
        p (float): probability of applying the transform

    Targets:
        image

    Image types:
        uint8, float32

    Returns:
        numpy.ndarray: normalized image
    """
    transform = A.Compose([A.augmentations.transforms.Normalize(mean, std, max_pixel_value, always_apply, p)])
    transformed = transform(image=image)
    transformed_image = transformed["image"]
    return transformed_image


# PixelDistributionAdaptation
"""
def PixelDistributionAdaptation (image,reference_images, blend_ratio=(0.25, 1.0), read_fn='', transform_type='pca', always_apply=False, p=1.0):
1.0), read_fn='', transform_type='pca', always_apply=False, p=1.0):
    Apply pixel-level domain adaptation by fitting a simple transform (such as PCA, StandardScaler, or MinMaxScaler)
    on both the original and reference image, transforms the original image with a transform trained on this
    image, and then performs an inverse transformation using a transform fitted on the reference image.

    Args:
        image (ndarray): input image to transform.
        reference_images (List[str] or List[ndarray]): list of file paths for reference images or list of reference images.
        blend_ratio (tuple): tuple of min and max blend ratio. Matched image will be blended with original with random blend
            factor for increased diversity of generated images. Default: (0.25, 1.0).
        read_fn (Callable): user-defined function to read image. Function should get image path and return numpy array of
            image pixels. Usually it's default `read_rgb_image` when images paths are used as reference, otherwise it could
            be identity function `lambda x: x` if reference images have been read in advance. Default: ''.
        transform_type (str): type of transform; "pca", "standard", "minmax" are allowed. Default: 'pca'.
        always_apply (bool): apply the transform always. Default: False.
        p (float): probability of applying the transform. Default: 1.0.

    Targets:
        image

    Image types:
        uint8, float32

    See also: https://github.com/arsenyinfo/qudida

    transform = A.Compose([A.augmentations.domain_adaptation.PixelDistributionAdaptation(reference_images, blend_ratio, read_fn, transform_type, always_apply, p)])
    transformed = transform(image=image)
    transformed_image = transformed["image"]
    return transformed_image
"""

# Posterize
def Posterize (image,num_bits=4, always_apply=False, p=1.0):
    """Reduce the number of bits for each color channel.

    Args:
    num_bits ((int, int) or int,
                or list of ints [r, g, b],
                or list of ints [[r1, r1], [g1, g2], [b1, b2]]): number of high bits.
            If num_bits is a single value, the range will be [num_bits, num_bits].
            Must be in range [0, 8]. Default: 4.
            num_bits (int, tuple(int, int), list[int], list[list[int, int]]):
            Number of high bits. If num_bits is a single value, the range will be [num_bits, num_bits].
            Must be in range [0, 8]. Default: 4.
    p (float): Probability of applying the transform. Default: 1.0.

    Targets:
    image

    Image types:
    uint8

    Returns:
    The transformed image.

    """
    transform = A.Compose([A.augmentations.transforms.Posterize (num_bits, always_apply, p)])
    transformed = transform(image=image)
    transformed_image = transformed["image"]
    return transformed_image

# RGBShift
def RGBShift (image,r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, always_apply=False, p=1.0):
    """Randomly shift values for each channel of the input RGB image.

    Args:
    r_shift_limit (int or tuple(int, int)):
    Range for changing values for the red channel. If r_shift_limit is a single
    int, the range will be (-r_shift_limit, r_shift_limit). Default: (-20, 20).
    g_shift_limit (int or tuple(int, int)):
    Range for changing values for the green channel. If g_shift_limit is a
    single int, the range will be (-g_shift_limit, g_shift_limit). Default: (-20, 20).
    b_shift_limit (int or tuple(int, int)):
    Range for changing values for the blue channel. If b_shift_limit is a single
    int, the range will be (-b_shift_limit, b_shift_limit). Default: (-20, 20).
    p (float): Probability of applying the transform. Default: 1.0.

    Targets:
    image

    Image types:
    uint8, float32

    Returns:
    The transformed image.

    """
    transform = A.Compose([A.augmentations.transforms.RGBShift (r_shift_limit, g_shift_limit, b_shift_limit, always_apply, p)])
    transformed = transform(image=image)
    transformed_image = transformed["image"]
    return transformed_image

# Sharpen
def Sharpen (image,alpha=(0.2, 0.5), lightness=(0.5, 1.0), always_apply=False, p=1.0):
    """Sharpen the input image and overlays the result with the original image.

    Args:
    alpha (tuple(float, float)):
    Range to choose the visibility of the sharpened image. At 0, only the original image is
    visible, at 1.0 only its sharpened version is visible. Default: (0.2, 0.5).
    lightness (tuple(float, float)):
    Range to choose the lightness of the sharpened image. Default: (0.5, 1.0).
    p (float): Probability of applying the transform. Default: 1.0.

    Targets:
    image

    Returns:
    The transformed image.

    """
    transform = A.Compose([A.augmentations.transforms.Sharpen (alpha, lightness, always_apply, p)])
    transformed = transform(image=image)
    transformed_image = transformed["image"]
    return transformed_image

# Solarize
def Solarize (image,threshold=128, always_apply=False, p=1.0):
    """
    Apply the Solarize transformation to invert all pixel values above the threshold. Invert all pixel values above a threshold.
    Args:
        image (numpy.ndarray):
            Input image.

        threshold ((int, int) or int, or (float, float) or float, optional):
            Range for solarizing threshold. If threshold is a single value, the range will be [threshold, threshold].
            Default is 128.

        p (float, optional):
            Probability of applying the transform. Default is 1.0.

    Returns:
        numpy.ndarray:
            Transformed image.
    """

    transform = A.Compose([
        A.augmentations.transforms.Solarize(threshold, always_apply=True, p=p)
    ])

    transformed = transform(image=image)
    transformed_image = transformed["image"]

    return transformed_image

# Superpixels
def Superpixels (image, p_replace=0.1, n_segments=100, max_size=128, interpolation=1, always_apply=False, p=1.0):
    """Transform images partially/completely to their superpixel representation.
    This implementation uses skimage's version of the SLIC algorithm.
    Args:
        image (numpy.ndarray):
            Input image.

        p_replace (float or tuple of float):
            Defines for any segment the probability that the pixels within that segment are replaced by their
            average color (otherwise, the pixels are not changed).

            Examples:
            - A probability of ``0.0`` would mean, that the pixels in no segment are replaced by their average color
            (image is not changed at all).
            - A probability of ``1.0`` would mean, that around half of all segments are replaced by their average color.
            - A probability of ``1.0`` would mean, that all segments are replaced by their average color (resulting in a voronoi image).

            Behaviour based on chosen data types for this parameter:
            - If a ``float``, then that ``flat`` will always be used.
            - If ``tuple`` ``(a, b)``, then a random probability will be sampled from the interval ``[a, b]`` per image.

        n_segments (int or tuple of int):
            Rough target number of how many superpixels to generate (the algorithm may deviate from this number).
            Lower value will lead to coarser superpixels. Higher values are computationally more intensive and will hence lead to a slowdown.

            - If a single ``int``, then that value will always be used as the number of segments.
            - If a ``tuple`` ``(a, b)``, then a value from the discrete interval ``[a..b]`` will be sampled per image.

        max_size (int or None):
            Maximum image size at which the augmentation is performed. If the width or height of an image exceeds this
            value, it will be downscaled before the augmentation so that the longest side matches `max_size`.
            This is done to speed up the process. The final output image has the same size as the input image.
            Note that in case `p_replace` is below ``1.0``, the down-/upscaling will affect the not-replaced pixels too.
            Use ``None`` to apply no down-/upscaling.

        interpolation (cv2 flag):
            Flag that is used to specify the interpolation algorithm. Should be one of:
            - cv2.INTER_NEAREST
            - cv2.INTER_LINEAR
            - cv2.INTER_CUBIC
            - cv2.INTER_AREA
            - cv2.INTER_LANCZOS4.

            Default is cv2.INTER_LINEAR.

        always_apply (bool):
            Apply the transformation to all images, regardless of the probability defined by `p`.

        p (float):
            Probability of applying the transform. Default is 1.0.

    Targets:
        image

    Returns:
        numpy.ndarray:
            Transformed image.
    """
    transform = A.Compose([
        A.augmentations.transforms.Superpixels(p_replace=p_replace,
                                            n_segments=n_segments,
                                            max_size=max_size,
                                            interpolation=interpolation,
                                            always_apply=always_apply,
                                            p=p)
    ])
    transformed = transform(image=image)
    transformed_image = transformed["image"]
    return transformed_image

# ToFloat
def ToFloat (image,max_value=None, always_apply=False, p=1.0):
    """Divide pixel values by `max_value` to get a float32 output array where all values lie in the range [0, 1.0].
    If `max_value` is None the transform will try to infer the maximum value by inspecting the data type of the input
    image.
    See Also:
        :class:`~albumentations.augmentations.transforms.FromFloat`
    Args:
        max_value (float): maximum possible input value. Default: None.
        p (float): probability of applying the transform. Default: 1.0.
    Targets:
        image
    Image types:
        any type
    """
    transform = A.Compose([A.augmentations.transforms.ToFloat(max_value, always_apply, p)])
    transformed = transform(image=image)
    transformed_image = transformed["image"]
    return transformed_image

# ToGray
def ToGray(image,p=1.0):
    """Converts an RGB image to grayscale. If the mean pixel value of the grayscale image is greater than 127,
    invert the resulting grayscale image.

    Args:
        image (numpy.ndarray): Input image.
        p (float): Probability of applying the transform. Default: 1.0.

    Targets:
        image

    Image types:
        uint8, float32
    """
    transform = A.Compose([A.augmentations.transforms.ToGray(p)])
    transformed = transform(image=image)
    transformed_image = transformed["image"]
    return transformed_image

# ToSepia
def ToSepia (image,always_apply=False, p=1.0):
    """
    Applies a sepia filter to the input RGB image.

    Args:
        image (numpy.ndarray): Input RGB image.
        always_apply (bool): Indicates if the transform should always be applied. Default: False.
        p (float): Probability of applying the transform. Default: 1.0.

    Targets:
        image

    Image types:
        uint8, float32
    """

    transform = A.Compose([A.augmentations.transforms.ToSepia(always_apply=always_apply, p=p)])
    transformed = transform(image=image)
    transformed_image = transformed["image"]
    return transformed_image

# VerticalFlip
def VerticalFlip (image, p=1.0):
    """Flip the input vertically around the x-axis.

    Args:
        image (ndarray): input image.
        p (float): probability of applying the transform. Default: 1.0.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    Returns:
        ndarray: transformed image
    """
    transform = A.Compose([A.augmentations.transforms.VerticalFlip(p)])
    transformed = transform(image=image)
    transformed_image = transformed["image"]
    return transformed_image

# HorizontalFlip
def HorizontalFlip (image, p=1.0):
    """Flip the input horizontally around the y-axis.

    Args:
        image (numpy.ndarray): Input image to be flipped.
        p (float): Probability of applying the transform. Default: 1.0.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """
    transform = A.Compose([A.augmentations.transforms.HorizontalFlip(p)])
    transformed = transform(image=image)
    transformed_image = transformed["image"]
    return transformed_image

# #### Flip (Random_Flip)
def Flip (image, p=1.0):
    """
    Flip the input either horizontally, vertically or both horizontally and vertically.

    Args:
        image (numpy.ndarray): Input image.
        p (float): Probability of applying the transform. Default: 1.0.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    transform = A.Compose([A.augmentations.transforms.Flip(p)])
    transformed = transform(image=image)
    transformed_image = transformed["image"]
    return transformed_image

#d = random.randint(-1, 1)
#https://github.com/albumentations-team/albumentations/blob/master/albumentations/augmentations/transforms.py#L337
'''
def random_flip(img, code):
    return cv2.flip(img, code)
'''
# https://github.com/albumentations-team/albumentations/blob/6de7dd01410a666c23c70cf69c548f171c94a1a7/albumentations/augmentations/functional.py#L119

'''
def bbox_vflip(bbox, rows, cols):  # skipcq: PYL-W0613
    """Flip a bounding box vertically around the x-axis.
    Args:
        bbox (tuple): A bounding box `(x_min, y_min, x_max, y_max)`.
        rows (int): Image rows.
        cols (int): Image cols.
    Returns:
        tuple: A bounding box `(x_min, y_min, x_max, y_max)`.
    """
    x_min, y_min, x_max, y_max = bbox[:4]
    return x_min, 1 - y_max, x_max, 1 - y_min


def bbox_hflip(bbox, rows, cols):  # skipcq: PYL-W0613
    """Flip a bounding box horizontally around the y-axis.
    Args:
        bbox (tuple): A bounding box `(x_min, y_min, x_max, y_max)`.
        rows (int): Image rows.
        cols (int): Image cols.
    Returns:
        tuple: A bounding box `(x_min, y_min, x_max, y_max)`.
    """
    x_min, y_min, x_max, y_max = bbox[:4]
    return 1 - x_max, y_min, 1 - x_min, y_max

def bbox_flip(bbox, d, rows, cols):
    """Flip a bounding box either vertically, horizontally or both depending on the value of `d`.
    Args:
        bbox (tuple): A bounding box `(x_min, y_min, x_max, y_max)`.
        d (int):
        rows (int): Image rows.
        cols (int): Image cols.
    Returns:
        tuple: A bounding box `(x_min, y_min, x_max, y_max)`.
    Raises:
        ValueError: if value of `d` is not -1, 0 or 1.
    """
    if d == 0:
        bbox = bbox_vflip(bbox, rows, cols)
    elif d == 1:
        bbox = bbox_hflip(bbox, rows, cols)
    elif d == -1:
        bbox = bbox_hflip(bbox, rows, cols)
        bbox = bbox_vflip(bbox, rows, cols)
    else:
        raise ValueError("Invalid d value {}. Valid values are -1, 0 and 1".format(d))
    return bbox
'''
#https://github.com/albumentations-team/albumentations/blob/6de7dd01410a666c23c70cf69c548f171c94a1a7/albumentations/augmentations/functional.py#L1320


# Transpose
def Transpose (image, p=1.0):
    """Transpose the input by swapping rows and columns.

    Args:
        image: The input image.
        p (float): Probability of applying the transform. Default: 1.0.
    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """
    transform = A.Compose([A.augmentations.transforms.Transpose(p=p)])
    transformed = transform(image=image)
    transformed_image = transformed["image"]
    return transformed_image


# OpticalDistortion
def OpticalDistortion (image,distort_limit=0.05, shift_limit=0.05, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=1.0):
    """
    Args:
        distort_limit (float, (float, float)): If distort_limit is a single float, the range
            will be (-distort_limit, distort_limit). Default: (-0.05, 0.05).
        shift_limit (float, (float, float))): If shift_limit is a single float, the range
            will be (-shift_limit, shift_limit). Default: (-0.05, 0.05).
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        border_mode (OpenCV flag): flag that is used to specify the pixel extrapolation method. Should be one of:
            cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_WRAP, cv2.BORDER_REFLECT_101.
            Default: cv2.BORDER_REFLECT_101
        value (int, float, list of ints, list of float): padding value if border_mode is cv2.BORDER_CONSTANT.
        mask_value (int, float,
                    list of ints,
                    list of float): padding value if border_mode is cv2.BORDER_CONSTANT applied for masks.
    Targets:
        image, mask
    Image types:
        uint8, float32
    """
    transform = A.Compose([A.augmentations.transforms.OpticalDistortion (distort_limit, shift_limit, interpolation, border_mode, value, mask_value, always_apply, p)])
    transformed = transform(image=image)
    transformed_image = transformed["image"]
    return transformed_image


# GridDistortion
def GridDistortion (image,num_steps=5, distort_limit=0.3, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=1) :
    """
    Apply grid distortion to the input image.

    Args:
        image (numpy.ndarray): input image to be distorted.
        num_steps (int): number of grid cells on each side. Default: 5.
        distort_limit (float or tuple): maximum amount of distortion. If distort_limit is a single float, the range
            will be (-distort_limit, distort_limit). Default: 0.3.
        interpolation (int): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        border_mode (int): flag that is used to specify the pixel extrapolation method. Should be one of:
            cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_WRAP, cv2.BORDER_REFLECT_101.
            Default: cv2.BORDER_REFLECT_101.
        value (int, float or list): padding value if border_mode is cv2.BORDER_CONSTANT.
        mask_value (int, float or list): padding value if border_mode is cv2.BORDER_CONSTANT applied for masks.
        always_apply (bool): if True, always apply the transform. Default: False.
        p (float): probability of applying the transform. Default: 1.0.

    Targets:
        image, mask
    Image types:
        uint8, float32
    """
    transform = A.Compose([
        A.augmentations.transforms.GridDistortion(
            num_steps=num_steps,
            distort_limit=distort_limit,
            interpolation=interpolation,
            border_mode=border_mode,
            value=value,
            mask_value=mask_value,
            always_apply=always_apply,
            p=p
        )
    ])
    transformed = transform(image=image)
    transformed_image = transformed["image"]
    return transformed_image


# PadIfNeeded
"""
def PadIfNeeded (image,min_height=1024, min_width=1024, pad_height_divisor=None, pad_width_divisor=None, position=PositionType.CENTER, border_mode=4, value=None, mask_value=None, always_apply=False, p=1.0):
    Pad side of the image / max if side is less than desired number.

    Args:
        image (numpy.ndarray): Image to be transformed.
        min_height (int): Minimal result image height.
        min_width (int): Minimal result image width.
        pad_height_divisor (int): If not None, ensures image height is dividable by value of this argument.
        pad_width_divisor (int): If not None, ensures image width is dividable by value of this argument.
        position (Union[str, PositionType]): Position of the image. Should be PositionType.CENTER or
            PositionType.TOP_LEFT or PositionType.TOP_RIGHT or PositionType.BOTTOM_LEFT or PositionType.BOTTOM_RIGHT.
            Default: PositionType.CENTER.
        border_mode (int): OpenCV border mode.
        value (int, float, list of int, list of float): Padding value if border_mode is cv2.BORDER_CONSTANT.
        mask_value (int, float, list of int, list of float): Padding value for mask if border_mode is cv2.BORDER_CONSTANT.
        always_apply (bool): Apply the transform in any case. Default: False.
        p (float): Probability of applying the transform. Default: 1.0.

    Targets:
        image, mask, bbox, keypoints

    Image types:
        uint8, float32
    transform = A.Compose([
        A.augmentations.transforms.PadIfNeeded(
            min_height=min_height, min_width=min_width, pad_height_divisor=pad_height_divisor,
            pad_width_divisor=pad_width_divisor, position=position, border_mode=border_mode,
            value=value, mask_value=mask_value, always_apply=always_apply, p=p
        )
    ])
    transformed = transform(image=image)
    transformed_image = transformed["image"]
    return transformed_image
    """

# JpegCompression
def JpegCompression (image,quality_lower=99, quality_upper=100, always_apply=False, p=1.0):
    """
    Decrease Jpeg compression of an image.

    Args:
        quality_lower (float): lower bound on the jpeg quality. Should be in [0, 100] range.
        quality_upper (float): upper bound on the jpeg quality. Should be in [0, 100] range.
        always_apply (bool): apply the transform always. Default: False.
        p (float): probability of applying the transform. Default: 1.0.

    Targets:
        image

    Image types:
        uint8, float32
    """
    transform = A.Compose([A.augmentations.transforms.JpegCompression(quality_lower, quality_upper, always_apply, p)])
    transformed = transform(image=image)
    transformed_image = transformed["image"]
    return transformed_image

# Cutout
def Cutout (image,num_holes=8, max_h_size=8, max_w_size=8, fill_value=0, always_apply=False, p=1.0):
    
    """
    CoarseDropout of the square regions in the image.

    Args:
        image (numpy.ndarray): Input image.
        num_holes (int): Number of regions to zero out.
        max_h_size (int): Maximum height of the hole.
        max_w_size (int): Maximum width of the hole.
        fill_value (int, float, list of int, list of float): Value for dropped pixels.
        always_apply (bool): Whether to apply the transformation always.
        p (float): Probability of applying the transform.

    Targets:
        image

    Image types:
        uint8, float32

    Reference:
        https://arxiv.org/abs/1708.04552
        https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
        https://github.com/aleju/imgaug/blob/master/imgaug/augmenters/arithmetic.py
    """
    transform = A.Compose([A.augmentations.transforms.Cutout(num_holes, max_h_size, max_w_size, fill_value, always_apply, p)])
    transformed = transform(image=image)
    transformed_image = transformed["image"]
    return transformed_image

# CoarseDropout
def CoarseDropout (image,max_holes=8, max_height=8, max_width=8, min_holes=None, min_height=None, min_width=None, fill_value=0, mask_fill_value=None, always_apply=False, p=1.0):
    """CoarseDropout of the rectangular regions in the image.
    Args:
        max_holes (int): Maximum number of regions to zero out.
        max_height (int, float): Maximum height of the hole.
        If float, it is calculated as a fraction of the image height.
        max_width (int, float): Maximum width of the hole.
        If float, it is calculated as a fraction of the image width.
        min_holes (int): Minimum number of regions to zero out. If `None`,
            `min_holes` is be set to `max_holes`. Default: `None`.
        min_height (int, float): Minimum height of the hole. Default: None. If `None`,
            `min_height` is set to `max_height`. Default: `None`.
            If float, it is calculated as a fraction of the image height.
        min_width (int, float): Minimum width of the hole. If `None`, `min_height` is
            set to `max_width`. Default: `None`.
            If float, it is calculated as a fraction of the image width.
        fill_value (int, float, list of int, list of float): value for dropped pixels.
        mask_fill_value (int, float, list of int, list of float): fill value for dropped pixels
            in mask. If `None` - mask is not affected. Default: `None`.
    Targets:
        image, mask
    Image types:
        uint8, float32
    Reference:
    |  https://arxiv.org/abs/1708.04552
    |  https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
    |  https://github.com/aleju/imgaug/blob/master/imgaug/augmenters/arithmetic.py
    """
    transform = A.Compose([A.augmentations.transforms.CoarseDropout (max_holes, max_height, max_width, min_holes, min_height, min_width, fill_value, mask_fill_value, always_apply, p)])
    transformed = transform(image=image)
    transformed_image = transformed["image"]
    return transformed_image


# #### Lambda
"""
    A flexible transformation class for using user-defined transformation functions per targets.
    Function signature must include **kwargs to accept optional arguments like interpolation method, image size, etc:

    Args:
        image (callable): Image transformation function.
        mask (callable): Mask transformation function.
        keypoint (callable): Keypoint transformation function.
        bbox (callable): BBox transformation function.
        always_apply (bool): Indicates whether this transformation should be always applied.
        p (float): probability of applying the transform. Default: 1.0.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        Any

    transform = A.Compose([A.augmentations.transforms.Lambda(image=image, mask=mask, keypoint=keypoint, bbox=bbox, name=name, always_apply=always_apply, p=p)])
    transformed = transform(image=image)
    transformed_image = transformed["image"]
    return transformed_image
"""


# #### MaskDropout
def MaskDropout (image, max_objects=1, image_fill_value=0, mask_fill_value=0, always_apply=False, p=1.0):
    """
    Image & mask augmentation that zero out mask and image regions corresponding
    to randomly chosen object instance from mask.
    Mask must be single-channel image, zero values treated as background.
    Image can be any number of channels.
    Inspired by https://www.kaggle.com/c/severstal-steel-defect-detection/discussion/114254
    """
    transform = A.Compose([A.augmentations.transforms.MaskDropout (max_objects, image_fill_value, mask_fill_value, always_apply, p)])
    transformed = transform(image=image)
    transformed_image = transformed["image"]
    return transformed_image


# #### GridDropout
def GridDropout (image,ratio=0.5, unit_size_min=None, unit_size_max=None, holes_number_x=None, holes_number_y=None, shift_x=0, shift_y=0, random_offset=False, fill_value=0, mask_fill_value=None, always_apply=False, p=1.0):
    """GridDropout, drops out rectangular regions of an image and the corresponding mask in a grid fashion.
    Args:
        ratio (float): the ratio of the mask holes to the unit_size (same for horizontal and vertical directions).
            Must be between 0 and 1. Default: 0.5.
        unit_size_min (int): minimum size of the grid unit. Must be between 2 and the image shorter edge.
            If 'None', holes_number_x and holes_number_y are used to setup the grid. Default: `None`.
        unit_size_max (int): maximum size of the grid unit. Must be between 2 and the image shorter edge.
            If 'None', holes_number_x and holes_number_y are used to setup the grid. Default: `None`.
        holes_number_x (int): the number of grid units in x direction. Must be between 1 and image width//2.
            If 'None', grid unit width is set as image_width//10. Default: `None`.
        holes_number_y (int): the number of grid units in y direction. Must be between 1 and image height//2.
            If `None`, grid unit height is set equal to the grid unit width or image height, whatever is smaller.
        shift_x (int): offsets of the grid start in x direction from (0,0) coordinate.
            Clipped between 0 and grid unit_width - hole_width. Default: 0.
        shift_y (int): offsets of the grid start in y direction from (0,0) coordinate.
            Clipped between 0 and grid unit height - hole_height. Default: 0.
        random_offset (boolean): weather to offset the grid randomly between 0 and grid unit size - hole size
            If 'True', entered shift_x, shift_y are ignored and set randomly. Default: `False`.
        fill_value (int): value for the dropped pixels. Default = 0
        mask_fill_value (int): value for the dropped pixels in mask.
            If `None`, transformation is not applied to the mask. Default: `None`.
    Targets:
        image, mask
    Image types:
        uint8, float32
    References:
        https://arxiv.org/abs/2001.04086
    """
    transform = A.Compose([A.augmentations.transforms.GridDropout (ratio, unit_size_min, unit_size_max, holes_number_x, holes_number_y, shift_x, shift_y, random_offset, fill_value, mask_fill_value, always_apply, p) ])
    transformed = transform(image=image)
    transformed_image = transformed["image"]
    return transformed_image


# #### TemplateTransform
"""
def TemplateTransform (image,templates=None, img_weight=0.5, template_weight=0.5, template_transform=None, name=None, always_apply=False, p=1.0):
    Apply blending of input image with specified templates
    Args:
        templates (numpy array or list of numpy arrays): Images as template for transform.
        img_weight ((float, float) or float): If single float will be used as weight for input image.
            If tuple of float img_weight will be in range `[img_weight[0], img_weight[1])`. Default: 0.5.
        template_weight ((float, float) or float): If single float will be used as weight for template.
            If tuple of float template_weight will be in range `[template_weight[0], template_weight[1])`.
            Default: 0.5.
        template_transform: transformation object which could be applied to template,
            must produce template the same size as input image.
        name (string): (Optional) Name of transform, used only for deserialization.
        p (float): probability of applying the transform. Default: 1.0.
    Targets:
        image
    Image types:
        uint8, float32

    transform = A.Compose([A.augmentations.transforms.TemplateTransform (templates, img_weight, template_weight, template_transform, name, always_apply, p)])
    transformed = transform(image=image)
    transformed_image = transformed["image"]
    return transformed_image
"""


# #### RingingOvershoot
"""
def RingingOvershoot (image,blur_limit=(7, 15), cutoff=(0.7853981633974483, 1.5707963267948966), always_apply=False, p=1.0):
    Create ringing or overshoot artefacts by conlvolving image with 2D sinc filter.
    Args:
        blur_limit (int, (int, int)): maximum kernel size for sinc filter.
            Should be in range [3, inf). Default: (7, 15).
        cutoff (float, (float, float)): range to choose the cutoff frequency in radians.
            Should be in range (0, np.pi)
            Default: (np.pi / 4, np.pi / 2).
        p (float): probability of applying the transform. Default: 1.0
    Reference:
        dsp.stackexchange.com/questions/58301/2-d-circularly-symmetric-low-pass-filter
        https://arxiv.org/abs/2107.10833
    Targets:
        image
    transform = A.Compose([A.augmentations.transforms.RingingOvershoot (blur_limit, cutoff, always_apply, p)])
    transformed = transform(image=image)
    transformed_image = transformed["image"]
    return transformed_image
"""


# #### UnsharpMask
"""
def UnsharpMask (image,blur_limit=(3, 7), sigma_limit=0.0, alpha=(0.2, 1.0), threshold=10, always_apply=False, p=1.0):
    Sharpen the input image using Unsharp Masking processing and overlays the result with the original image.
    Args:
        blur_limit (int, (int, int)): maximum Gaussian kernel size for blurring the input image.
            Must be zero or odd and in range [0, inf). If set to 0 it will be computed from sigma
            as `round(sigma * (3 if img.dtype == np.uint8 else 4) * 2 + 1) + 1`.
            If set single value `blur_limit` will be in range (0, blur_limit).
            Default: (3, 7).
        sigma_limit (float, (float, float)): Gaussian kernel standard deviation. Must be in range [0, inf).
            If set single value `sigma_limit` will be in range (0, sigma_limit).
            If set to 0 sigma will be computed as `sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8`. Default: 0.
        alpha (float, (float, float)): range to choose the visibility of the sharpened image.
            At 0, only the original image is visible, at 1.0 only its sharpened version is visible.
            Default: (0.2, 0.5).
        threshold (int): Value to limit sharpening only for areas with high pixel difference between original image
            and it's smoothed version. Higher threshold means less sharpening on flat areas.
            Must be in range [0, 255]. Default: 10.
        p (float): probability of applying the transform. Default: 1.0
    Reference:
        arxiv.org/pdf/2107.10833.pdf
    Targets:
        image
    transform = A.Compose([A.augmentations.transforms.UnsharpMask (blur_limit, sigma_limit, alpha, threshold, always_apply, p)])
    transformed = transform(image=image)
    transformed_image = transformed["image"]
    return transformed_image"""

# ## Read & Display Images
# #### Read image Custom funtion
def readImage(image_path):
    imageInBGR= cv2.imread(image_path)
    imageBGR2RGB=cv2.cvtColor(imageInBGR, cv2.COLOR_BGR2RGB)
    return imageBGR2RGB

# #### Display Image
def visualize(image):
    plt.imshow(image)
    plt.axis("OFF")
    plt.show()

# ### Custom Data Generator
class StaticTransformDataGenerator(Sequence):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """
    def __init__(self, image,num_sample):
        """Initialization
        :param list_IDs: list of all 'label' ids to use in the generator
        :param labels: list of image labels (file names)
        :param image_path: path to images location
        :param mask_path: path to masks location
        :param to_fit: True to return X and y, False to return X only
        :param num_sample: batch size at each iteration
        :param dim: tuple indicating image dimension
        :param n_channels: number of image channels
        :param n_classes: number of output masks
        :param shuffle: True to shuffle label indexes after every epoch
        """
        self.image=image
        self.num_sample=1
        #self.list_IDs = list_IDs
        #self.labels = labels
        #self.image_path = image_path
        #self.mask_path = mask_path
        #self.to_fit = to_fit
        #self.num_sample = num_sample
        #self.dim = dim
        #self.n_channels = n_channels
        #self.shuffle = shuffle
        #self.on_epoch_end()
    # First, we define the constructor to initialize the configuration of the generator.
    # we assume the path to the data is in a dataframe column.
    # Hence, we define the x_col and y_col parameters.
    # This could also be a directory name from where you can load the data.
    #Another utility method we have is __len__.
    #It essentially returns the number of steps in an epoch, using the samples and the batch size.
    def __getitem__(self, index):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X = self._generate_X(list_IDs_temp)

        if self.to_fit:
            y = self._generate_y(list_IDs_temp)
            return X, y
        else:
            return X

    def __len__(self):
            # Denotes the number of batches per epoch
        return int(np.floor(len(self.list_IDs) / self.batch_size))
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """

    def Blur (self,blur_limit=7, always_apply=False, p=1.0):
        for i in range(self.num_sample):
            img=Blur(self.image,blur_limit, always_apply, p)
            transform_type='Blur'
            return img

    def CLAHE (self,clip_limit=4.0, tile_grid_size=(8, 8), always_apply=False, p=1.0):
        for i in range(1):
            img=CLAHE(self.image,clip_limit, tile_grid_size, always_apply, p)
            transform_type='CLAHE'
            return img

    def ChannelDropout(self,channel_drop_range=(1, 1), fill_value=0, always_apply=False, p=1.0):
        for i in range(self.num_sample):
            img=ChannelDropout(self.image)
            transform_type='ChannelDropout'
            return img


    def ChannelShuffle (self,p=1.0):
        for i in range(self.num_sample):
            img=ChannelShuffle(self.image,p)
            transform_type='ChannelShuffle'
            return img

    def ColorJitter (self):
        for i in range(self.num_sample):
            img=ColorJitter(self.image)
            transform_type='ColorJitter'
            return img

    def Downscale (self):
        for i in range(self.num_sample):
            img=Downscale(self.image)
            transform_type='Downscale'
            return img


    def Emboss (self):
        for i in range(self.num_sample):
            img=Emboss(self.image)
            transform_type='Emboss'
            return img

    """def Equalize (self):
        for i in range(self.num_sample):
            img=Equalize(self.image)
            transform_type='Equalize'
            return img  """

    """def FDA (self):
        for i in range(self.num_sample):
            img=FDA(self.image)
            transform_type='FDA'
            return img  """

    def FancyPCA (self):
        for i in range(self.num_sample):
            img=FancyPCA(self.image)
            transform_type='FancyPCA'
            return img

    def FromFloat (self):
        for i in range(self.num_sample):
            img=FromFloat(self.image)
            transform_type='FromFloat'
            return img

    def GaussNoise (self):
        for i in range(self.num_sample):
            img=GaussNoise(self.image)
            transform_type='GaussNoise'
            return img

    def GaussianBlur (self):
        for i in range(self.num_sample):
            img=GaussianBlur(self.image)
            transform_type='GaussianBlur'
            return img

    def GlassBlur (self):
        for i in range(self.num_sample):
            img=GlassBlur(self.image)
            transform_type='GlassBlur'
            return img

    """def HistogramMatching (self):
        for i in range(self.num_sample):
            img=HistogramMatching(self.image)
            transform_type='HistogramMatching'
            return img   """

    def HueSaturationValue (self):
        for i in range(self.num_sample):
            img=HueSaturationValue(self.image)
            transform_type='HueSaturationValue'
            return img

    def ISONoise (self):
        for i in range(self.num_sample):
            img=ISONoise(self.image)
            transform_type='ISONoise'
            return img

    """def ImageCompression (self):
        for i in range(self.num_sample):
            img=ImageCompression(self.image)
            transform_type='ImageCompression'
            return img """

    def InvertImg (self):
        for i in range(self.num_sample):
            img=InvertImg(self.image)
            transform_type='InvertImg'
            return img

    def MedianBlur (self):
        for i in range(self.num_sample):
            img=MedianBlur(self.image)
            transform_type='MedianBlur'
            return img

    def MotionBlur (self):
        for i in range(self.num_sample):
            img=MotionBlur(self.image)
            transform_type='MotionBlur'
            return img

    def MultiplicativeNoise (self):
        for i in range(self.num_sample):
            img=MultiplicativeNoise(self.image)
            transform_type='MultiplicativeNoise'
            return img

    def Normalize (self):
        for i in range(self.num_sample):
            img=Normalize(self.image)
            transform_type='Normalize'
            return img

    """def PixelDistributionAdaptation (self):
        for i in range(self.num_sample):
            img=PixelDistributionAdaptation(self.image)
            transform_type='PixelDistributionAdaptation'
            return img"""

    def Posterize (self):
        for i in range(self.num_sample):
            img=Posterize(self.image)
            transform_type='Posterize'
            return img

    def RGBShift (self):
        for i in range(self.num_sample):
            img=RGBShift(self.image)
            transform_type='RGBShift'
            return img

    def Sharpen (self):
        for i in range(self.num_sample):
            img=Sharpen(self.image)
            transform_type='Sharpen'
            return img

    def Solarize (self):
        for i in range(self.num_sample):
            img=Solarize(self.image)
            transform_type='Solarize'
            return img

    def Superpixels (self):
        for i in range(self.num_sample):
            img=Superpixels(self.image)
            transform_type='Superpixels'
            return img

    def ToFloat (self):
        for i in range(self.num_sample):
            img=ToFloat(self.image)
            transform_type='ToFloat'
            return img

    def ToGray (self):
        for i in range(self.num_sample):
            img=ToGray(self.image)
            transform_type='ToGray'
            return img

    def ToSepia (self):
        for i in range(self.num_sample):
            img=ToSepia(self.image)
            transform_type='ToSepia'
            return img

    def VerticalFlip (self):
        for i in range(self.num_sample):
            img=VerticalFlip(self.image)
            transform_type='VerticalFlip'
            return img

    def HorizontalFlip (self):
        for i in range(self.num_sample):
            img=HorizontalFlip(self.image)
            transform_type='HorizontalFlip'
            return img

    def Flip (self):
        for i in range(self.num_sample):
            img=Flip(self.image)
            transform_type='Flip'
            return img

    def Transpose (self):
        for i in range(self.num_sample):
            img=Transpose(self.image)
            transform_type='Transpose'
            return img

    def OpticalDistortion (self):
        for i in range(self.num_sample):
            img=OpticalDistortion(self.image)
            transform_type='OpticalDistortion'
            return img

    def GridDistortion (self):
        for i in range(self.num_sample):
            img=GridDistortion(self.image)
            transform_type='GridDistortion'
            return img

    """def PadIfNeeded (self):
        for i in range(self.num_sample):
            img=PadIfNeeded(self.image)
            transform_type='PadIfNeeded'
            return img
    """

    def JpegCompression (self):
        for i in range(self.num_sample):
            img=JpegCompression(self.image)
            transform_type='JpegCompression'
            return img

    def Cutout (self):
        for i in range(self.num_sample):
            img=Cutout(self.image)
            transform_type='Cutout'
            return img

    def CoarseDropout (self):
        for i in range(self.num_sample):
            img=CoarseDropout(self.image)
            transform_type='CoarseDropout'
            return img

    """def Lambda (self):
        for i in range(self.num_sample):
            img=Lambda(self.image)
            transform_type='Lambda'
            return img
    """

    def MaskDropout (self):
        for i in range(self.num_sample):
            img=MaskDropout(self.image)
            transform_type='MaskDropout'
            return img

    def GridDropout (self):
        for i in range(self.num_sample):
            img=GridDropout(self.image)
            transform_type='GridDropout'
            return img

    """def TemplateTransform (self):
        for i in range(self.num_sample):
            img=TemplateTransform(self.image)
            transform_type='TemplateTransform'
            return img
    """

    """def RingingOvershoot (self):
        for i in range(self.num_sample):
            img=RingingOvershoot(self.image)
            transform_type='RingingOvershoot'
            return img
    """

    """def UnsharpMask (self,image=self.image,=1):
        for i in range(self.num_sample):
            img=UnsharpMask(self.image)
            transform_type='UnsharpMask'
            return img
    """

    def _load_grayscale_image(self, image_path):
        """Load grayscale image
        :param image_path: path to image to load
        :return: loaded image
        """
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img / 255
        return img