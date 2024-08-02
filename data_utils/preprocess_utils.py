from scipy.ndimage import gaussian_filter
import numpy as np


def per_band_gaussian_filter(img: np.ndarray, sigma: float = 1):
    """
    For each band in the image, apply a gaussian filter with the given sigma.

    Parameters
    ----------
    img : np.ndarray
        The image to be filtered.
    sigma : float
        The sigma of the gaussian filter.

    Returns
    -------
    np.ndarray
        The filtered image.
    """
    for i in range(img.shape[0]):
        img[i] = gaussian_filter(img[i], sigma)
    return img


def quantile_clip(img_stack: np.ndarray,
                  clip_quantile: float,
                  ) -> np.ndarray:
    """
    This function clips the outliers of the image stack by the given quantile.

    Parameters
    ----------
    img_stack : np.ndarray
        The image stack to be clipped.
    clip_quantile : float
        The quantile to clip the outliers by.

    Returns
    -------
    np.ndarray
        The clipped image stack.
    """
    axis = (-2, -1)
    data_lower_bound = np.quantile(
        img_stack,
        clip_quantile,
        axis=axis,
        keepdims=True
        )
    data_upper_bound = np.quantile(
        img_stack,
        1-clip_quantile,
        axis=axis,
        keepdims=True
        )
    img_stack = np.clip(img_stack, data_lower_bound, data_upper_bound)

    return img_stack


def minmax_scale(img: np.ndarray):
    """
    This function minmax scales the image stack.

    Parameters
    ----------
    img : np.ndarray
        The image stack to be minmax scaled.

    Returns
    -------
    np.ndarray
        The minmax scaled image stack.
    """
    axis = (-2, -1)
    img = img.astype(np.float32)
    min_val = img.min(axis=axis, keepdims=True)
    max_val = img.max(axis=axis, keepdims=True)
    normalized_img = (img - min_val) / (max_val - min_val)
    return normalized_img


def brighten(img, alpha=0.13, beta=0):
    """
    Function to brighten the image.

    Parameters
    ----------
    img : np.ndarray
        The image to be brightened.
    alpha : float
        The alpha parameter of the brightening.
    beta : float
        The beta parameter of the brightening.

    Returns
    -------
    np.ndarray
        The brightened image.
    """
    return np.clip(alpha * img + beta, 0.0, 1.0)


def gammacorr(band, gamma: float = 2.0):
    """
    This function applies a gamma correction to the image.

    Parameters
    ----------
    band : np.ndarray
        The image to be gamma corrected.
    gamma : float
        The gamma parameter of the gamma correction.

    Returns
    -------
    np.ndarray
        The gamma corrected image.
    """
    return np.power(band, 1/gamma)