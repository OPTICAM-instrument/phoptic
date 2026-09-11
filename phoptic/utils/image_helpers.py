from numpy.typing import NDArray




def rebin_image(
    image: NDArray,
    factor: int,
    ) -> NDArray:
    """
    Rebin an image in both dimensions by summing.
    
    Parameters
    ----------
    image : NDArray
        The image to rebin.
    factor : int
        The factor to rebin by.
    
    Returns
    -------
    NDArray
        The rebinned image.
    
    Raises
    ------
    ValueError
        If the image cannot be downsampled by the desired factor.
    """
    
    if image.shape[0] % factor != 0 or image.shape[1] % factor != 0:
        raise ValueError(f'[PHOPTIC] The dimensions of the input data must be divisible by the rebinning factor. Got shape {image.shape} and factor {factor}.')
    
    shape = (image.shape[0] // factor, factor, image.shape[1] // factor, factor)
    reshaped_data = image.reshape(shape)
    
    return reshaped_data.sum(axis=(1, 3))

