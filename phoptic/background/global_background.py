from abc import ABC, abstractmethod
from typing import Tuple


from numpy.typing import NDArray
from photutils.background import Background2D




class BaseBackground(ABC):
    """
    Base class for OPTICAM background estimators.
    """


    def __init__(
        self,
        box_size: int | Tuple[int, int],
        ):
        """
        Initialize a background estimator.
        
        Parameters
        ----------
        box_size : int | Tuple[int, int]
            Size of the background mesh "pixels". If an integer is provided, the mesh pixels are squares of size
            `box_size` x `box_size`. If a tuple is provided, the mesh pixels are rectangles of width `box_size[1]` and
            height `box_size[0]`.
        """
        
        if not isinstance(box_size, int):
            assert len(box_size) == 2, "[PHOPTIC] Incompatible box_size parameter passed to DefaultBackground. box_size must be either an integer or an interable of dimensions (e.g., [height, width])."
        
        self.box_size = box_size


    @abstractmethod
    def __call__(
        self,
        image: NDArray,
        ) -> Background2D:
        """
        Compute the 2D background for an image.
        
        Parameters
        ----------
        image : NDArray
            The image.
        
        Returns
        -------
        Background2D
            The two-dimensional background.
        """
        
        pass




class DefaultBackground(BaseBackground):
    """
    Default background estimator.
    """


    def __call__(
        self,
        image: NDArray,
        ) -> Background2D:
        """
        Compute the 2D background for an image.
        
        Parameters
        ----------
        image : NDArray
            The image.
        
        Returns
        -------
        Background2D
            The two-dimensional background.
        """
        
        return Background2D(image, box_size=self.box_size)