from dataclasses import dataclass
from pathlib import Path
import warnings


from astropy.io import fits
from astropy.io.fits import Header
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm


from phoptic.utils.constants import bar_format




@dataclass
class MEFSlice:
    """
    Helper class to represent a slice of a Multi-extension FITS (MEF) file.
    
    Parameters
    ----------
    path : Path
        The file path.
    ext : int
        The slice's extension number.
    """
    
    path: Path
    ext: int


    @property
    def key(self) -> str:
        """
        The file path and extension number combined to create a uniquely identifiable string.
        
        Returns
        -------
        str
            The uniquely identifiable string.
        """
        return str(self.path) + ' ' + str(self.ext)


    def get_header(self) -> Header:
        """
        Get the slice's header.
        
        Returns
        -------
        Header
            The slice's header.
        """
        
        with fits.open(self.path) as hdul:
            header = hdul[self.ext].header
        
        return header


    def get_data(self) -> NDArray[np.float64]:
        """
        Get the slice's data.
        
        Returns
        -------
        NDArray[np.float64]
            The slice's data.
        """
        
        with fits.open(self.path) as hdul:
            data = hdul[self.ext].data.astype(np.float64)
        
        return data


    def get_data_and_header(self) -> tuple[NDArray[np.float64], Header]:
        """
        Get the slice's data and header.
        
        Returns
        -------
        tuple[NDArray[np.float64], Header]
            The slice's data and header.
        """
        
        with fits.open(self.path) as hdul:
            header = hdul[self.ext].header
            data = hdul[self.ext].data.astype(np.float64)
        
        return data, header




def create_file_paths(
    data_directory: Path,
    ) -> list[MEFSlice]:
    """
    Given a directory, return all the (multi-extension) FITS file paths and extension numbers.
    
    Parameters
    ----------
    data_directory : Path
        The directory containing one or more (multi-extension) FITS files.
    
    Returns
    -------
    list[MEFSlice]
        The list of (multi-extension) FITS slices.
    """
    
    file_paths: list[MEFSlice] = []
    
    fits_files = list(data_directory.glob('*'))
    
    for path in tqdm(fits_files, desc='[PHOPTIC] Scanning data directory', bar_format=bar_format):
        if path.is_file():
            try:
                with fits.open(path.resolve()) as hdul:
                    for ext, hdu in enumerate(hdul):
                        if hdu.data is not None:
                            file_paths.append(MEFSlice(path=path.resolve(), ext=ext))
            except Exception as e:
                warnings.warn(f'[PHOPTIC] Could not open file {path.resolve()} due to the following exception: {e}')
    
    return file_paths