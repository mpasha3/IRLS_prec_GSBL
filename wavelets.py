import numpy as np
import pywt
#from scipy.sparse.linalg import LinearOperator

from jlinops.base import LinearOperator

# Function to flatten wavelet coefficients into a vector and pad if necessary
def flatten_coeffs(coeffs, original_shape):
    flattened = np.hstack([coeffs[0].flatten()] + [np.hstack([d.flatten() for d in details]) for details in coeffs[1:]])
    # Pad with zeros to match the original size if needed
    original_size = np.prod(original_shape)
    if flattened.size < original_size:
        flattened = np.pad(flattened, (0, original_size - flattened.size), mode='constant')
    return flattened

# Function to restore wavelet coefficients from a flattened vector
def unstack_coeffs(y, coeff_shapes):
    coeffs_restored = []
    idx = 0
    # Restore the approximation coefficients
    cA_shape = coeff_shapes[0]
    cA_size = np.prod(cA_shape)
    cA = y[idx:idx + cA_size].reshape(cA_shape)
    idx += cA_size
    coeffs_restored.append(cA)
    
    # Restore the detail coefficients for each level
    for cH_shape, cV_shape, cD_shape in coeff_shapes[1:]:
        cH_size = np.prod(cH_shape)
        cV_size = np.prod(cV_shape)
        cD_size = np.prod(cD_shape)
        cH = y[idx:idx + cH_size].reshape(cH_shape)
        idx += cH_size
        cV = y[idx:idx + cV_size].reshape(cV_shape)
        idx += cV_size
        cD = y[idx:idx + cD_size].reshape(cD_shape)
        idx += cD_size
        coeffs_restored.append((cH, cV, cD))
    
    return coeffs_restored

# Define the LinearOperator class for the wavelet transform
class WaveletTransformOperator2D(LinearOperator):
    def __init__(self, image_shape, wavelet='haar', level=3, device="cpu"):
        # Initialize the operator with the shape and dtype
        self.image_shape = image_shape
        self.wavelet = wavelet
        self.level = level

        # Perform a wavelet transform to determine the size of the flattened wavelet coefficients
        sample_image = np.zeros(image_shape)
        coeffs = pywt.wavedec2(sample_image, wavelet, level=level)
        self.coeff_shapes = [coeffs[0].shape] + [(d[0].shape, d[1].shape, d[2].shape) for d in coeffs[1:]]
        self.coeff_size = sum(np.prod(shape) for shape in [coeffs[0].shape] + [d[0].shape for d in coeffs[1:]])

        # Define the shape of the operator (image flattened length, image flattened length)
        super().__init__(dtype=np.float64, shape=(np.prod(image_shape), np.prod(image_shape)), device=device)

        # Initialize indices for approximation and detail coefficients
        self.approximation_indices, self.detail_indices = self._compute_indices()

    def _compute_indices(self):
        """Compute the indices of approximation and detail coefficients."""
        # Compute the starting index for detail coefficients
        approximation_size = np.prod(self.coeff_shapes[0])

        # Indices for the approximation coefficients (first part of the vector)
        approximation_indices = np.arange(approximation_size)

        # Indices for detail coefficients (everything after the approximation coefficients)
        detail_indices = np.arange(approximation_size, np.prod(self.image_shape))

        return approximation_indices, detail_indices

    def _matvec(self, x):
        """Apply the wavelet transform (A * x)."""
        # Reshape the input vector x to a 2D image
        image = x.reshape(self.image_shape)
        
        # Perform the wavelet transform
        coeffs = pywt.wavedec2(image, self.wavelet, level=self.level)
        
        # Flatten the coefficients into a vector and pad it to match the original size
        return flatten_coeffs(coeffs, self.image_shape)

    def _rmatvec(self, y):
        """Apply the adjoint (inverse wavelet transform, A^T * y)."""
        # Reshape the vector y into wavelet coefficients
        coeffs_restored = unstack_coeffs(y, self.coeff_shapes)
        
        # Perform the inverse wavelet transform
        image_reconstructed = pywt.waverec2(coeffs_restored, self.wavelet)
        
        # Flatten the reconstructed image
        return image_reconstructed.flatten()




