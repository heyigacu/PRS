"""
# Author: Yi He
# reference mdanalysis
"""
import numpy as np

def ml_covariance_estimator(coordinates, reference_coordinates=None):
    """
    Standard maximum likelihood estimator of the covariance matrix.

    Parameters
    ----------

    coordinates : numpy.array
        Flattened array of coordiantes

    reference_coordinates : numpy.array
        Optional reference to use instead of mean

    Returns
    -------

    cov_mat : numpy.array
        Estimate of  covariance matrix

    """

    if reference_coordinates is not None:

        # Offset from reference
        coordinates_offset = coordinates - reference_coordinates

    else:
        # Normal covariance calculation: distance to the average
        coordinates_offset = coordinates - np.average(coordinates, axis=0)

    # Calculate covariance manually
    coordinates_cov = np.zeros((coordinates.shape[1],
                                coordinates.shape[1]))
    for frame in coordinates_offset:
        coordinates_cov += np.outer(frame, frame)
    coordinates_cov /= coordinates.shape[0]

    return coordinates_cov

def shrinkage_covariance_estimator( coordinates,
                                    reference_coordinates=None,
                                    shrinkage_parameter=None):
    """
    Shrinkage estimator of the covariance matrix using the method described in

    Improved Estimation of the Covariance Matrix of Stock Returns With an
    Application to Portfolio Selection. Ledoit, O.; Wolf, M., Journal of
    Empirical Finance, 10, 5, 2003

    This implementation is based on the matlab code made available by Olivier
    Ledoit on his website:
    http://www.ledoit.net/ole2_abstract.htm

    Parameters
    ----------

    coordinates : numpy.array
        Flattened array of coordinates

    reference_coordinates: numpy.array
        Optional reference to use instead of mean

    shrinkage_parameter: None or float
        Optional shrinkage parameter

    Returns
    --------

    cov_mat : nump.array
        Covariance matrix
    """

    x = coordinates.astype(float)
    t = x.shape[0]
    n = x.shape[1]

    mean_x = np.average(x, axis=0)

    # Use provided coordinates as "mean" if provided
    if reference_coordinates is not None:
        mean_x = reference_coordinates

    x = x - mean_x
    xmkt = np.average(x, axis=1)

    # Call maximum likelihood estimator (note the additional column)
    sample = ml_covariance_estimator(np.hstack([x, xmkt[:, np.newaxis]]), 0)\
        * (t-1)/float(t)

    # Split covariance matrix into components
    covmkt = sample[0:n, n]
    varmkt = sample[n, n]
    sample = sample[:n, :n]

    # Prior
    prior = np.outer(covmkt, covmkt)/varmkt
    prior[np.ma.make_mask(np.eye(n))] = np.diag(sample)

    # If shrinkage parameter is not set, estimate it
    if shrinkage_parameter is None:

        # Frobenius norm
        c = np.linalg.norm(sample - prior, ord='fro')**2

        y = x**2
        p = 1/float(t)*np.sum(np.dot(np.transpose(y), y))\
            - np.sum(np.sum(sample**2))
        rdiag = 1/float(t)*np.sum(np.sum(y**2))\
            - np.sum(np.diag(sample)**2)
        z = x * np.repeat(xmkt[:, np.newaxis], n, axis=1)
        v1 = 1/float(t) * np.dot(np.transpose(y), z) \
            - np.repeat(covmkt[:, np.newaxis], n, axis=1)*sample
        roff1 = (np.sum(
            v1*np.transpose(
                np.repeat(
                    covmkt[:, np.newaxis], n, axis=1)
                )
            )/varmkt -
                 np.sum(np.diag(v1)*covmkt)/varmkt)
        v3 = 1/float(t)*np.dot(np.transpose(z), z) - varmkt*sample
        roff3 = (np.sum(v3*np.outer(covmkt, covmkt))/varmkt**2 -
                 np.sum(np.diag(v3)*covmkt**2)/varmkt**2)
        roff = 2*roff1-roff3
        r = rdiag+roff

        # Shrinkage constant
        k = (p-r)/c
        shrinkage_parameter = max(0, min(1, k/float(t)))

    # calculate covariance matrix
    sigma = shrinkage_parameter*prior+(1-shrinkage_parameter)*sample

    return sigma