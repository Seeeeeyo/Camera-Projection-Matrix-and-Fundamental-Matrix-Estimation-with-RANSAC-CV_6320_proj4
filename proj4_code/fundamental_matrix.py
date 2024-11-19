"""Fundamental matrix utilities."""

import math
import numpy as np

def point_line_distance(line, point):
    """Calculate line-point distance according to the formula
    from the project webpage.

    d(l, x) = (au + bv + c) / sqrt(a^2 + b^2)

    Arguments:
        line {3-vector} -- Line coordinates a, b, c
        point {3-vector} -- homogeneous point u, v, w

        Note that we don't really use w because w = 1 for the
        homogeneous coordinates

    Returns:
        float -- signed distance between line and point
    """

    a, b, c = line
    u, v, w = point
    error = 0

    #######################################################################
    # YOUR CODE HERE                                                      #
    #######################################################################
    error = (a * u + b * v + c) / math.sqrt(a ** 2 + b ** 2)
    #######################################################################
    #                           END OF YOUR CODE                          #
    #######################################################################

    return error

def signed_point_line_errors(x_0s, F, x_1s):
    """Calculate all signed line-to-point distances. Takes as input
    the list of x_0 and x_1 points, as well as the current estimate
    for F, and calculates errors for every pair of points and
    returns it as a list of floats.

    You'll want to call point_line_distance() to get the error
    between line and point.

    Keep in mind that this is a symmetric line-to-point error,
    meaning we calculate the line-to-point distance between Fx_1 and
    x_0, as well as F'x_0 and x_1, where F' is F transposed. You'll
    also have to append the errors to the errors list in that order,
    d(Fx_1,x_0) first then d(F'x_0,x_1) for every pair of points.

    Helpful functions: np.dot()

    Arguments:
        x_0s {Nx3 list} -- points in image 1
        F {3x3 array} -- Fundamental matrix
        x_1s {Nx3 list} -- points in image 2

    Returns:
        [float] {2N} -- list of d(Fx_1,x_0) and d(F'x_0,x_1) for each
        pair of points, because SciPy takes care of squaring and
        summing
    """
    assert F.shape == (3, 3)
    assert len(x_0s) == len(x_1s)
    errors = []

    #######################################################################
    # YOUR CODE HERE  |  Distance Computation                             #
    #######################################################################
    x_0s = np.array(x_0s)
    x_1s = np.array(x_1s)
    F = np.array(F)
    rows = x_0s.shape[0]
    channels = x_0s.shape[1]

    if channels != 3:
        x_0s = np.insert(x_0s, 2, values=1, axis=1)
        x_1s = np.insert(x_1s, 2, values=1, axis=1)

    for i in range(0, rows):
        Fx1 = np.dot(F, x_1s[i][:])
        Fx0 = np.dot(F.T, x_0s[i][:])
        errors.append(point_line_distance(Fx1, x_0s[i][:]))
        errors.append(point_line_distance(Fx0, x_1s[i][:]))

    #######################################################################
    #                           END OF YOUR CODE                          #
    #######################################################################

    return errors

def skew(x, y, z):
    """Skew symmetric matrix."""
    # if x is numpy.ndarray, then x.flatten() will return a 1-D array
    if type(x) is np.ndarray:
        x = x[0]
        y = y[0]
        z = z[0]
    result = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return result


def create_F(K, R, t):
    """Create F from calibration and pose R, t between two views."""
    x, y, z = t  # Make sure t is indeed a 3-element list or array
    T = skew(x, y, z)
    Kinv = np.linalg.inv(K)
    F = np.dot(Kinv.T, T).dot(R).dot(Kinv)
    return F / np.linalg.norm(F)
