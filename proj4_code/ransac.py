import numpy as np
import math
from least_squares_fundamental_matrix import solve_F
import two_view_data
import fundamental_matrix


def calculate_num_ransac_iterations(prob_success, sample_size, ind_prob_correct):
    """
    Calculate the number of RANSAC iterations needed for a given guarantee of success.

    Args:
    -   prob_success: float representing the desired guarantee of success
    -   sample_size: int the number of samples included in each RANSAC iteration
    -   ind_prob_success: float the probability that each element in a sample is correct

    Returns:
    -   num_samples: int the number of RANSAC iterations needed

    """

    #######################################################################
    # YOUR CODE HERE                                                      #
    #######################################################################
    num_samples = math.log(1 - prob_success) / math.log(1 - ind_prob_correct ** sample_size)
    #######################################################################
    #                           END OF YOUR CODE                          #
    #######################################################################

    return int(num_samples)


def find_inliers(x_0s, F, x_1s, threshold):
    """ Find the inliers' indices for a given model.

    There are multiple methods you could use for calculating the error
    to determine your inliers vs outliers at each pass. However, we suggest
    using the line to point distance function we wrote for the
    optimization in part 2.

    Args:
    -   x_0s: A numpy array of shape (N, 2) representing the coordinates
                   of possibly matching points from the left image
    -   F: The proposed fundamental matrix
    -   x_1s: A numpy array of shape (N, 2) representing the coordinates
                   of possibly matching points from the right image
    -   threshold: the maximum error for a point correspondence to be
                    considered an inlier
    Each row in x_1s and x_0s is a proposed correspondence (e.g. row #42 of x_0s is a point that
    corresponds to row #42 of x_1s)

    Returns:
    -    inliers: 1D array of the indices of the inliers in x_0s and x_1s

    """

    #######################################################################
    # YOUR CODE HERE                                                      #
    #######################################################################
    errors = np.abs(np.array(fundamental_matrix.signed_point_line_errors(x_0s, F, x_1s)))
    error = (errors[0: len(errors): 2] + errors[1: len(errors): 2])/2
    inliers = np.where(error < threshold)[0]

    #######################################################################
    #                           END OF YOUR CODE                          #
    #######################################################################

    return inliers


def ransac_fundamental_matrix(x_0s, x_1s):
    """Find the fundamental matrix with RANSAC.

    Use RANSAC to find the best fundamental matrix by
    randomly sampling interest points. You will call your
    solve_F() from part 2 of this assignment
    and calculate_num_ransac_iterations().

    You will also need to define a new function (see above) for finding
    inliers after you have calculated F for a given sample.

    Tips:
        0. You will need to determine your P, k, and p values.
            What is an acceptable rate of success? How many points
            do you want to sample? What is your estimate of the correspondence
            accuracy in your dataset?
        1. A potentially useful function is numpy.random.choice for
            creating your random samples
        2. You will want to call your function for solving F with the random
            sample and then you will want to call your function for finding
            the inliers.
        3. You will also need to choose an error threshold to separate your
            inliers from your outliers. We suggest a threshold of 1.

    Args:
    -   x_0s: A numpy array of shape (N, 2) representing the coordinates
                   of possibly matching points from the left image
    -   x_1s: A numpy array of shape (N, 2) representing the coordinates
                   of possibly matching points from the right image
    Each row is a proposed correspondence (e.g. row #42 of x_0s is a point that
    corresponds to row #42 of x_1s)

    Returns:
    -   best_F: A numpy array of shape (3, 3) representing the best fundamental
                matrix estimation
    -   inliers_x_0: A numpy array of shape (M, 2) representing the subset of
                   corresponding points from the left image that are inliers with
                   respect to best_F
    -   inliers_x_1: A numpy array of shape (M, 2) representing the subset of
                   corresponding points from the right image that are inliers with
                   respect to best_F

    """

    #######################################################################
    # YOUR CODE HERE                                                      #
    #######################################################################
    num_samples = 9
    ind_prob_correct = 0.9
    threshold = 1
    prob_success = 0.99
    num_samples = calculate_num_ransac_iterations(prob_success, num_samples, ind_prob_correct)
    best_F = None
    inliers_x_0 = None
    inliers_x_1 = None
    max_inliers = 0
    for i in range(num_samples):
        sample_indices = np.random.choice(len(x_0s), num_samples, replace=False)
        sample_x_0s = x_0s[sample_indices]
        sample_x_1s = x_1s[sample_indices]
        F = solve_F(sample_x_0s, sample_x_1s)
        inliers = find_inliers(x_0s, F, x_1s, threshold)
        if len(inliers) > max_inliers:
            max_inliers = len(inliers)
            best_F = F
            inliers_x_0 = x_0s[inliers]
            inliers_x_1 = x_1s[inliers]
    #######################################################################
    #                           END OF YOUR CODE                          #
    #######################################################################

    return best_F, inliers_x_0, inliers_x_1


def test_with_epipolar_lines():
    """Unit test you will create for your RANSAC implementation.

    It should take no arguments and it does not need to return anything,
    but it **must** display the images when run.

    Use the code in the jupyter notebook as an example for how to open the
    image files and perform the necessary operations on them in our workflow.
    Remember the steps are Harris, SIFT, match features, RANSAC fundamental matrix.

    Display the proposed correspondences, the true inlier correspondences
    found by RANSAC, and most importantly the epipolar lines in both of your images.
    It should be clear that the epipolar lines intersect where the second image
    was taken, and the true point correspondences should indeed be good matches.

    """

    #######################################################################
    # YOUR CODE HERE                                                      #
    #######################################################################


    #######################################################################
    #                           END OF YOUR CODE                          #
    #######################################################################
