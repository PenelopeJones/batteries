import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rc('font', family='Times New Roman')

import pdb

def rotate_points(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return qx, qy

def find_peak(arr):
    idx = []
    for i in range(1, arr.shape[0] - 1):
        if (arr[i-1] <= arr[i]) and (arr[i+1] <= arr[i]):
            idx.append(i)
    idx = np.array(idx)
    if idx.shape[0] == 1:
        return idx[0]
    elif idx.shape[0] == 0:
        return None
    else:
        opt = arr[idx].argmax()
        return idx[opt]

def cubic_minimiser(p):
    disc = p[1]**2 - 3*p[0]*p[2]
    if disc > 0:
        if p[0] < 0:
            return (-p[1] - disc**0.5) / (3*p[0])
        else:
            return (-p[1] + disc ** 0.5) / (3 * p[0])
    else:
        return None

def find_inflection(re_z, im_z, rotation_angle, rotation_origin, idxmin, idxmax):
    x = []
    y = []

    for i in range(re_z.shape[0]):
        projection_x, projection_y = rotate_points(rotation_origin, (re_z[i], im_z[i]), -rotation_angle)
        x.append(projection_x)
        y.append(projection_y)
    x = np.array(x)
    y = np.array(y)
    gradients = []
    for i in range(idxmin, idxmax):
        gradient = (y[i] - y[i-1]) / (x[i] - x[i-1])
        gradients.append(gradient)
    gradients = np.array(gradients)
    p = np.polyfit(x[idxmin:idxmax], gradients, 3)
    x_min = cubic_minimiser(p)
    if x_min is None:
        return None
    else:
        idx = (np.abs(x - x_min)).argmin()
        return idx

def find_valley(arr):
    idx = []
    for i in range(1, arr.shape[0] - 1):
        if (arr[i-1] >= arr[i]) and (arr[i+1] >= arr[i]):
            idx.append(i)
    idx = np.array(idx)
    if idx.shape[0] == 1:
        return idx[0]

    elif idx.shape[0] == 0:
        return None

    else:
        opt = arr[idx].argmin()
        return idx[opt]

def distance_angle(x1, y1, x2, y2):
    distance = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
    angle = np.arctan((y2 - y1) / (x2 - x1))

    return distance, angle

def perp_parallel_distance(p0, p1, p2):
    """
    Returns the perpendicular distance of p2 from the line between p0 and p1,
    and also the parallel distance from p0 along the line between p0 and p1 to
    get to the point of perpendicular intersection
    :param p0:
    :param p1:
    :param p2:
    :return:
    """
    perp = np.abs(np.cross((p1 - p0), (p2 - p0)) / np.linalg.norm(p1 - p0))
    par = np.abs(np.dot((p1 - p0), (p2 - p0)) / np.linalg.norm(p1 - p0))
    return perp, par



def extract_features(re_z, im_z, log_omega):

    assert re_z.shape[0] == im_z.shape[0] == log_omega.shape[0]

    # First important datapoint is in lower left of EIS
    idx0 = re_z.argsort()[0]
    re0 = re_z[idx0]
    im0 = im_z[idx0]
    w0 = log_omega[idx0]

    # Second important datapoint is the "valley" - find location plus the angle
    idx1 = find_valley(im_z)
    if idx1 is None:
        return None
    re1 = re_z[idx1]
    im1 = im_z[idx1]
    w1 = log_omega[idx1]

    # Third important datapoint is the "peak" (i.e. maximum in Im(Z)?)
    idx2 = find_peak(im_z)
    if idx2 is None:
        return None
    re2 = re_z[idx2]
    im2 = im_z[idx2]
    w2 = log_omega[idx2]

    # Find the distance between these points, L1, and the angle from the horizontal, theta
    L1, theta = distance_angle(re0, im0, re1, im1)

    # Fourth important datapoint is the point of inflection...
    idx3 = find_inflection(re_z, im_z, theta, (re0, im0), idx0, idx2)
    if idx3 is None:
        return None
    re3 = re_z[idx3]
    im3 = im_z[idx3]
    w3 = log_omega[idx3]

    # Find perpendicular distance of p2 from the line between p0 and p1
    p0 = np.array([re0, im0])
    p1 = np.array([re1, im1])
    p2 = np.array([re2, im2])
    p3 = np.array([re3, im3])

    H2, L2 = perp_parallel_distance(p0, p1, p2)
    H3, L3 = perp_parallel_distance(p0, p1, p3)

    features = np.array([re0, im0, L1, theta, w1, H2, L2, w2, H3, L3, w3])

    return features


def plot_setup(figsize, linewidth, xmin, xmax, ymin, ymax, xticks, xticklabels, yticks, yticklabels, fontsize):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    for axis in ['bottom', 'left']:
        ax.spines[axis].set_linewidth(linewidth)
    for axis in ['top', 'right']:
        ax.spines[axis].set_visible(False)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels(xticklabels, fontsize=fontsize)
    ax.set_yticklabels(yticklabels, fontsize=fontsize)
    return fig, ax