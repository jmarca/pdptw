import numpy as np
from multiprocessing import Pool
from itertools import combinations

def generate_locations( extents, num, stddev = 6 ):
    """
    A function to generate locations inside a box.

    Args:
       extents:  hash with the following entries:
           'lllon': lower left longitude,
           'lllat': lower left latitude
           'urlon': upper right longitide
           'urlat': upper right latitiude
       num:  how many points to generate
       stddev: the standard deviation of the normal distribution.  Default 6

    Returns:
        zip (lats, lons)

    """
    stdv = 6  # the number of standard deviations 99.9% will be within +-3
    lats = (extents['lllat'] +
            np.random.randn(num) *
            (extents['urlat'] - extents['lllat']) / stdv)
    lons = (extents['lllon'] +
            np.random.randn(num) *
            (extents['urlon'] - extents['lllon']) / stdv)

    return (lats, lons)


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth specified in decimal degrees of latitude and longitude.
    https://en.wikipedia.org/wiki/Haversine_formula

    Args:
        lon1: longitude of pt 1,
        lat1: latitude of pt 1,
        lon2: longitude of pt 2,
        lat2: latitude of pt 2

    Returns:
        the distace in km between pt1 and pt2
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (np.sin(dlat / 2) ** 2 + np.cos(lat1) *
         np.cos(lat2) * np.sin(dlon / 2) ** 2)
    c = 2 * np.arcsin(np.sqrt(a))

    # 6367 km is the radius of the Earth
    km = 6367 * c
    return km

def f(paired_indices):
    a,b = paired_indices
    alat,alon = a
    blat,blon = b
    tripd = haversine(alon,
                      alat,
                      blon,
                      blat)
    return tripd

def make_distance_mat(customers, method='haversine'):
    """
    Return a distance matrix and make it a member of Customer, using the
    method given in the call. Currently only Haversine (GC distance) is
    implemented, but Manhattan, or using a maps API could be added here.
    Raises an AssertionError for all other methods.

    Args:
        method (Optional[str]): method of distance calculation to use. The
            Haversine formula is the only method implemented.

    Returns:
        Numpy array of node to node distances.

    Examples:
        >>> dist_mat = customers.make_distance_mat(method='haversine')
        >>> dist_mat = customers.make_distance_mat(method='manhattan')
    AssertionError
    """
    number = len(customers)
    print ("DIST MAT SIZE",number,number)
    distmat = np.zeros((number, number))
    methods = {'haversine': haversine}
    assert(method in methods)


    def g(paired_indices,tripd):
        # print(paired_indices,tripd)
        frm_idx,to_idx = paired_indices

        if customers[frm_idx].demand == 0 or customers[to_idx].demand == 0:
            distmat[frm_idx, to_idx] = 0.1 * tripd
            distmat[to_idx, frm_idx] = 0.1 * tripd
        else:
            distmat[frm_idx, to_idx] = tripd
            distmat[to_idx, frm_idx] = tripd



    p = Pool(4)
    distances = p.map(f,combinations(((c.lat,c.lon) for c in customers),2))
    for i in map (g,combinations(range(number),2),distances):
        next

    # print (distmat)
    # print (1/0)
    return distmat
