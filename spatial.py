import numpy as np

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
