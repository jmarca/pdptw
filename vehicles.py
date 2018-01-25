from collections import namedtuple
import numpy as np
class Vehicles():
    """
    A Class to create and hold vehicle information.

    The Vehicles in a CVRPTW problem service the customers and belong to a
    depot. The class Vehicles creates a list of named tuples describing the
    Vehicles.  The main characteristics are the vehicle capacity, fixed cost,
    and cost per km.  The fixed cost of using a certain type of vehicles can be
    higher or lower than others. If a vehicle is used, i.e. this vehicle serves
    at least one node, then this cost is added to the objective function.

    Note:
        If numpy arrays are given for capacity and cost, then they must be of
        the same length, and the number of vehicles are infered from them.
        If scalars are given, the fleet is homogenious, and the number of
        vehicles is determied by number.

    Args:
        capacity (scalar or numpy array): The integer capacity of demand units.

        cost (scalar or numpy array): The fixed cost of the vehicle.

        number (Optional [int]): The number of vehicles in a homogeneous fleet.
    """
    def __init__(self, capacity=100, cost=100, number=None):

        Vehicle = namedtuple("Vehicle", ['index', 'capacity', 'cost'])

        if number is None:
            self.number = np.size(capacity)
        else:
            self.number = number
        idxs = np.array(range(0, self.number))

        if np.isscalar(capacity):
            capacities = capacity * np.ones_like(idxs)
        elif np.size(capacity) != np.size(capacity):
            print('capacity is neither scalar, nor the same size as num!')
        else:
            capacities = capacity

        if np.isscalar(cost):
            costs = cost * np.ones_like(idxs)
        elif np.size(cost) != self.number:
            print(np.size(cost))
            print('cost is neither scalar, nor the same size as num!')
        else:
            costs = cost

        self.vehicles = [Vehicle(idx, capacity, cost) for idx, capacity, cost
                         in zip(idxs, capacities, costs)]

    def get_total_capacity(self):
        return(sum([c.capacity for c in self.vehicles]))

    def return_starting_callback(self, customers, sameStartFinish=True):
        # create a different starting and finishing depot for each vehicle
        depots = customers.depots # array of depot indices
        num_depots = len(depots)
        if sameStartFinish:
            self.starts = [depots[o % num_depots]
                           for o in
                           range(self.number)]
            self.ends = self.starts
        else:
            start_depots = round(num_depots/2)
            end_depots = num_depots - start_depots
            self.starts = [depots[o % start_depots]
                           for o in
                           range(self.number)]
            self.ends = [depots[start_depots + (o % end_depots)]
                         for o in
                         range(self.number)]
            # print(self.starts)
            # print(self.ends)

        # the depots will not have demands, so zero them.  they should
        # already be zero.  this is vestigal but catches bugs while
        # I'm transitioning code
        for depot in self.starts:
            customers.zero_depot_demands(depot)
        for depot in self.ends:
            customers.zero_depot_demands(depot)
        def start_return(v): return(self.starts[v])
        return start_return
