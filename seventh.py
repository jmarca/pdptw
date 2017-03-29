# first attempt at pdptw, modifying vehicle routing with resource
# constraints (sixth.py)

from __future__ import print_function
import math
from six.moves import xrange
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2


#------------------------------data reading-------------------

# basic approach for reading file taken from pdptw.cc in google or code

def readdata(filename):
    """Read data from <filename>."""
    f = open(filename)
    num_vehicles, capacity, speed = [int(nb) for nb in f.readline().split()]
    assert num_vehicles > 0
    assert capacity > 0
    assert speed > 0

    # lists to store the parsed file
    customer_ids = []
    coords = []
    demands = []
    open_times = []
    close_times = []
    service_times = []
    pickups = []
    deliveries = []

    horizon = 0
    depot = 0

    for line in f:
        split_line = [int(nb) for nb in line.split()]
        assert len(split_line) == 9
        assert split_line[0] >= 0
        assert split_line[4] >= 0
        assert split_line[5] >= 0
        assert split_line[6] >= 0
        assert split_line[7] >= 0
        assert split_line[8] >= 0

        customer_id = split_line[0]
        x = split_line[1]
        y = split_line[2]
        delivery = split_line[8] # parse delivery before demand
        # conditional assignment in python
        demand = -split_line[3] if delivery == 0 else split_line[3]
        open_time = split_line[4]
        close_time = split_line[5]
        service_time = split_line[6]
        pickup = split_line[7]

        # put those items into the lists, as appropriate
        customer_ids.append(customer_id)

        # have to figure out how I'm doing coordinates
        coords.append((x, y))

        demand.append(demand)
        open_times.append(open_time)
        close_times.append(close_time)
        service_times.append(service_time)

        # C++ code used class RoutingModel::NodeIndex for these
        pickups.append(pickup)
        deliveries.append(delivery)

        if pickup == 0 and delivery == 0:
            depot = len(pickups) -1

        if horizon < close_time:
            horizon = close_time


    # bad componentization
    # can I return a collection?
    problem = {'customer_ids': customer_ids,
               'coords'        :coords,
               'demands'       :demands,
               'open_times'    :open_times,
               'close_times'   :close_times,
               'service_times' :service_times,
               'pickups'       :pickups,
               'deliveries'    :deliveries,
               'horizon'       :horizon,
               'depot'         :depot}
    return problem




def distance(x1, y1, x2, y2):
    # Manhattan distance
    dist = abs(x1 - x2) + abs(y1 - y2)

    return dist

# Distance callback

class CreateDistanceCallback(object):
    """Create callback to calculate distances and travel times between points."""

    def __init__(self, locations):
        """Initialize distance array."""

        num_locations = len(locations)
        self.matrix = {}

        for from_node in xrange(num_locations):
            self.matrix[from_node] = {}
            for to_node in xrange(num_locations):
                x1 = locations[from_node][0]
                y1 = locations[from_node][1]
                x2 = locations[to_node][0]
                y2 = locations[to_node][1]
                self.matrix[from_node][to_node] = distance(x1, y1, x2, y2)


    def Distance(self, from_node, to_node):
        return self.matrix[from_node][to_node]


# Demand callback
class CreateDemandCallback(object):
    """Create callback to get demands at each node."""

    def __init__(self, demands):
        self.matrix = demands

        def Demand(self, from_node, to_node):
            return self.matrix[from_node]

# Service time (proportional to demand) + transition time callback.
class CreateServiceTimeCallback(object):
    """Create callback to get time windows at each node."""

    def __init__(self, demands, time_per_demand_unit):
        self.matrix = demands
        self.time_per_demand_unit = time_per_demand_unit

        def ServiceTime(self, from_node, to_node):
            return self.matrix[from_node] * self.time_per_demand_unit

# Create total_time callback (equals service time plus travel time).
class CreateTotalTimeCallback(object):
    def __init__(self, service_time_callback, dist_callback, speed):
        self.service_time_callback = service_time_callback
        self.dist_callback = dist_callback
        self.speed = speed

    def TotalTime(self, from_node, to_node):
        stime = self.service_time_callback(from_node, to_node)
        dtime = self.dist_callback(from_node, to_node) / self.speed
        return stime + dtime

def DisplayPlan(routing, assignment):

    # Display dropped orders.
    dropped = ''

    for order in range(1, routing.nodes()):
        if assignment.Value(routing.NextVar(order)) == order:
            if dropped.empty():
                dropped += " %d", order
            else: dropped += ", %d", order

    if not dropped.empty():
        plan_output += "Dropped orders:" + dropped + "\n"

    return plan_output

def main():
    print('to be done')

if __name__ == '__main__':
    main()
