from __future__ import print_function
import math
from six.moves import xrange
from ortools.constraint_solver import pywrapcp
# You need to import routing_enums_pb2 after pywrapcp!
from ortools.constraint_solver import routing_enums_pb2

def distance(x1, y1, x2, y2):
    # Manhattan distance
    dist = abs(x1 - x2) + abs(y1 - y2)

    return dist
class CreateDistanceCallback(object):
  """Create callback to calculate distances between points."""

  def __init__(self, locations):
    """Initialize distance array."""
    size = len(locations)
    depot = 0
    self.matrix = {}

    for from_node in xrange(size):
      self.matrix[from_node] = {}
      for to_node in xrange(size):
        if from_node == depot or to_node == depot:
            # Define the distance from the depot to any node to be 0.
            self.matrix[from_node][to_node] = 0
        else:
            x1 = locations[from_node][0]
            y1 = locations[from_node][1]
            x2 = locations[to_node][0]
            y2 = locations[to_node][1]
            self.matrix[from_node][to_node] = distance(x1, y1, x2, y2)


  def Distance(self, from_node, to_node):
    return self.matrix[from_node][to_node]

# Demand callback
class CreateDemandCallback(object):
  """Create callback to get demands at each location."""

  def __init__(self, demands):
    self.matrix = demands

  def Demand(self, from_node, to_node):
    return self.matrix[from_node]

def main():
  # Create the data.
  data = create_data_array()
  locations = data[0]
  demands = data[1]
  num_locations = len(locations)
  depot = 0    # The depot is the start and end point of each route.
  num_vehicles = 5

  # Create routing model.
  if num_locations > 0:
    routing = pywrapcp.RoutingModel(num_locations, num_vehicles, depot)
    search_parameters = pywrapcp.RoutingModel.DefaultSearchParameters()

    # Setting first solution heuristic: the
    # method for finding a first solution to the problem.
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    # The 'PATH_CHEAPEST_ARC' method does the following:
    # Starting from a route "start" node, connect it to the node which produces the
    # cheapest route segment, then extend the route by iterating on the last
    # node added to the route.

    # Put a callback to the distance function here. The callback takes two
    # arguments (the from and to node indices) and returns the distance between
    # these nodes.

    dist_between_locations = CreateDistanceCallback(locations)
    dist_callback = dist_between_locations.Distance
    routing.SetArcCostEvaluatorOfAllVehicles(dist_callback)

    # Put a callback to the demands.
    demands_at_locations = CreateDemandCallback(demands)
    demands_callback = demands_at_locations.Demand

    # Add a dimension for demand.
    slack_max = 0
    vehicle_capacity = 100
    fix_start_cumul_to_zero = True
    demand = "Demand"
    routing.AddDimension(demands_callback, slack_max, vehicle_capacity,
                         fix_start_cumul_to_zero, demand)

    # Solve, displays a solution if any.
    assignment = routing.SolveWithParameters(search_parameters)
    if assignment:
      # Display solution.
      # Solution cost.
      print ("Total distance of all routes: " + str(assignment.ObjectiveValue()) + "\n")

      for vehicle_nbr in range(num_vehicles):
        index = routing.Start(vehicle_nbr)
        index_next = assignment.Value(routing.NextVar(index))
        route = ''
        route_dist = 0
        route_demand = 0

        while not routing.IsEnd(index_next):
          node_index = routing.IndexToNode(index)
          node_index_next = routing.IndexToNode(index_next)
          if node_index != depot:
            route += str(node_index) + " -> "

          # Add the distance to the next node.
          route_dist += dist_callback(node_index, node_index_next)
          # Add demand.
          route_demand += demands[node_index_next]
          index = index_next
          index_next = assignment.Value(routing.NextVar(index))

        node_index = routing.IndexToNode(index)
        node_index_next = routing.IndexToNode(index_next)
        route += str(node_index)#  + " -> " + str(node_index_next)
        route_dist += dist_callback(node_index, node_index_next)
        print ("Route for vehicle " + str(vehicle_nbr) + ":\n\n" + route + "\n")
        print ("Distance of route " + str(vehicle_nbr) + ": " + str(route_dist))
        print ("Demand met by vehicle " + str(vehicle_nbr) + ": " + str(route_demand) + "\n")
    else:
      print ('No solution found.')
  else:
    print ('Specify an instance greater than 0.')

def create_data_array():

  locations = [[82, 76], [96, 44], [50, 5], [49, 8], [13, 7], [29, 89], [58, 30], [84, 39],
               [14, 24], [12, 39], [3, 82], [5, 10], [98, 52], [84, 25], [61, 59], [1, 65],
               [88, 51], [91, 2], [19, 32], [93, 3], [50, 93], [98, 14], [5, 42], [42, 9],
               [61, 62], [9, 97], [80, 55], [57, 69], [23, 15], [20, 70], [85, 60], [98, 5]]

  demands = [0, 19, 21, 6, 19, 7, 12, 16, 6, 16, 8, 14, 21, 16, 3, 22, 18,
             19, 1, 24, 8, 12, 4, 8, 24, 24, 2, 20, 15, 2, 14, 9]
  data = [locations, demands]
  return data
if __name__ == '__main__':
  main()
