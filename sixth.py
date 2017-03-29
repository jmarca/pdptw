# vehicle routing with resource constraints

from __future__ import print_function
import math
from six.moves import xrange
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2

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
    return self.service_time_callback(from_node, to_node) + self.dist_callback(from_node, to_node) / self.speed

def DisplayPlan(routing, assignment):

  # Display dropped orders.
  dropped = ''

  for order in range(1, routing.nodes()):
    if assignment.Value(routing.NextVar(order)) == order:
      if (dropped.empty()):
        dropped += " %d", order
      else: dropped += ", %d", order

  if not dropped.empty():
    plan_output += "Dropped orders:" + dropped + "\n"

  return plan_output

def main():
  # Create the data.
  data = create_data_array()
  locations = data[0]
  demands = data[1]
  start_times = data[2]
  num_locations = len(locations)
  depot = 0
  num_vehicles = 10
  search_time_limit = 5000

  # Create routing model.
  if num_locations > 0:

    # The number of nodes of the VRP is num_locations.
    # Nodes are indexed from 0 to tsp_size - 1. By default the start of
    # a route is node 0.
    routing = pywrapcp.RoutingModel(num_locations, num_vehicles, depot)
    search_parameters = pywrapcp.RoutingModel.DefaultSearchParameters()

    # Set first solution heuristic: the
    # method for finding a first solution to the problem.
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    # The 'PATH_CHEAPEST_ARC' method does the following:
    # Starting from a route "start" node, connect it to the node which produces the
    # cheapest route segment, then extend the route by iterating on the last
    # node added to the route.

    # Set the time limit
    search_parameters.time_limit_ms = search_time_limit

    # Put callbacks to the distance function here.

    dist_between_locations = CreateDistanceCallback(locations)
    dist_callback = dist_between_locations.Distance

    routing.SetArcCostEvaluatorOfAllVehicles(dist_callback)
    demands_at_locations = CreateDemandCallback(demands)
    demands_callback = demands_at_locations.Demand

    # Add capacity dimension constraints.
    vehicle_capacity = 50;
    null_capacity_slack = 0;
    fix_start_cumul_to_zero = True
    capacity = "Capacity"
    routing.AddDimension(demands_callback, null_capacity_slack, vehicle_capacity,
                         fix_start_cumul_to_zero, capacity)

    # Adding time dimension constraints.
    time_per_demand_unit = 300
    horizon = 24 * 3600
    time = "Time"
    tw_duration = 5 * 3600
    speed = 10

    service_times = CreateServiceTimeCallback(demands, time_per_demand_unit)
    service_time_callback = service_times.ServiceTime
    total_times = CreateTotalTimeCallback(service_time_callback, dist_callback, speed)
    total_time_callback = total_times.TotalTime

    # Note: In this case fix_start_cumul_to_zero is set to False,
    # because some vehicles start their routes after time 0, due to resource constraints.

    fix_start_cumul_to_zero = False
    # Add a dimension for time and a limit on the total time_horizon
    routing.AddDimension(total_time_callback,  # total time function callback
                         horizon,
                         horizon,
                         fix_start_cumul_to_zero,
                         time)

    time_dimension = routing.GetDimensionOrDie("Time")

    for order in range(1, num_locations):
      start = start_times[order]
      time_dimension.CumulVar(order).SetRange(start, start + tw_duration)
    # Add resource constraints at the depot (start and end location of routes).

    vehicle_load_time = 180
    vehicle_unload_time = 180
    solver = routing.solver()
    intervals = []
    for i in range(num_vehicles):
      # Add time windows at start of routes
      intervals.append(solver.FixedDurationIntervalVar(routing.CumulVar(routing.Start(i), time),
                                                     vehicle_load_time,
                                                     "depot_interval"))
      # Add time windows at end of routes.
      intervals.append(solver.FixedDurationIntervalVar(routing.CumulVar(routing.End(i), time),
                                                     vehicle_unload_time,
                                                     "depot_interval"))
    # Constrain the number of maximum simultaneous intervals at depot.
    depot_capacity = 2;
    depot_usage = [1 for i in range(num_vehicles * 2)]
    solver.AddConstraint(
      solver.Cumulative(intervals, depot_usage, depot_capacity, "depot"))
    # Instantiate route start and end times to produce feasible times.
    for i in range(num_vehicles):
      routing.AddVariableMinimizedByFinalizer(routing.CumulVar(routing.End(i), time))
      routing.AddVariableMinimizedByFinalizer(routing.CumulVar(routing.Start(i), time))

    # Solve, displays a solution if any.
    assignment = routing.SolveWithParameters(search_parameters)
    if assignment:

      # Solution distance.
      print ("Total distance of all routes: " + str(assignment.ObjectiveValue()) + "\n")
      # Display solution.

      capacity_dimension = routing.GetDimensionOrDie(capacity);
      time_dimension = routing.GetDimensionOrDie(time);

      for vehicle_nbr in range(num_vehicles):
        index = routing.Start(int(vehicle_nbr))
        plan_output = 'Route {0}:'.format(vehicle_nbr)

        while not routing.IsEnd(index):
          node_index = routing.IndexToNode(index)
          load_var = capacity_dimension.CumulVar(index)
          time_var = time_dimension.CumulVar(index)
          plan_output += \
                    " {node_index} Load({load}) Time({tmin}, {tmax}) -> ".format(
                        node_index=node_index,
                        load=assignment.Value(load_var),
                        tmin=str(assignment.Min(time_var)),
                        tmax=str(assignment.Max(time_var)))
          index = assignment.Value(routing.NextVar(index))

        node_index = routing.IndexToNode(index)  # Convert index to node
        load_var = capacity_dimension.CumulVar(index)
        time_var = time_dimension.CumulVar(index)
        plan_output += \
                  " {node_index} Load({load}) Time({tmin}, {tmax})".format(
                      node_index=node_index,
                      load=assignment.Value(load_var),
                      tmin=str(assignment.Min(time_var)),
                      tmax=str(assignment.Max(time_var)))
        print (plan_output)
        print ("\n")
    else:
      print ('No solution found within '+str(search_time_limit/1000)+' seconds')
  else:
    print ('Specify an instance greater than 0.')
def create_data_array():

  locations = [[82, 76],
               [96, 44],
               [50, 5],
               [49, 8],
               [13, 7],
               [29, 89],
               [58, 30],
               [84, 39],
               [14, 24],
               [12, 39],
               [3, 82],
               [5, 10],
               [98, 52],
               [84, 25],
               [61, 59],
               [1, 65],
               [88, 51],
               [91, 2],
               [19, 32],
               [93, 3],
               [50, 93],
               [98, 14],
               [5, 42],
               [42, 9],
               [61, 62],
               [9, 97],
               [80, 55],
               [57, 69],
               [23, 15],
               [20, 70],
               [85, 60],
               [98, 5]]

  demands =  [0, 19, 21, 6, 19, 7, 12, 16, 6, 16, 8, 14, 21, 16, 3, 22, 18,
             19, 1, 24, 8, 12, 4, 8, 24, 24, 2, 20, 15, 2, 14, 9]

  start_times =  [28842, 50891, 10351, 49370, 22553, 53131, 8908,
                  56509, 54032, 10883, 60235, 46644, 35674, 30304,
                  39950, 38297, 36273, 52108, 2333, 48986, 44552,
                  31869, 38027, 5532, 57458, 51521, 11039, 31063,
                  38781, 49169, 32833, 7392]

  data = [locations, demands, start_times]
  return data

if __name__ == '__main__':
  main()
