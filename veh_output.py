from datetime import datetime, timedelta


def build_vehicle_route(routing, plan, customers, veh_number):
    """
    Build a route for a vehicle by starting at the strat node and
    continuing to the end node.

    Args:
        routing (ortools.constraint_solver.pywrapcp.RoutingModel): routing.

        plan (ortools.constraint_solver.pywrapcp.Assignment): the assignment.

        customers (Customers): the customers instance.

        veh_number (int): index of the vehicle
    Returns:
        (List) route: indexes of the customers for vehicle veh_number
    """
    veh_used = routing.IsVehicleUsed(plan, veh_number)
    print('Vehicle {0} is used {1}'.format(veh_number, veh_used))
    if veh_used:
        route = []
        node = routing.Start(veh_number)  # Get the starting node index
        route.append(customers.customers[routing.IndexToNode(node)])
        while not routing.IsEnd(node):
            route.append(customers.customers[routing.IndexToNode(node)])
            node = plan.Value(routing.NextVar(node))

        route.append(customers.customers[routing.IndexToNode(node)])
        return route
    else:
        return None


def vehicle_output_string(routing, plan):
    """
    Return a string displaying the output of the routing instance and
    assignment (plan).

    Args:
        routing (ortools.constraint_solver.pywrapcp.RoutingModel): routing.

        plan (ortools.constraint_solver.pywrapcp.Assignment): the assignment.
    Returns:
        (string) plan_output: describing each vehicle's plan.

        (List) dropped: list of dropped orders.

    """
    dropped = []
    for order in range(routing.Size()):
        if (plan.Value(routing.NextVar(order)) == order):
            dropped.append(str(order))

    capacity_dimension = routing.GetDimensionOrDie("Capacity")
    time_dimension = routing.GetDimensionOrDie("Time")
    plan_output = ''

    for route_number in range(routing.vehicles()):
        order = routing.Start(route_number)
        plan_output += 'Route {0}:'.format(route_number)
        if routing.IsEnd(plan.Value(routing.NextVar(order))):
            plan_output += ' Empty \n'
        else:
            while True:
                load_var = capacity_dimension.CumulVar(order)
                time_var = time_dimension.CumulVar(order)
                plan_output += \
                    " {order} Load({load}) Time({tmin}, {tmax}) -> ".format(
                        order=order,
                        load=plan.Value(load_var),
                        tmin=str(timedelta(seconds=plan.Min(time_var))),
                        tmax=str(timedelta(seconds=plan.Max(time_var))))

                if routing.IsEnd(order):
                    plan_output += ' EndRoute {0}. \n'.format(route_number)
                    break
                order = plan.Value(routing.NextVar(order))
        plan_output += "\n"

    return(plan_output, dropped)
