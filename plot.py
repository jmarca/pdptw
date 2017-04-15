import numpy as np
from matplotlib import pyplot as plt

def discrete_cmap(N, base_cmap=None):
    """
    Create an N-bin discrete colormap from the specified input map
    """
    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)

def plot_vehicle_routes(veh_route, ax1, customers, vehicles):
    """
    Plot the vehicle routes on matplotlib axis ax1.

    Args:
        veh_route (dict): a dictionary of routes keyed by vehicle idx.

        ax1 (matplotlib.axes._subplots.AxesSubplot): Matplotlib axes

        customers (Customers): the customers instance.

        vehicles (Vehicles): the vehicles instance.
    """
    veh_used = [v for v in veh_route if veh_route[v] is not None]

    cmap = discrete_cmap(vehicles.number+2, 'nipy_spectral')



    for veh_number in veh_used:

        lats, lons, demands = zip(*[(c.lat, c.lon, c.demand) for c in veh_route[veh_number]])
        lats = np.array(lats)
        lons = np.array(lons)
        pickup_lats, pickup_lons = zip(*[(lats[i],lons[i]) for i in range(0,len(lats)) if demands[i] > 0])
        delivery_lats, delivery_lons = zip(*[(lats[i],lons[i]) for i in range(0,len(lats)) if demands[i] < 0])
        sigil = ['v' if d > 0 else '^' for d in demands]
        s_dep = customers.customers[vehicles.starts[veh_number]]
        s_fin = customers.customers[vehicles.ends[veh_number]]
        ax1.annotate('v({veh}) S @ {node}'.format(
                        veh=veh_number,
                        node=vehicles.starts[veh_number]),
                     xy=(s_dep.lon, s_dep.lat),
                     xytext=(10, 10),
                     xycoords='data',
                     textcoords='offset points',
                     arrowprops=dict(
                        arrowstyle="->",
                        connectionstyle="angle3,angleA=90,angleB=0",
                        shrinkA=0.05),
                     )
        ax1.annotate('v({veh}) F @ {node}'.format(
                        veh=veh_number,
                        node=vehicles.ends[veh_number]),
                     xy=(s_fin.lon, s_fin.lat),
                     xytext=(10, -20),
                     xycoords='data',
                     textcoords='offset points',
                     arrowprops=dict(
                        arrowstyle="->",
                        connectionstyle="angle3,angleA=-90,angleB=0",
                        shrinkA=0.05),
                     )
        #ax1.plot(lons, lats, 'o', mfc=cmap(veh_number+1))
        ax1.plot(pickup_lons, pickup_lats, '^', mfc=cmap(veh_number+1))
        ax1.plot(delivery_lons, delivery_lats, 'v', mfc=cmap(veh_number+1))
        ax1.quiver(lons[:-1], lats[:-1],
                   lons[1:]-lons[:-1], lats[1:]-lats[:-1],
                   scale_units='xy', angles='xy', scale=1,
                   color=cmap(veh_number+1))
