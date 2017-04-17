from collections import namedtuple
import numpy as np
from datetime import datetime, timedelta
import spatial as spatial

class Customers():
    """
        A class that generates and holds customers information.

        Randomly normally distribute a number of customers and locations within
        a region described by a rectangle.  Generate a random demand for each
        customer. Generate a random time window for each customer.
        May either be initiated with the extents, as a dictionary describing
        two corners of a rectangle in latitude and longitude OR as a center
        point (lat, lon), and box_size in km.  The default arguments are for a
        10 x 10 km square centered in Sheffield).

        Args:
            extents (Optional[Dict]): A dictionary describing a rectangle in
                latitude and longitude with the keys 'llcrnrlat', 'llcrnrlon' &
                'urcrnrlat' & 'urcrnrlat'

            center (Optional(Tuple): A tuple of (latitude, longitude)
                describing the centre of the rectangle.

            box_size (Optional float: The length in km of the box's sides.

            num_custs (int): The number of customers, including the depots that
                are placed normally distributed in the rectangle.

            min_demand (int): Lower limit on the randomly generated demand at
                each customer.

            max_demand (int): Upper limit on the randomly generated demand at
                each customer.

            min_tw: shortest random time window for a customer, in hours.

            max_tw: longest random time window for a customer, in hours.

            num_depots: the number of depots to create.  Default 10.

            load_time: the time to load or unload one unit of demand, in
                 seconds.  Default is 300 (5 minutes).

        Examples:
            To place 100 customers randomly within 100 km x 100 km rectangle,
            centered in the default location, with a random demand of between 5
            and 10 units:

            >>> customers = Customers(num_custs=100, box_size=100,
            ...                 min_demand=5, max_demand=10)

            alternatively, to place 75 customers in the same area with default
            arguments for demand:

            >>> extents = {'urcrnrlon': 0.03403, 'llcrnrlon': -2.98325,
            ...     'urcrnrlat': 54.28127, 'llcrnrlat': 52.48150}
            >>> customers = Customers(num_custs=75, extents=extents)


    """
    def __init__(self, extents=None, center=(53.381393, -1.474611),
                 box_size=10, n=100,
                 min_demand=1, max_demand=5,
                 min_tw=1, max_tw=5, num_depots=10, load_time=300,
                 return_pu_win=5,
                 avg_speed=50,
                 earliest_start=8*3600):

        # the number of customer trips is 2 * the number of custs (n).
        # because right now everybody has an outbound and a return trip
        num_trips = 2*n
        self.num_trips = num_trips
        self.num_custs = n
        self.number = 2*num_trips+num_depots  #: The number of pickups, delivs, depots
        self.depots = range(2*num_trips,self.number)
        #: Location, a named tuple for locations.
        Location = namedtuple("Location", ['lat', 'lon'])
        if extents is not None:
            self.extents = extents  #: The lower left and upper right points
            #: Location[lat,lon]: the centre point of the area.
            self.center = Location(extents['urcrnrlat'] -
                                   0.5 * (extents['urcrnrlat'] -
                                          extents['llcrnrlat']),
                                   extents['urcrnrlon'] -
                                   0.5 * (extents['urcrnrlon'] -
                                          extents['llcrnrlon']))
        else:
            #: Location[lat,lon]: the centre point of the area.
            (clat, clon) = self.center = Location(center[0], center[1])
            rad_earth = 6367  # km
            circ_earth = np.pi * rad_earth
            #: The lower left and upper right points
            self.extents = {'lllon': (clon - 180 * box_size /
                                          (circ_earth *
                                           np.cos(np.deg2rad(clat)))),
                            'lllat': clat - 180 * box_size / circ_earth,
                            'urlon': (clon + 180 * box_size /
                                          (circ_earth *
                                           np.cos(np.deg2rad(clat)))),
                            'urlat': clat + 180 * box_size / circ_earth}
        # The 'name' of the cust.  Actually will be stops and depots
        custs = np.array(range(0, num_trips))
        # The 'name' of the delivery, indexed from num_trips to 2*num_trips-1
        delivs = np.array(range(num_trips,2*num_trips))

        depots = np.array(range(2*num_trips,2*num_trips + num_depots))

        # smush together
        stops = np.concatenate((custs,delivs,depots))

        # normaly distributed random distribution of custs and delivs within the box
        lats_out_orig, lons_out_orig = spatial.generate_locations (self.extents,n,6)
        lats_out_dest, lons_out_dest = spatial.generate_locations (self.extents,n,6)
        lats_in_orig = np.copy(lats_out_dest)
        lons_in_orig = np.copy(lons_out_dest)
        lats_in_dest = np.copy(lats_out_orig)
        lons_in_dest = np.copy(lons_out_orig)
        lats_dep, lons_dep = spatial.generate_locations (self.extents,num_depots,6)

        # ordering here should be outbound origin, return trip origin,
        # outbound destination (which is the same as return trip
        # origin right now), and then return trip destination.
        # Origins are grouped, and destinations are grouped.  Dumb,
        # but there you go.  This whole thing is a hack.
        lats = np.concatenate((lats_out_orig,lats_in_orig,lats_out_dest,lats_in_dest,lats_dep))
        lons = np.concatenate((lons_out_orig,lons_in_orig,lons_out_dest,lons_in_dest,lons_dep))

        # uniformly distributed integer demands.
        cust_demands_out = np.random.randint(min_demand, max_demand, n)

        # assume all trips out require a return trip with the same demand as the out trip
        cust_demands_in = np.copy(cust_demands_out)

        cust_demands = np.concatenate((cust_demands_out,cust_demands_in))

        # negative demands for deliveries
        cust_deliveries = -cust_demands
        depot_demands = np.array([0 for i in range(0,num_depots)])
        # smush together
        demands = np.concatenate((cust_demands,cust_deliveries,depot_demands))
        self.time_horizon = 24 * 60 ** 2  # A 24 hour period.

        # The customers demand min_tw to max_tw hour time window for each
        # pickup (outbound trips)
        pu_time_windows_out = np.random.random_integers(min_tw * 3600,
                                                        max_tw * 3600, n)

        # The last time a pickup window can start
        latest_time = self.time_horizon - pu_time_windows_out - 9*3600 # not sure here, [but lopping off 6 hrs seems to make things possible] make 9 to account for returns
        start_times = [None for o in range(0,2*num_trips+num_depots)]
        end_times = [None for o in range(0,2*num_trips+num_depots)]
        hard_stop = timedelta(seconds=self.time_horizon - 600)
        # Make random timedeltas, nominaly from the start of the day.
        # start with out trips
        for idx in range(0,n):
            # base time windows on destination, not origin
            deliv_idx = idx+num_trips
            stime = int(np.random.random_integers(earliest_start, latest_time[idx]))

            # time window for pickup
            start_times[idx] = timedelta(seconds=stime)
            end_times[idx] = (start_times[idx] +
                              timedelta(seconds=int(pu_time_windows_out[idx])))

            # account for time to travel by growing window as needed
            from_lat = lats[idx]
            from_lon = lons[idx]
            to_lat = lats[idx+num_trips]
            to_lon = lons[idx+num_trips]
            od_dist_out = round( spatial.haversine(from_lon,
                                             from_lat,
                                             to_lon,
                                             to_lat))
            dtime_out = self.travel_time( od_dist_out, speed_kmph=avg_speed )

            od_dist_in = round( spatial.haversine(to_lon,
                                             to_lat,
                                             from_lon,
                                             from_lat))
            dtime_in = self.travel_time( od_dist_in, speed_kmph=avg_speed )

            # time window for delivery.  should be travel time plus some error (say 25%)
            start_times[deliv_idx] = start_times[idx] + timedelta(seconds=(load_time*cust_demands[idx]+dtime_out))  # earliest pickup plus travel time
            #print('stdi',deliv_idx,str(start_times[idx]),str(timedelta(seconds=dtime_out)),str(start_times[deliv_idx]))
            end_times[deliv_idx] = end_times[idx] + timedelta(seconds=(load_time*cust_demands[idx]+1.25*float(dtime_out))) # latest pickup plus 1.25 times travel time

            # time windows can't be any later than hard_stop
#            start_times[idx] = min(hard_stop,start_times[idx] + timedelta(seconds=dtime_out)
#                                   + timedelta(seconds=(2 * load_time * cust_demands[idx])) )
#            end_times[idx] = min(hard_stop,end_times[idx] + timedelta(seconds=dtime_out)
#                                 + timedelta(seconds=(2 * load_time * cust_demands[idx])) + timedelta(seconds=1800) )

            # return trip time window will be latest delivery time (end of pickup window + dtime + "activity time"
            # "activity time" a constant for now (say 60 minutes), but should probably be a random variable
            activity_time = 1*3600
            idx_in = idx+n

            # Earliest start time of return pickup should be end_time of delivery plus activity time
            stime_in = end_times[deliv_idx] + timedelta(seconds=activity_time)
            start_times[idx_in] = stime_in

            # latest end time of return pickup should be return_pu_win minutes past the start time
            end_times[idx_in] = (start_times[idx_in] +
                                 timedelta(seconds=(return_pu_win*60)))  # assume tight time window of 5 minutes after end of activity

            deliv_idx_in = idx+n+num_trips
            start_times[deliv_idx_in] = (start_times[idx_in] +          # earliest return pickup time plus travel time
                                         timedelta(seconds=(load_time*cust_demands[idx]+dtime_in))) # travel time
            #print('stdii',deliv_idx_in,str(start_times[idx_in]),str(timedelta(seconds=dtime_in)),str(start_times[deliv_idx_in]))
            end_times[deliv_idx_in] = (end_times[idx_in] +
                                       timedelta(seconds=(load_time*cust_demands[idx]+1.25*float(dtime_in))))  # latest return pickup time plus 1.25% of travel time
        print('done generating time windows at origins, destinations')

        # A named tuple for the customer
        Customer = namedtuple("Customer", ['index',  # the index of the cust
                                           'demand',  # the demand for the cust
                                           'lat',  # the latitude of the cust
                                           'lon',  # the longitude of the cust
                                           'tw_open',  # timedelta window open
                                           'tw_close'])  # timedelta window cls

        self.customers = [Customer(idx, dem, lat, lon, tw_open, tw_close) for
                          idx, dem, lat, lon, tw_open, tw_close
                          in zip(stops, demands, lats, lons,
                                 start_times, end_times)]

        # The number of seconds needed to 'unload' 1 unit of goods.
        self.service_time_per_dem = load_time  # seconds

    def central_start_node(self, invert=False):
        """
        Return a random starting node, with probability weighted by distance
        from the centre of the extents, so that a central starting node is
        likely.

        Args:
            invert (Optional bool): When True, a peripheral starting node is
                most likely.
        Returns:
            int: a node index.

        Examples:
            >>> customers.central_start_node(invert=True)
            42
        """
        num_nodes = len(self.customers)
        dist = np.empty((num_nodes, 1))
        for idx_to in range(num_nodes):
            dist[idx_to] = spatial.haversine(self.center.lon,
                                           self.center.lat,
                                           self.customers[idx_to].lon,
                                           self.customers[idx_to].lat)
        furthest = np.max(dist)

        if invert:
            prob = dist * 1.0 / sum(dist)
        else:
            prob = (furthest - dist * 1.0) / sum(furthest - dist)
        indexes = np.array([range(num_nodes)])
        start_node = np.random.choice(indexes.flatten(),
                                      size=1,
                                      replace=True,
                                      p=prob.flatten())
        return start_node[0]


    def get_total_demand(self):
        """
        Return the total demand of all customers.
        """
        return(sum([c.demand for c in self.customers]))

    def return_dist_callback(self, **kwargs):
        """
        Return a callback function for the distance matrix.

        Args:
            **kwargs: Arbitrary keyword arguments passed on to
                make_distance_mat()

        Returns:
            function: dist_return(a,b) A function that takes the 'from' node
                index and the 'to' node index and returns the distance in km.
        """
        print('generatng distmat')
        self.distmat = spatial.make_distance_mat(self.customers,**kwargs)
        print('done with distmat')
        print(self.distmat)
        def dist_return(a, b): return(self.distmat[a][b])

        return dist_return

    def return_dem_callback(self):
        """
        Return a callback function that gives the demands.

        Returns:
            function: dem_return(a,b) A function that takes the 'from' node
                index and the 'to' node index and returns the distance in km.
        """
        def dem_return(a, b): return(self.customers[a].demand)

        return dem_return

    def zero_depot_demands(self, depot):
        """
        Zero out the demands and time windows of depot.  The Depots do not have
        demands or time windows so this function clears them.

        Args:

            depot (int): index of the cust to modify into a depot.

        Examples:

        >>> customers.zero_depot_demands(5)
        >>> customers.customers[5].demand == 0
        True


        """
        start_depot = self.customers[depot]
        self.customers[depot] = start_depot._replace(demand=0,
                                                     tw_open=None,
                                                     tw_close=None)

    def make_service_time_call_callback(self):
        """
        Return a callback function that provides the time spent servicing the
        customer.  Here is it proportional to the demand given by
        self.service_time_per_dem, default 300 seconds per unit demand.

        Returns:
            function [dem_return(a, b)]: A function that takes the from/a node
                index and the to/b node index and returns the service time at a

        """
        def service_time_return(a, b):
            service_cost = self.customers[a].demand * self.service_time_per_dem
            # print("demand: "+str(self.customers[a].demand) + "cost: "+str(service_cost))
            return(abs(service_cost))

        return service_time_return

    def travel_time (self, distance, speed_kmph=50):
        """
        Creates a callback function for transit time. Assuming an average
        speed of speed_kmph
        Args:
            speed_kmph: the average speed in km/h
        Returns:
            travel time to cover the given distance.
        """
        return (distance / (speed_kmph * 1.0 / 60 ** 2))

    def make_transit_time_callback(self, speed_kmph=50):
        """
        Creates a callback function for transit time. Assuming an average
        speed of speed_kmph
        Args:
            speed_kmph: the average speed in km/h
        Returns:
            function [tranit_time_return(a, b)]: A function that takes the
                from/a node index and the to/b node index and returns the
                tranit time from a to b.
        """
        def tranit_time_return(a, b):
            return(self.distmat[a][b] / (speed_kmph * 1.0 / 60 ** 2))

        return tranit_time_return

    def get_index_of_other_end(self,idx):
        """
        If idx is an origin, will get the corresponding destination.  If
        idx is a destination, will get the corresponding origin.
        Args:
           idx: the index of the node
        Returns:
           int: the index of the paired origin or destination of the passed
                in index.
        """
        opposite_idx = idx+self.num_trips
        if  idx >= self.num_trips:
            opposite_idx = idx - self.num_trips
        return opposite_idx

    def get_index_of_opposite_trip(self,idx):
        """
        If idx is a pickup node, will get the return pickup node idx.
        If idx is a delivery node, will get the return delivery node idx.
        Or vice versa.
        Args:
           idx: the index of the node
        Returns:
           int: the index of the opposite pickup or delivery node
        """
        opposite_idx = idx+self.num_custs
        if  idx >= self.num_custs and idx < self.num_trips:
            opposite_idx = idx - self.num_custs
        if  idx >= self.num_trips + self.num_custs:
            opposite_idx = idx - self.num_custs
        return opposite_idx
