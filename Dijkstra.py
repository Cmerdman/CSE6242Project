
import heapq
import pandas as pd
from math import radians, sin, cos, sqrt, atan2
import os

#Input: 
#Road_Data: A dataframe of the roads of New York City, with the following columns: 
#     See Data.py for how this is generated.
#  
#Route_Select: List of Strings or ints.  Should be 1/"Least Traffic", 2/"Safest", and/or 3/"Fastest"  This will tell the algorithm whether to use the shortest path, the shortest path with least accidents, or the fastest path
#  Default value: The fastest path.
#  Note: Shortest path is in terms of distance.  Fastest is in terms of time.
#  Note: We should make this a checkbox.  I.e. If the user wants the shortest and fastest path, both parameters will be used when calculating weight.

#Gets the Distance from a start node to ALL nodes in the graph
#This may be used in the visualization of existing taxi trips
def AllDijkstra(start, List_Selected_Parameters):

    """
    Calculates the shortest path from a start node to all other nodes in a graph.

    Args:
        start: The id of the starting node.
        List of 

    Returns:
        A list containing the shortest distance from the start node to each node.
        If a node is unreachable, its distance will be infinity.
    """

    #Read Edges for Graph
    Road_Data = pd.read_csv('Debug/Edges.csv')
    #Convert data to source, target, weight (weight will be length in time), Traffic Modifier, Accident Modifier
    #NOTE: The One Way boolean isn't needed, as all edges are directed (source, target also exists as target, source for two way streets)
    Road_Data = Road_Data.drop(['Length', 'OneWay', 'edgeID', 'RoadName', 'SpeedLimit'], axis=1)

    Num_Nodes = 54128 #Number of nodes in our graph.  Static.  Hard-coding saves time over reading all the nodes every time
    #Determine which weight modifiers to use
    #Least Traffic Route
    if ("Least Traffic" in List_Selected_Parameters):
        TrafficWeight = 1
    else:
        TrafficWeight = 0
    #Safest Route
    if ("Safest" in List_Selected_Parameters):
        SafeWeight = 1
    else: 
        SafeWeight = 0
    #Fastest Route 
    if ("Fastest" in List_Selected_Parameters):
        FastWeight = 1
    #Set to default if neither the shortest or safest paths are picked
    elif(TrafficWeight == 0 and SafeWeight == 0):
        FastWeight = 1
    else: 
        FastWeight = 0


    adj_list = [[] for _ in range(Num_Nodes)]
    for source, dest, weight, traffic, accident in Road_Data:
        adj_list[source].append((dest, weight, traffic, accident))

    distances = [float('inf')] * Num_Nodes
    distances[start] = 0
    priority_queue = [(0, start)]  # (distance, node)

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)

        if current_distance > distances[current_node]:
            continue

        for neighbor, weight, traffic, accident in adj_list[current_node]:
            #Note on combined_weight: This takes into account the impact that accidents and traffic may have on the travel time, but still prioritizes by the fastest travel time.
            #I.e. If a street has twice the traffic of another but is a fourth of the travel time without traffic, it will be chosen as 2x traffic * 1/4 travel time = 1/2 total weight 
            combined_weight = (weight * traffic* TrafficWeight) + (weight * SafeWeight * accident) + (weight * FastWeight)
            new_distance = distances[current_node] + combined_weight
            if new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                heapq.heappush(priority_queue, (new_distance, neighbor))

    return distances

#Gets the Distance from a start node to a specified end node in the graph
#This may be used in the visualization of a user-defined taxi trip
def Dijkstra(start, end, List_Selected_Parameters):
    """
    Finds the shortest path from start to end node in a graph using Dijkstra's algorithm.

    Args:
        edges: A list of tuples representing edges (u, v, weight).
        start: The starting node.
        end: The target ending node.

    Returns:
        The shortest distance from start to end, or float('inf') if no path exists.
    """
    #Read Edges for Graph
    Road_Data = pd.read_csv('Debug/Edges.csv')
    #Convert data to source, target, weight (weight will be length in time), Traffic Modifier, Accident Modifier
    #NOTE: The One Way boolean isn't needed, as all edges are directed (source, target also exists as target, source for two way streets)
    Road_Data = Road_Data.drop(['Length', 'OneWay', 'edgeID', 'RoadName', 'SpeedLimit'], axis=1)

    #Determine which weight modifiers to use
    #Least Traffic Route
    if ("Least Traffic" in List_Selected_Parameters):
        TrafficWeight = 1
    else:
        TrafficWeight = 0
    #Safest Route
    if ("Safest" in List_Selected_Parameters):
        SafeWeight = 1
    else: 
        SafeWeight = 0
    #Fastest Route 
    if ("Fastest" in List_Selected_Parameters):
        FastWeight = 1
    #Set to default if neither the shortest or safest paths are picked
    elif(TrafficWeight == 0 and SafeWeight == 0):
        FastWeight = 1
    else: 
        FastWeight = 0

    graph = {}
    for u, v, weight, traffic, accident in Road_Data:
        graph.setdefault(u, []).append((v, weight, traffic, accident))
        graph.setdefault(v, []).append((u, weight, traffic, accident))

    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    priority_queue = [(0, start)]  # (distance, node)

    while priority_queue:
        dist, current_node = heapq.heappop(priority_queue)

        if dist > distances[current_node]:
            continue

        if current_node == end:
            return dist

        for neighbor, weight in graph.get(current_node, []):
            #Note on combined_weight: This takes into account the impact that accidents and traffic may have on the travel time, but still prioritizes by the fastest travel time.
            #I.e. If a street has twice the traffic of another but is a fourth of the travel time without traffic, it will be chosen as 2x traffic * 1/4 travel time = 1/2 total weight 
            combined_weight = (weight * traffic* TrafficWeight) + (weight * SafeWeight * accident) + (weight * FastWeight)
            new_distance = distances[current_node] + combined_weight
            if new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                heapq.heappush(priority_queue, (new_distance, neighbor))

    return float('inf') # No path found


#Determines the node ID closest to start and end point
def GetStartAndEnd(start, end):
    #All Nodes:
    nodes_df = pd.read_csv('Debug/Nodes.csv')

    startoffsets = nodes_df.copy()
    startoffsets['StartLat'] = start[0]
    startoffsets['StartLong'] = start[1]

    endoffsets = nodes_df.copy()
    endoffsets['EndLat'] = end[0]
    endoffsets['EndLong'] = end[1]

    #Distance difference in longitude and latitude from start and end points using haversine formula
    #NOTE: The unit used for distance doesn't matter here since we're just using this to get the shortest distance offset
    startoffsets["Offset"] = startoffsets.apply(lambda row: haversine(row['StartLat'], row['StartLong'], row['Lat'], row['Long']), axis=1)
    endoffsets["Offset"] = endoffsets.apply(lambda row: haversine(row['EndLat'], row['EndLong'], row['Lat'], row['Long']), axis=1)
    
    #Obtain the road ID that represents the node with the smallest offset
    startmin = startoffsets["Offset"].min()
    startnodeID = startoffsets[startoffsets["Offset"] == startmin]
    startnodeID =  startnodeID['RoadID'].to_numpy()[0]
    
    #Obtain the road ID that represents the node with the smallest offset
    endmin = endoffsets["Offset"].min()
    endnodeID = endoffsets[endoffsets["Offset"] == endmin]
    endnodeID = endnodeID["RoadID"].to_numpy()[0]

    return startnodeID, endnodeID

#Haversine formula for calculating distance between two lat long pairs
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of the Earth in kilometers

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance


def main(): 
    
    #TODO: Determine how to use Dijkstra's Algorithm for all nodes, do we need to?  Or do we use the specified start and end for each trip in our database?

    #TODO: We need user input
    start = (40.655273, -73.931772) #User defined starting point (lat long)
    end = (41, -73.4) #User defined ending point (lat long)
    User_Parameters = ["Fastest"] #User defined
    #Determine starting and ending nodes 
    startid, endid = GetStartAndEnd(start, end)

    #Dijkstra(startid, endid, User_Parameters, Spare_Time)
    

if __name__ == "__main__":
    main()