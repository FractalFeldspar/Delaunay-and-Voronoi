import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from numpy import random
import math

def distance(point_1, point_2):
   return math.sqrt((point_1[0]-point_2[0])**2 + (point_1[1]-point_2[1])**2)

def find_center(point_1, point_2, point_3):
   a1 = 2*(point_2[0]-point_1[0])
   a2 = 2*(point_3[0]-point_1[0])
   b1 = 2*(point_2[1]-point_1[1])
   b2 = 2*(point_3[1]-point_1[1])
   c1 = (point_2[0])**2+(point_2[1])**2-(point_1[0])**2-(point_1[1])**2
   c2 = (point_3[0])**2+(point_3[1])**2-(point_1[0])**2-(point_1[1])**2
   xc = (c1*b2-c2*b1)/(a1*b2-a2*b1)
   yc = (a1*c2-a2*c1)/(a1*b2-a2*b1)
   center = [xc, yc]
   return center

def find_radius(point_1, point_2, point_3):
   return distance(point_1, find_center(point_1, point_2, point_3))

# def is_point_inside_circumcircle(point_1, point_2, point_3, point_4):
#     is_point_inside = False
#     adx = point_1[0]-point_4[0]
#     bdx = point_2[0]-point_4[0]
#     cdx = point_3[0]-point_4[0]
#     ady = point_1[1]-point_4[1]
#     bdy = point_2[1]-point_4[1]
#     cdy = point_3[1]-point_4[1]        
#     matrix = np.array([[adx, ady, adx**2+ady**2],
#                       [bdx, bdy, bdx**2+bdy**2],
#                       [cdx, cdy, cdx**2+cdy**2]])
#     det = np.linalg.det(matrix)
#     if det>0:
#       is_point_inside = True
#     return is_point_inside

def is_point_inside_circumcircle(point_1, point_2, point_3, point_4):
    is_point_inside = False
    center = find_center(point_1, point_2, point_3)
    radius = distance(point_1, center)
    distance_to_point_4 = distance(center, point_4)
    if distance_to_point_4<radius:
      is_point_inside = True
    return is_point_inside

# Returns a vector that points from point 1 to point 2
def vector(point_1, point_2):
   the_vector = [point_2[0]-point_1[0], point_2[1]-point_1[1]]
   return the_vector

def find_matching_edge(edge, triplets):
   triplet_index = None
   for triplet in range(len(triplets)):
      edge_exists = False
      triplet_point_A, triplet_point_B, triplet_point_C = triplets[triplet]
      if edge[0]==triplet_point_A and edge[1]==triplet_point_B:
         edge_exists = True
      elif edge[0]==triplet_point_B and edge[1]==triplet_point_C:
         edge_exists = True
      elif edge[0]==triplet_point_C and edge[1]==triplet_point_A:
         edge_exists = True
      if edge_exists:
           triplet_index = triplet
   return triplet_index

# def find_duplicate_edge(edge, edges):
#      duplicate_index = None
#      for edge_index in range(len(edges)):
#           if edges[edge_index][0]==edge[0] and edges[edge_index][1]==edge[1]:
#                duplicate_index = edge_index
#           elif edges[edge_index][1]==edge[0] and edges[edge_index][0]==edge[1]:
#                duplicate_index = edge_index
#      return duplicate_index

def find_duplicate_edge(edge, edges):
    edge_set = set(edge)
    for i, e in enumerate(edges):
        if set(e) == edge_set:
            return i
    return None

num_points = 20
max_value = 100
points = []
edges = []
triplets = []
active_triplets = []
voronoi_center_edges = []
voronoi_boundary_edges = []
for i in range(num_points):
  newPoint = [random.uniform(0,max_value), random.uniform(0,max_value)]
  points.append(newPoint)

# Find the point nearest to point P0
distance_1 = distance(points[0], points[1])
nearest_point = 1
for point in range(2, num_points):
  distance_2 = distance(points[0], points[point])
  if distance_2 < distance_1:
      distance_1 = distance_2
      nearest_point = point
point_A = 0
point_B = nearest_point
print("Point A:", point_A)
print("Point B:", point_B)
print("Distance: ", distance_1)

# Find a circumcircle through a third point C that does not contain any other points
initialize_point_C = True
for point in range(1, num_points):
  if (point!=point_A) and (point!=point_B):
			edge_AB = np.array([points[point_B][0]-points[point_A][0], points[point_B][1]-points[point_A][1], 0])
			edge_AC = np.array([points[point][0]-points[point_A][0], points[point][1]-points[point_A][1], 0])
			point_orientation = np.cross(edge_AB, edge_AC)[2]
			if point_orientation>0:
					if initialize_point_C:
							point_C = point
							initialize_point_C = False
							print("Initial point C: ", point_C)
					else:
							update_point_C = is_point_inside_circumcircle(points[point_A], points[point_B], points[point_C], points[point])
							if update_point_C:
									point_C = point
									print("Point C: ", point_C)
# Handle situations where no third point is found
if initialize_point_C:
   voronoi_boundary_edge = [0, point_A, point_B]
   voronoi_boundary_edges.append(voronoi_boundary_edge)
else:
   triplets.append([point_A, point_B, point_C])
   active_triplets.append(len(triplets)-1)
   edges.append([point_A, point_B])
   edges.append([point_B, point_C])
   edges.append([point_C, point_A])


print("triplets: ", triplets)
print("active triplets: ", active_triplets)
while len(active_triplets) > 0:
  active_triplet = active_triplets[0]
  triplet_point_A, triplet_point_B, triplet_point_C = triplets[active_triplet]
  ref_triplet_edges = []
  if active_triplet==0:
     ref_triplet_edges.append([triplet_point_B, triplet_point_A])
  # ref_triplet_edges.append([triplet_point_B, triplet_point_A])
  ref_triplet_edges.append([triplet_point_C, triplet_point_B])
  ref_triplet_edges.append([triplet_point_A, triplet_point_C])
  for ref_edge in range(len(ref_triplet_edges)):
    #  Check if edge already has a circumcircle in both orientations
    matching_edge_triplet_index = find_matching_edge(ref_triplet_edges[ref_edge], triplets)
    if matching_edge_triplet_index is not None:
        if find_duplicate_edge([matching_edge_triplet_index, active_triplet], voronoi_center_edges) is None:
             voronoi_center_edges.append([matching_edge_triplet_index, active_triplet])
    else:
        point_A = ref_triplet_edges[ref_edge][0]
        point_B = ref_triplet_edges[ref_edge][1]
        # Find a circumcircle through a third point C that does not contain any other points
        initialize_point_C = True
        for point in range(1, num_points):
          if (point!=point_A) and (point!=point_B):
              edge_AB = np.array([points[point_B][0]-points[point_A][0], points[point_B][1]-points[point_A][1], 0])
              edge_AC = np.array([points[point][0]-points[point_A][0], points[point][1]-points[point_A][1], 0])
              point_orientation = np.cross(edge_AB, edge_AC)[2]
              if (point_orientation>0):
                  if initialize_point_C:
                    point_C = point
                    initialize_point_C = False
                  else:
                      update_point_C = is_point_inside_circumcircle(points[point_A], points[point_B], points[point_C], points[point])
                      if update_point_C:
                          point_C = point
        # Handle situations where no third point is found
        if initialize_point_C:
          voronoi_boundary_edge = [active_triplet, point_A, point_B]
          voronoi_boundary_edges.append(voronoi_boundary_edge)
        else:
          triplets.append([point_A, point_B, point_C])
          active_triplets.append(len(triplets)-1)
          if find_duplicate_edge([point_A, point_B], edges) == None:
               edges.append([point_A, point_B])
          if find_duplicate_edge([point_B, point_C], edges) == None:
               edges.append([point_B, point_C])
          if find_duplicate_edge([point_C, point_A], edges) == None:
               edges.append([point_C, point_A])
          voronoi_center_edges.append([active_triplet, active_triplets[-1]])
  active_triplets.pop(0)

# print("Voronoi boundary edges: ", voronoi_boundary_edges)
# print("Voronoi center edges: ", voronoi_center_edges)

xValues, yValues = zip(*points)

fig, ax = plt.subplots()
plt.plot(xValues, yValues, 'o')
for i in range(num_points):
    plt.text(xValues[i] + 0.2, yValues[i] + 0.2, str(i), fontsize=10)

print("triplets: ", triplets)
# Plot circumcircles
for triplet in range(len(triplets)):
    circle_center = find_center(points[triplets[triplet][0]], points[triplets[triplet][1]], points[triplets[triplet][2]])
    circle_radius = find_radius(points[triplets[triplet][0]], points[triplets[triplet][1]], points[triplets[triplet][2]])
    circle = patches.Circle((circle_center[0], circle_center[1]), radius=circle_radius, edgecolor='blue', facecolor='none', linewidth=0.5)
    ax.add_patch(circle)

# Plot Delaunay edges
print("edges: ", edges)
print("Number of edges: ", len(edges))
for edge in range(len(edges)):
   start_point = points[edges[edge][0]]
   end_point = points[edges[edge][1]]
   plt.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], 'k:')

# Plot Voronoi center edges
print("Voronoi center edges: ", voronoi_center_edges)
print("Number of center edges: ", len(voronoi_center_edges))
for center_edge in range(len(voronoi_center_edges)):
    triplet_index_A = voronoi_center_edges[center_edge][0]
    triplet_index_B = voronoi_center_edges[center_edge][1]
    triplet_center_A = find_center(points[triplets[triplet_index_A][0]], points[triplets[triplet_index_A][1]], points[triplets[triplet_index_A][2]])
    triplet_center_B = find_center(points[triplets[triplet_index_B][0]], points[triplets[triplet_index_B][1]], points[triplets[triplet_index_B][2]])
    plt.plot([triplet_center_A[0], triplet_center_B[0]], [triplet_center_A[1], triplet_center_B[1]], 'r')

# Plot Voronoi boundary edges
print("Voronoi boundary edges: ", voronoi_boundary_edges)
print("Number of boundary edges: ", len(voronoi_boundary_edges))
if len(triplets)>0:
    for boundary_edge in range(len(voronoi_boundary_edges)):
        triplet_index = voronoi_boundary_edges[boundary_edge][0]
        triplet_center = find_center(points[triplets[triplet_index][0]], points[triplets[triplet_index][1]], points[triplets[triplet_index][2]])
        point_index_A = voronoi_boundary_edges[boundary_edge][1]
        point_index_B = voronoi_boundary_edges[boundary_edge][2]
        vector_AB = [points[point_index_B][0]-points[point_index_A][0], points[point_index_B][1]-points[point_index_A][1]]
        outward_vector = [-1*vector_AB[1], vector_AB[0]]
        outward_vector_mag = math.sqrt((outward_vector[0])**2+(outward_vector[1])**2)
        outward_vector = [100/outward_vector_mag*element for element in outward_vector]
        # min_x = min(xValues)
        # max_x = max(xValues)
        # min_y = min(yValues)
        # max_y = max(yValues)
        # ave_x = 0.5*(min_x+max_x)
        # ave_y = 0.5*(min_y+max_y)
        # outward_ref_dir = np.array([triplet_center[0]-ave_x, triplet_center[1]-ave_y])
        # print("dot product: ", np.dot(outward_ref_dir, np.array(outward_vector)))
        # if np.dot(outward_ref_dir, np.array(outward_vector))<0:
        #      outward_vector = [-1*element for element in outward_vector]
        outward_point = [triplet_center[0]+outward_vector[0], triplet_center[1]+outward_vector[1]]
        plt.plot([triplet_center[0], outward_point[0]], [triplet_center[1], outward_point[1]], 'r')
else:
     print("Unable to begin Delaunay triangulation. Please re-run the code or generate more points.")

extra_border = 15
plt.xlim(0-extra_border, max_value+extra_border)
plt.ylim(0-extra_border, max_value+extra_border)
plt.gca().set_aspect('equal')
plt.show()

