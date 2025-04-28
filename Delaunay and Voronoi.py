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

# Returns a vector that points from point 1 to point 2
def vector(point_1, point_2):
   the_vector = [point_2[0]-point_1[0], point_2[1]-point_1[1]]
   return the_vector

def does_edge_exist(edge, triplets):
   edge_exists = False
   for triplet in range(len(triplets)):
      triplet_point_A, triplet_point_B, triplet_point_C = triplets[triplet]
      if edge[0]==triplet_point_A and edge[1]==triplet_point_B:
         edge_exists = True
      elif edge[0]==triplet_point_B and edge[1]==triplet_point_C:
         edge_exists = True
      elif edge[0]==triplet_point_C and edge[1]==triplet_point_A:
         edge_exists = True
   return edge_exists

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

# Find the smallest circumcircle through a third point C. Only search on one side of the edge AB
initialize_point_C = True
for point in range(1, num_points):
  if (point!=point_A) and (point!=point_B):
      edge_AB = np.array([points[point_B][0]-points[point_A][0], points[point_B][1]-points[point_A][1], 0])
      edge_AC = np.array([points[point][0]-points[point_A][0], points[point][1]-points[point_A][1], 0])
      point_orientation = np.cross(edge_AB, edge_AC)[2]
      if (point_orientation>0):
          if initialize_point_C:
             point_C = point
             radius_1 = find_radius(points[point_A], points[point_B], points[point_C])
             initialize_point_C = False
             print("Initial point C: ", point_C)
          else:
             radius_2 = find_radius(points[point_A], points[point_B], points[point])
             if radius_2 < radius_1:
                radius_1 = radius_2
                point_C = point
                print("Point C: ", point_C)
# Handle situations where no third point is found
if initialize_point_C:
   voronoi_boundary_edge = [0, point_A, point_B]
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
  ref_triplet_edges.append([triplet_point_C, triplet_point_B])
  ref_triplet_edges.append([triplet_point_A, triplet_point_C])
  for ref_edge in range(len(ref_triplet_edges)):
    #  Check if edge already has a circumcircle in both orientations
    if not does_edge_exist(ref_triplet_edges[ref_edge], triplets):
        point_A = ref_triplet_edges[ref_edge][0]
        point_B = ref_triplet_edges[ref_edge][1]
        # Find the smallest circumcircle through a third point C. Only search on one side of the edge AB
        initialize_point_C = True
        for point in range(1, num_points):
          if (point!=point_A) and (point!=point_B):
              edge_AB = np.array([points[point_B][0]-points[point_A][0], points[point_B][1]-points[point_A][1], 0])
              edge_AC = np.array([points[point][0]-points[point_A][0], points[point][1]-points[point_A][1], 0])
              point_orientation = np.cross(edge_AB, edge_AC)[2]
              if (point_orientation>0):
                  if initialize_point_C:
                    point_C = point
                    radius_1 = find_radius(points[point_A], points[point_B], points[point_C])
                    initialize_point_C = False
                  else:
                    radius_2 = find_radius(points[point_A], points[point_B], points[point])
                    if radius_2 < radius_1:
                        radius_1 = radius_2
                        point_C = point
        # Handle situations where no third point is found
        if initialize_point_C:
          # voronoi_boundary_edge = [active_triplets[active_triplet], point_A, point_B]
          1
        else:
          triplets.append([point_A, point_B, point_C])
          active_triplets.append(len(triplets)-1)
          edges.append([point_A, point_B])
          edges.append([point_B, point_C])
          edges.append([point_C, point_A])
          # voronoi_center_edges.append([active_triplets[active_triplet], active_triplets[-1]])
  active_triplets.pop(0)

xValues, yValues = zip(*points)

fig, ax = plt.subplots()
plt.plot(xValues, yValues, 'o')
for i in range(num_points):
    plt.text(xValues[i] + 0.2, yValues[i] + 0.2, str(i), fontsize=10)

print("triplets: ", triplets)
print("edges: ", edges)
# Plot circumcircles
for triplet in range(len(triplets)):
    circle_center = find_center(points[triplets[triplet][0]], points[triplets[triplet][1]], points[triplets[triplet][2]])
    circle_radius = find_radius(points[triplets[triplet][0]], points[triplets[triplet][1]], points[triplets[triplet][2]])
    circle = patches.Circle((circle_center[0], circle_center[1]), radius=circle_radius, edgecolor='blue', facecolor='none', linewidth=0.5)
    ax.add_patch(circle)

# Plot edges
for edge in range(len(edges)):
   start_point = points[edges[edge][0]]
   end_point = points[edges[edge][1]]
   plt.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], 'k:')

plt.xlim(0, max_value)  # x-axis will range from 0 to 5
plt.ylim(0, max_value)  # y-axis will range from 0 to 20
plt.gca().set_aspect('equal')
plt.show()

