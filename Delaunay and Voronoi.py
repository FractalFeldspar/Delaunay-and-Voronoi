import matplotlib.pyplot as plt
import numpy as np
from numpy import random
import math

num_points = 20
points = []
for i in range(num_points):
  newPoint = [random.uniform(0,100), random.uniform(0,100)]
  points.append(newPoint)

# print(points)

# Find the point nearest to point P0
distance = math.sqrt((points[1][0]-points[0][0])**2 + (points[1][1]-points[0][1])**2)
nearest_point = 1
for point in range(2, num_points):
  new_distance = math.sqrt((points[point][0]-points[0][0])**2 + (points[point][1]-points[0][1])**2)
  if new_distance < distance:
      distance = new_distance
      nearest_point = point
point_A = 0
point_B = nearest_point
print(point_A)
print(point_B)
print(distance)

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

# Find the circumcircle through a third point C that does not enclose any other points. Only search on one side of the edge AB
point_C = 1
for point in range(1, num_points):
  if (point!=point_A) and (point!=point_B):
      edge_AB = np.array([points[point_B][0]-points[point_A][0], points[point_B][1]-points[point_A][1], 0])
      edge_AC = np.array([points[point][0]-points[point_A][0], points[point][1]-points[point_A][1], 0])
      point_orientation = np.cross(edge_AB, edge_AC)
      if (point_orientation<0):
          adx = points[point_A][0]-points[point][0]
          bdx = points[point_B][0]-points[point][0]
          cdx = points[point_C][0]-points[point][0]
          ady = points[point_A][1]-points[point][1]
          bdy = points[point_B][1]-points[point][1]
          cdy = points[point_C][1]-points[point][1]          
          matrix = np.array([[adx, ady, adx**2+ady**2],
                            [bdx, bdy, bdx**2+bdy**2],
                            [cdx, cdy, cdx**2+cdy**2]])
          det = np.linalg.det(matrix)
          if det<0:
             point_C = point


triplets = [] # points are stored as indices
active_triplets = []

xValues, yValues = zip(*points)
# plt.plot(xValues, yValues, 'o')
# plt.show()

# plt.figure(figsize=(6,6))
# plt.scatter(xValues, yValues, color='blue', label='Random Points')
plt.plot(xValues, yValues, 'o')
for i in range(num_points):
    plt.text(xValues[i] + 0.2, yValues[i] + 0.2, str(i), fontsize=10)
plt.gca().set_aspect('equal')
plt.show()

