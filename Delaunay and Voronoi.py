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

# Find the point that creates the smallest circumcircle on one side of the edge between point A and B


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

