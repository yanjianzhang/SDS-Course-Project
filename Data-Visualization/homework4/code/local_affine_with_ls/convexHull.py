from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import numpy as np
import math

def dot(v,w):
    x,y = v
    X,Y = w
    return x*X + y*Y

def length(v):
    x,y = v
    return math.sqrt(x*x + y*y)

def vector(b,e):
    x,y = b
    X,Y = e
    return (X-x, Y-y)

def unit(v):
    x,y = v
    mag = length(v)
    return (x/mag, y/mag)

def distance(p0,p1):
    return length(vector(p0,p1))

def scale(v,sc):
    x,y = v
    return (x * sc, y * sc)

def add(v,w):
    x,y = v
    X,Y = w
    return (x+X, y+Y)


# Given a line with coordinates 'start' and 'end' and the
# coordinates of a point 'pnt' the proc returns the shortest
# distance from pnt to the line and the coordinates of the
# nearest point on the line.
#
# 1  Convert the line segment to a vector ('line_vec').
# 2  Create a vector connecting start to pnt ('pnt_vec').
# 3  Find the length of the line vector ('line_len').
# 4  Convert line_vec to a unit vector ('line_unitvec').
# 5  Scale pnt_vec by line_len ('pnt_vec_scaled').
# 6  Get the dot product of line_unitvec and pnt_vec_scaled ('t').
# 7  Ensure t is in the range 0 to 1.
# 8  Use t to get the nearest location on the line to the end
#    of vector pnt_vec_scaled ('nearest').
# 9  Calculate the distance from nearest to pnt_vec_scaled.
# 10 Translate nearest back to the start/end line.
# Malcolm Kesson 16 Dec 2012

def pnt2line(pnt, start, end):
    line_vec = vector(start, end)
    pnt_vec = vector(start, pnt)
    line_len = length(line_vec)
    line_unitvec = unit(line_vec)
    pnt_vec_scaled = scale(pnt_vec, 1.0/line_len)
    t = dot(line_unitvec, pnt_vec_scaled)
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    nearest = scale(line_vec, t)
    dist = distance(nearest, pnt_vec)
    nearest = add(nearest, start)
    return (dist, nearest)

def point_in_poly(x,y,poly):

    n = len(poly)
    inside = False

    p1x,p1y = poly[0]
    for i in range(n+1):
        p2x,p2y = poly[i % n]
        if y > min(p1y,p2y):
            if y <= max(p1y,p2y):
                if x <= max(p1x,p2x):
                    if p1y != p2y:
                        xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x,p1y = p2x,p2y

    return inside


def convexHull(points,newpoints):
    # Original points, hull and test points
    ##points = np.random.rand(30, 2)   # 30 random points in 2-D
    hull = ConvexHull(points)
    ##newpoints = np.random.rand(30, 2)   # 30 random points in 2-D


    pt_dist = []
    for p_idx in range(len(newpoints)):
        ##print(p_idx)
        pt = newpoints[p_idx,:]
        ## print(pt)
        dist_list = []
        for v_idx in range(len(hull.vertices)):
            v1 = hull.vertices[v_idx - 1]
            v2 = hull.vertices[v_idx]
            start = points[v1]
            end = points[v2]
            temp = pnt2line(pt, start, end)
            dist_list.append(temp[0])

        #Check point is within polygon
        inside =  point_in_poly(pt[0],pt[1],points[hull.vertices])
        if (inside == True):
            dist_temp = -1. * min(dist_list)
        else:
            dist_temp = min(dist_list)
        ##print(dist_temp)
        pt_dist.append(dist_temp)


    # Plot original points, hull and new points
    # plt.plot(points[:,0], points[:,1], 'ro')
    # plt.plot(points[hull.vertices,0], points[hull.vertices,1], 'r--', lw=2)
    # plt.plot(newpoints[:,0], newpoints[:,1], 'go')

    # for p_idx in range(len(newpoints)):
    #     pt = newpoints[p_idx,:]
    #     pt[1] = pt[1] + 0.01
    #     dist = pt_dist[p_idx]
    #     distLabel = "%.2f" % dist
    #     plt.annotate(distLabel,xy=pt)
    # # plt.show()
    #print(pt_dist)
    return pt_dist
    # for p_idx in range(30):
    #     pt = newpoints[p_idx,:]
    #     pt[1] = pt[1] + 0.01
    #     dist = pt_dist[p_idx]
    #     distLabel = "%.2f" % dist
    #     plt.annotate(distLabel,xy=pt)
    # plt.show()

if __name__ == '__main__':
    points = np.random.rand(30, 2)   # 30 random points in 2-D
    newpoints = np.random.rand(30, 2)   # 30 random points in 2-D
    convexHull(points, newpoints)
##https://stackoverflow.com/questions/23937076/distance-to-convexhull