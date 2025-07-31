import numpy as np
# import random as rd
import matplotlib.pyplot as plt
import matplotlib.colors as clr
from numdifftools import Derivative as dv
from numdifftools import directionaldiff as ddv
from scipy import integrate as inte
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.integrate import dblquad
# import calfem.geometry as cfg
# import calfem.mesh as cfm
# import calfem.vis_mpl as cfv
# from scipy.spatial import ConvexHull
# import matplotlib.tri as mtri
# from matplotlib.path import Path
# from numba import jit


def centroid(A, B, C):
    return (A + B + C) / 3

def triangleArea(A, B, C):
    return 0.5 * norm(np.cross(B - A, C - A))

def dist(x,y):
    return np.sqrt( (y[0] - x[0])**2 + (y[1] - x[1])**2 + (y[2] - x[2])**2 )

def norm(x):
    return dist(x,[0,0,0])

def mid(p, q):
    p = np.array(p)
    q = np.array(q)
    return (p + q) / 2

def pushToSphere(p,x,r):
    p = np.array(p)
    x = np.array(x)
    a = p - x
    a = r * a / norm(a)
    return x + a

def splitTriangle(triangle):
    a,b,c = triangle
    ab = mid(a,b)
    ac = mid(a,c)
    bc = mid(b,c)

    return [
        [a, ab, ac],
        [ab, b, bc],
        [c, ac, bc],
        [ac, ab, bc]
    ]

def splitSphereTriangle(triangle, x, r):
    a,b,c = triangle
    a = pushToSphere(a,x,r)
    b = pushToSphere(b,x,r)
    c = pushToSphere(c,x,r)
    ab = pushToSphere(mid(a,b), x, r)
    bc = pushToSphere(mid(b,c), x, r)
    ac = pushToSphere(mid(a,c), x, r)

    return [
        [a, ab, ac],
        [ab, b, bc],
        [c, ac, bc],
        [ac, ab, bc]
    ]

def refineTriangle(triangles):
    refined = []
    for triangle, boundCond in triangles:
        split = splitTriangle(triangle)
        for t in split:
            refined.append([t, boundCond])
    return refined

def refineSphereTriangle(triangles, x, r):
    refined = []
    for triangle, boundCond in triangles:
        split = splitSphereTriangle(triangle, x, r)
        for t in split:
            refined.append([t, boundCond])
    return refined

def refine(mesh, n=5):
    m = mesh
    for i in range(n):
        m = refineTriangle(m)
    return m

def refineSphere(mesh, x, r, n=5):
    m = []
    for (a,b,c), boundCond in mesh:
        m.append([[pushToSphere(a, x, r), pushToSphere(b,x,r), pushToSphere(c,x,r)], boundCond])
    for i in range(n):
        m = refineSphereTriangle(m, x, r)
    return m

def fixBound(mesh, f):
    meshWithB = []
    for (x,y,z), bound in mesh:
        x = np.array(x)
        y = np.array(y)
        z = np.array(z)
        p = (x + y + z) / 3
        meshWithB.append([[x,y,z], f(p)])
    return meshWithB

def integrate_over_polygon(points, f, n=2):
    m = refine([[points, 0]], n)
    p1 = points[0]
    p2 = points[1]
    p3 = points[2]
    v1 = p2 - p1
    v2 = p3 - p1
    # normal = [v1[1] * v2[2] - v1[2] * v2[1], v1[2] * v2[0] - v1[0] * v2[2], v1[0] * v2[1] - v1[1] * v2[0]]
    normal = np.cross(v1,v2)
    area = 0.5 * norm(normal)
    integral = 0
    for (x,y,z), bound in m:
        point = (x + y + z) / 3
        integral += f(point)
    return area * integral / (4**n)

# def integrate_over_polygon(points, f, n=3):
#     x1 = points[0]
#     x2 = points[1]
#     x3 = points[2]

#     area = triangleArea(x1,x2,x3)

#     integral = 0
#     integral += inte.dblquad(
#         lambda y, x: f( (x2 - x1)*x + (x3 - x1)*y + x1 ),
#         0, 1,
#         lambda g: 0,
#         lambda x: 1 - x
#     )[0]
#     return 2 * area * integral

def singularIntegral(x1, x2, x3):
    x1 = np.array(x1)
    x2 = np.array(x2)
    x3 = np.array(x3)

    d = centroid(x1,x2,x3)

    triangs = [[d,x2,x3],[d,x1,x2],[d,x3,x1]]

    integral = 0
    for (a,b,c) in triangs:
        B = np.dot(c - a, b - a) / (norm(b - a)**2)
        C = norm(c-a)**2 / norm(b-a)**2
        integral += 2 * triangleArea(a,b,c) * inte.quad(lambda x: 1 / ((np.cos(x) + np.sin(x)) * np.sqrt(np.cos(x)**2 + B * np.sin(2*x) + C * np.sin(x)**2) ), 0, np.pi / 2)[0] / norm(b-a)
    return integral / (4 * np.pi)

def U(i,x):
    return 1 / (4 * np.pi * dist(i,x))

def Q(i,x,n):
    return 1 / (4 * np.pi) * np.dot(n / norm(n), np.array([i[0] - x[0], i[1] - x[1], i[2] - x[2]]) / (dist(i,x))**3)

def dQ(i,x,n):
    n = np.array(n)
    r = np.array(x) - np.array(i)
    nrm = norm(r)
    rdn = np.dot(r,n)
    temp1 = n / nrm**3
    temp2 = 3*r*rdn / nrm**5
    return -(temp1 - temp2) / (4 * np.pi)

def constant3D(triangs, boundCond, innP, center, radius, definition=20):
    N = len(triangs)
    triangs = np.array(triangs)

    nodes = []
    for (x,y,z) in triangs:
        nodes.append((x + y + z) / 3)

    H = np.zeros((N,N))
    G = np.zeros((N,N))

    for i in range(N):
        print(f'{i} out of {N}')
        for j in range(N):
            if (i != j):
                H[i,j] += integrate_over_polygon(triangs[j], lambda x: Q(nodes[i], x, -1 * np.cross(triangs[j][0] - triangs[j][1], triangs[j][2] - triangs[j][1])))
                # print(f'H[{i}, {j}]')
                G[i,j] += integrate_over_polygon(triangs[j], lambda x: U(nodes[i], x))
                # print(f'G[{i}, {j}]')
            else:
                H[i,j] += 1/2
                # print(f'H[{i}, {j}]')
                G[i,j] += singularIntegral(triangs[j][0], triangs[j][1], triangs[j][2])
                # print(f'G[{i}, {j}]')

    A = np.zeros((N,N))
    b = np.zeros(N)

    for i in range(N):
        for j in range(N):
            if (boundCond[j][0] == 0):
                b[i] -= H[i,j] * boundCond[j][1]
                A[i,j] -= G[i,j]
            if (boundCond[j][0] == 1):
                b[i] += G[i,j] * boundCond[j][1]
                A[i,j] += H[i,j]
            if (boundCond[j][0] == 2):
                A[i,j] += H[i,j] + ( boundCond[j][1][0] / boundCond[j][1][1] ) * G[i,j]
                b[i] += boundCond[j][1][2] * G[i,j] / boundCond[j][1][1]

    a = np.linalg.solve(A,b)

    # print(f'Ax = b: {A}{a.T} = {b}')

    u = np.zeros(N)
    q = np.zeros(N)

    for i in range(N):
        if (boundCond[i][0] == 0):
            u[i] += boundCond[i][1]
            q[i] += a[i]
        if (boundCond[i][0] == 1):
            u[i] += a[i]
            q[i] += boundCond[i][1]
        if (boundCond[i][0] == 2):
            u[i] += a[i]
            q[i] += ( boundCond[i][1][2] - boundCond[i][1][0] * u[i] ) / boundCond[i][1][1]

    def makeVecG(i):
        temp = np.zeros(N)
        for j in range(N):
            temp[j] += integrate_over_polygon(triangs[j], lambda x: U(i, x))
        return temp

    def makeVecH(i):
        temp = np.zeros(N)
        for j in range(N):
            temp[j] += integrate_over_polygon(triangs[j], lambda x: Q(i, x, -1 * np.cross(triangs[j][0] - triangs[j][1], triangs[j][2] - triangs[j][1])))
        return temp

    def sol(x):
        H = makeVecH(x)
        G = makeVecG(x)
        return np.dot(G,q) - np.dot(H,u)

    # print(f"u: {u}\nq: {q}")

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')

    # class Triangle3D:
    #     def __init__(self, x, y, z, color='m', label=None):
    #         self.x = x
    #         self.y = y
    #         self.z = z
    #         self.color = color
    #         self.label = label
    #     def plot(self, ax):
    #             ax.plot(self.x, self.y, self.z, color=self.color, label=self.label)

    # for (a,b,c) in triangs:

    #     x_val = [a[0],b[0],c[0],a[0]]
    #     y_vals = [a[1], b[1], c[1]]
    #     z_vals = [a[2], b[2], c[2]]
    #     y = np.append(y_vals, y_vals[0])
    #     z = np.append(z_vals, z_vals[0])
    #     label = f'triangle {i+1}' if i == 0 else None  # Only label the first for legend clarity
    #     triangle = Triangle3D(x_val, y, z, color='b', label=label)
    #     triangle.plot(ax)

    # all_points=np.array(triangs).reshape(-1,3)

    # something = []
    # for point in innP:
    #     something.append(point)
    # for (x,y,z) in triangs:
    #     something.append((x + y + z) / 3)
    # scatter_points = np.array(something)
    # ax.scatter(scatter_points[:, 0], scatter_points[:, 1], scatter_points[:,2], c='r', marker='o')

    # labels = []
    # for point in innP:
    #     print(f'{len(labels)} out of {len(innP)}')
    #     labels.append(str(round(sol(point), 3)))

    # for i in range(len(triangs)):
    #     labels.append(str(round(u[i], 3)))
    #     # labels.append(i)

    # for (x,y,z), label in zip(scatter_points, labels):
    #     ax.text(x,y,z, label, fontsize=10, color='blue')

    # all_points=np.vstack([all_points, scatter_points])

    # xmin, ymin, zmin = all_points.min(axis=0)
    # xmax, ymax, zmax = all_points.max(axis=0)
    # ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), zlim=(zmin, zmax),
    #        xlabel='X', ylabel='Y', zlabel='Z')
    # ax.view_init(elev=20., azim=-35, roll=0)
    # plt.show()

    (a,b,c) = center

    # Grid resolution and extent
    n = definition
    grid_lim = .95 * radius  # range will be [center - grid_lim, center + grid_lim]

    # Create figure
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    y_vals = np.linspace(b - grid_lim, b + grid_lim, n)
    z_vals = np.linspace(c - grid_lim, c + grid_lim, n)
    Y, Z = np.meshgrid(y_vals, z_vals)
    X = np.full_like(Y, a)

    values_xa = np.vectorize(lambda y, z: sol([a, y, z]))(Y, Z)
    im0 = axs[0].imshow(values_xa, extent=[b-grid_lim, b+grid_lim, c-grid_lim, c+grid_lim],
                        interpolation='spline36', origin='lower', aspect='auto')
    axs[0].set_title(f'Plane x = {a}')
    axs[0].set_xlabel('y')
    axs[0].set_ylabel('z')
    fig.colorbar(im0, ax=axs[0])

    x_vals = np.linspace(a - grid_lim, a + grid_lim, n)
    z_vals = np.linspace(c - grid_lim, c + grid_lim, n)
    X, Z = np.meshgrid(x_vals, z_vals)
    Y = np.full_like(X, b)

    values_yb = np.vectorize(lambda x, z: sol([x, b, z]))(X, Z)
    im1 = axs[1].imshow(values_yb, extent=[a-grid_lim, a+grid_lim, c-grid_lim, c+grid_lim],
                        interpolation='spline36', origin='lower', aspect='auto')
    axs[1].set_title(f'Plane y = {b}')
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('z')
    fig.colorbar(im1, ax=axs[1])

    x_vals = np.linspace(a - grid_lim, a + grid_lim, n)
    y_vals = np.linspace(b - grid_lim, b + grid_lim, n)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = np.full_like(X, c)

    values_zc = np.vectorize(lambda x, y: sol([x, y, c]))(X, Y)
    im2 = axs[2].imshow(values_zc, extent=[a-grid_lim, a+grid_lim, b-grid_lim, b+grid_lim],
                        interpolation='spline36', origin='lower', aspect='auto')
    axs[2].set_title(f'Plane z = {c}')
    axs[2].set_xlabel('x')
    axs[2].set_ylabel('y')
    fig.colorbar(im2, ax=axs[2])

    plt.tight_layout()
    plt.show()

    return 0


# make it look better plot #
# check convergence #
# new test case, 0 left 1 right #
# icosahedron sphere #
# electrostatics to compare solutions
# modify to stoke's flow problem
# find gradient on boundary
#   :broken_heart:
#
# 2 spheres

def makeIco(x):
    p  = (1 + np.sqrt(5)) / 2
    r0 = np.sqrt(1 + p**2)
    x  = np.array(x)

    def rotate(point, c, s):
        return [point[0], c*point[1] - s*point[2], s*point[1] + c*point[2]]

    V1  = [0,  1/r0, +p/r0]
    V2  = [0, -1/r0, +p/r0]
    V3  = [0,  1/r0, -p/r0]
    V4  = [0, -1/r0, -p/r0]
    V5  = [ 1/r0,  p/r0, 0]
    V6  = [-1/r0,  p/r0, 0]
    V7  = [ 1/r0, -p/r0, 0]
    V8  = [-1/r0, -p/r0, 0]
    V9  = [ p/r0, 0,  1/r0]
    V10 = [-p/r0, 0,  1/r0]
    V11 = [ p/r0, 0, -1/r0]
    V12 = [-p/r0, 0, -1/r0]

    V1   = rotate(np.array(V1), .850650808352, .525731112119) + x
    V2   = rotate(np.array(V2), .850650808352, .525731112119) + x
    V3   = rotate(np.array(V3), .850650808352, .525731112119) + x
    V4   = rotate(np.array(V4), .850650808352, .525731112119) + x
    V5   = rotate(np.array(V5), .850650808352, .525731112119) + x
    V6   = rotate(np.array(V6), .850650808352, .525731112119) + x
    V7   = rotate(np.array(V7), .850650808352, .525731112119) + x
    V8   = rotate(np.array(V8), .850650808352, .525731112119) + x
    V9   = rotate(np.array(V9), .850650808352, .525731112119) + x
    V10  = rotate(np.array(V10), .850650808352, .525731112119) + x
    V11  = rotate(np.array(V11), .850650808352, .525731112119) + x
    V12  = rotate(np.array(V12), .850650808352, .525731112119) + x

    Ico = [
        [[V1 , V2 , V10], 0],
        [[V1 , V10, V6] , 0],
        [[V1 , V6 , V5] , 0],
        [[V1 , V5 , V9] , 0],
        [[V1 , V9 , V2] , 0],
        [[V4 , V7 , V11], 0],
        [[V4 , V11, V3] , 0],
        [[V4 , V3 , V12], 0],
        [[V4 , V12, V8] , 0],
        [[V4 , V8 , V7] , 0],
        [[V8 , V2 , V7] , 0],
        [[V7 , V2 , V9] , 0],
        [[V7 , V9 , V11], 0],
        [[V11, V9 , V5] , 0],
        [[V11, V5 , V3] , 0],
        [[V3 , V5 , V6] , 0],
        [[V3 , V6 , V12], 0],
        [[V12, V6 , V10], 0],
        [[V12, V10, V8] , 0],
        [[V8 , V10, V2] , 0],
    ]

    return Ico

def makeCube(x, l):
    x = np.array(x)
    temp = np.array([.5,.5,.5])
    V1=np.array([0,0,0]) - temp
    V1= l*V1 + x
    V2=np.array([1,0,0]) - temp
    V2= l*V2 + x
    V3=np.array([0,1,0]) - temp
    V3= l*V3 + x
    V4=np.array([0,0,1]) - temp
    V4= l*V4 + x
    V5=np.array([1,1,0]) - temp
    V5= l*V5 + x
    V6=np.array([1,0,1]) - temp
    V6= l*V6 + x
    V7=np.array([0,1,1]) - temp
    V7= l*V7 + x
    V8=np.array([1,1,1]) - temp
    V8= l*V8 + x

    test = [
        [[V1, V2, V6], 0],
        [[V1, V6, V4], 0],
        [[V4, V6, V8], 0],
        [[V4, V8, V7], 0],
        [[V7, V8, V5], 0],
        [[V8, V5, V3], 0],
        [[V3, V5, V2], 0],
        [[V3, V2, V1], 0],
        [[V2, V1, V7], 0],
        [[V7, V1, V4], 0],
        [[V6, V2, V5], 0],
        [[V8, V6, V5], 0]
    ]

    return test

# # ~~~Test 1
# # Cube with constant potential on 1 pair of parallel faces and 0 flux elsewhere
# # inner points are a grid of n x n points uniformly spread

# test = makeCube([.5,.5,.5], 1)
# test = refine(test,1)

# def testFunc(x):
#     if (x[0] == 1 or x[0] == 0):
#         return [0,1]
#     else:
#         return [1,0]

# test = fixBound(test,testFunc)

# testC = [.5,.5,.5]
# testR = 1

# # ~~~Test 2
# # Cube with linear potential on the bottom face and 0 flux elsewhere
# # inner point is center of the cube

# test = makeCube([.5,.5,.5], 1)
# test = refine(test,1)

# # def testFunc(x):
# #     if (x[2] == 0):
# #         return [0,x[0]]
# #     else:
# #         return [1,0]

# def testFunc(x):
#     if (x[0] == 0):
#         return [0,0]
#     if (x[0] == 1):
#         return [0,1]
#     else:
#         return [1,0]

# test = fixBound(test, testFunc)

# testC = [.5,.5,.5]
# testR = 1

# # ~~~Test 3
# # Cube with linear potential on the bottom face, mixed condition u + q = 0, 0 flux else
# # # inner point is center of the cube

# test = makeCube([.5,.5,.5], 1)
# test = refine(test, 2)

# def testFunc(x):
#     if (x[2] == 0):
#         return [0,1]
#     elif (x[2] == 1):
#         return [2, [1,1,0]]
#     else:
#         return [1,0]

# test = fixBound(test, testFunc)

# testC = [.5,.5,.5]
# testR = 1

# # ~~~ Test 4/5/6/7
# # Cube with 0 potential bottom, 0 flux else, sphere inside with constant potential on surface
# # Cube with 1 potential bottom, 0 flux else, sphere with 0 flux surface
# # Cube with 0 flux bottom, 0 potential else, sphere with 1 potential surface


# # def testFunc1(x):
# #     if (x[2] == 0):
# #         return [0,0]
# #     else:
# #         return [1,0]

# def testFunc1(x):
#     if (x[2] == 0):
#         return [0,0]
#     else:
#         return [1,0]

# # def testFunc2(x):
# #     return [0,1]

# def testFunc2(x):
#     if (x[2] < .5):
#         return [0,1]
#     else:
#         return [1,0]

# # def testFunc2(x):
# #     return [1,0]


# test = makeCube([.5,.5,.5], 1)
# test = refine(test,3)
#
# testSphere = makeIco([.5,.5,.5])
# testSphere = refineSphere(testSphere, [.5,.5,.5], .25, 1)

# test = fixBound(test,testFunc1)
# testSphere = fixBound(testSphere,testFunc2)

# for i in testSphere:
#     test.append(i)

# testC = [.5,.5,.5]
# testR = 1

# Test 9?
# sphere center potential 1,

def testFunc1(x):
    return [0,100]

def testFunc2(x):
    return [0,0]

test = []
testSphere1 = makeIco([0,0,0])
testSphere1 = refineSphere(testSphere1, [0,0,0], 1, 2)

testSphere2 = makeCube([0,0,0], 1000)

testSphere1 = fixBound(testSphere1, testFunc1)
testSphere2 = fixBound(testSphere2, testFunc2)

for x in testSphere1:
    test.append(x)

for y in testSphere2:
    test.append(y)

testC = [0,0,0]
testR = 5

# RUNNING THE CODE

testInnP = []
testT = []
testB = []

for (x,y,z), bound in test:
    testT.append([x,y,z])
    testB.append(bound)

# testT = np.array(testT)

# trueTestT = []
# for (x,y,z) in testT:
#     trueTestT.append([ 2 * x - np.array([1,1,1]),
#                        2 * y - np.array([1,1,1]),
#                        2 * z - np.array([1,1,1])
#                       ])

# testT = trueTestT

constant3D(testT,testB,testInnP, testC, testR, 30)
