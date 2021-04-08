#!/usr/bin/env python3

import sys
import re
import os
import os.path
import math
import numpy as np

def create_cluster_inhex(input_cluster):
    file = open(input_cluster, 'r')
    N1, N2 = [int(x) for x in file.readline().split()]
    a1 = [float(x) for x in file.readline().split()]
    a2 = [float(x) for x in file.readline().split()]
    N = N1*N2
    pos = np.zeros((N,6))
    iN = 0
    print(N)
    print(a1[0], a1[1])
    print(a2[0], a2[1])
    for i in range(-N1//2+1, N1//2+1):
        for j in range(-N2//2+1, N2//2+1):
            pos[iN,0] = j*a1[0]+i*a2[0]
            pos[iN,1] = j*a1[1]+i*a2[1]
            print(i, j)
            iN = iN+1
    pos = pos - np.mean(pos,axis=0)
    print_xyz(pos)
    print_dat(pos)

def create_cluster(input_cluster, angle=0):
    """create clusters taking as input the two primitive vectors a1 and a2
    and the integer couples describing their positions.
    center of mass in zero."""

    file = open(input_cluster, 'r')
    N = int(file.readline())
    a1 = [float(x) for x in file.readline().split()]
    a2 = [float(x) for x in file.readline().split()]
    pos = np.zeros((N,2))
    for i in range(N):
        index = [float(x) for x in file.readline().split()]
        pos[i,0] = index[0]*a1[0]+index[1]*a2[0]
        pos[i,1] = index[0]*a1[1]+index[1]*a2[1]
    pos -= np.average(pos,axis=0)
    pos = rotate(pos, angle)
    return pos

def create_cluster_circle(input_cluster, outstream=sys.stdout, X0=0, Y0=0):
    file = open(input_cluster, 'r')
    N1, N2 = [int(x) for x in file.readline().split()]
    #DEBUG print("N1", N1, "N2", N2, file=sys.stderr)
    a1 = [float(x) for x in file.readline().split()]
    a2 = [float(x) for x in file.readline().split()]
    #DEBUG print("a", a1, "b", a2, file=sys.stderr)
    a1norm = np.linalg.norm(a1)
    a2norm = np.linalg.norm(a2)
    pos = np.zeros((0,6))
    ipos = np.zeros((0,2), dtype='int')
    iN = 0
    for i in range(-N1*2+1, N1*2+1):
        for j in range(-N2*2+1, N2*2+1):
            X = j*a1[0]+i*a2[0]
            Y = j*a1[1]+i*a2[1]
            if ((X*X + Y*Y) < N1/2*N2/2*a1norm*a2norm):
                pos = np.append(pos,[[X,Y,0,0,0,0]], axis=0)
                ipos = np.append(ipos,[[j, i]], axis=0)
                iN = iN+1
    print(iN, file=outstream)
    print(a1[0], a1[1], file=outstream)
    print(a2[0], a2[1], file=outstream)
    for i in range(iN):
        print(ipos[i,0], ipos[i,1], file=outstream)
    pos = pos - np.mean(pos,axis=0) + np.array([X0, Y0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    print_xyz(pos)
    print_dat(pos)
    return pos

def create_cluster_hex(input_cluster, outstream=sys.stdout, X0=0, Y0=0):
    file = open(input_cluster, 'r')
    N1, N2 = [int(x) for x in file.readline().split()]
    a1 = [float(x) for x in file.readline().split()]
    a2 = [float(x) for x in file.readline().split()]
    N = int(N1*N2+((N2-1)*N2)/2+N2*(N1-1)+((N1-2)*(N1-1))/2)
    pos = np.zeros((N,6))
    iN = 0
    print('{}'.format(N), file=outstream)
    print('{} {}'.format(a1[0], a1[1]), file=outstream)
    print('{} {}'.format(a2[0], a2[1]), file=outstream)
    for i in range(N2):
        for j in range(N1+i):
            pos[iN,0] = j*a1[0]+i*a2[0]
            pos[iN,1] = j*a1[1]+i*a2[1]
            print('{} {}'.format(i, j), file=outstream)
            iN = iN+1
    for i in range(N2,(N2+N1-1)):
        for j in range(N1+N2-2,i-N2,-1):
            pos[iN,0] = j*a1[0]+i*a2[0]
            pos[iN,1] = j*a1[1]+i*a2[1]
            print('{} {}'.format(i,j), file=outstream)
            iN = iN+1
    pos = pos - np.mean(pos,axis=0) + np.array([X0, Y0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    print_xyz(pos)
    print_dat(pos)
    return pos

def print_xyz(pos):
    N = pos.shape[0]
    xyz_out = open('cluster.xyz', 'w')
    xyz_out.write(str(N) + '\n#Cluster\n')
    for i in range(N):
    #  for j in range(3):
        #print("xyz", pos[i])
        print("C %20.15f %20.15f %20.15f" % tuple(pos[i,:3]), file=xyz_out)
    #xyz_out.write('\n')

def print_dat(pos):
    N = pos.shape[0]
    xyz_out = open('cluster.dat', 'w')
    for i in range(N):
        for j in range(3):
            xyz_out.write(str(pos[i,j])+'\n')
    for i in range(N*3+6):
        xyz_out.write('0.00000\n')

def cluster_inhex_Nl(N1, N2,  a1 = np.array([4.45, 0]), a2 = np.array([-4.45/2, 4.45*np.sqrt(3)/2]),
                     clgeom_fname = "input_pos.hex", cluster_f = create_cluster_circle):
    """Create input file in EP hex format for a cluster of Bravais size Nl"""
    from tempfile import NamedTemporaryFile
    clgeom_file = open(clgeom_fname, 'w') # Store it somewhere for MD function
    with NamedTemporaryFile(prefix=clgeom_fname, suffix='tmp', delete=True) as tmp: # write input on tempfile
        tmp.write(bytes("%i %i\n" % (N1, N2), encoding = 'utf-8'))
        tmp.write(bytes("%15.10g %15.10g\n" % (a1[0], a1[1]), encoding = 'utf-8'))
        tmp.write(bytes("%15.10g %15.10g\n" % (a2[0], a2[1]), encoding = 'utf-8'))
        tmp.seek(0) # Reset 'reading needle'
        pos = cluster_f(tmp.name, clgeom_file)[:,:2] # Pass name of tempfile to create function
    clgeom_file.close() # Close file so MD function can read it

    return pos, clgeom_fname

# rotates pos vector (the first two rows are X,Y) by an angle in degrees
def rotate(pos, angle):
    # Equivalent to. Or should be!
    #for i in range(pos.shape[0]):
    #    newx = pos[i,0] * np.cos(angle/180*np.pi) - pos[i,1] * np.sin(angle/180*np.pi)
    #    newy = pos[i,0] * np.sin(angle/180*np.pi) + pos[i,1] * np.cos(angle/180*np.pi)
    #    pos[i,0] = newx
    #    pos[i,1] = newy
    roto_mtr = np.array([[np.cos(angle/180*np.pi), -np.sin(angle/180*np.pi)],
                         [np.sin(angle/180*np.pi), np.cos(angle/180*np.pi)]])
    pos = np.dot(roto_mtr, pos.T).T # NumPy inverted convention on row/col
    return pos

if __name__ == "__main__":
    input_cluster = sys.argv[1]
    X0 = 0.0
    Y0 = 0.0
    clt_shape = sys.argv[2]
    if clt_shape == 'circle':
        create_cluster_func = create_cluster_circle
    elif clt_shape == 'hexagon':
        create_cluster_func = create_cluster_hex
    else:
        raise ValueError("Shape %s not implemented" % clt_shape)

    if len(sys.argv)>4:
        X0 = sys.argv[3]
        Y0 = sys.argv[4]
    create_cluster_func(input_cluster, X0=X0, Y0=Y0)
