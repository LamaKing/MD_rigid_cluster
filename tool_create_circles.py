#!/usr/bin/env python3

import sys
import re
import os
import os.path
import math
import numpy as np

def create_cluster(input_cluster):
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

def create_cluster_circle(input_cluster, outstream=sys.stdout):
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
 pos = pos - np.mean(pos,axis=0)
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

if __name__ == "__main__":
 input_cluster = sys.argv[1]
 create_cluster_circle(input_cluster)
