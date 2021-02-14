#!/usr/bin/env python

import sys
import re
import os
import os.path
import math
import numpy as np

def create_cluster_hex(input_cluster, X0, Y0):
 file = open(input_cluster, 'r')
 N1, N2 = [int(x) for x in file.readline().split()]
 a1 = [float(x) for x in file.readline().split()]
 a2 = [float(x) for x in file.readline().split()]
 N = int(N1*N2+((N2-1)*N2)/2+N2*(N1-1)+((N1-2)*(N1-1))/2)
 pos = np.zeros((N,6))
 iN = 0
 print('{}'.format(N))
 print('{} {}'.format(a1[0], a1[1]))
 print('{} {}'.format(a2[0], a2[1]))
 for i in range(N2):
  for j in range(N1+i):
   pos[iN,0] = j*a1[0]+i*a2[0]
   pos[iN,1] = j*a1[1]+i*a2[1]
   print('{} {}'.format(i, j))
   iN = iN+1
 for i in range(N2,(N2+N1-1)):
  for j in range(N1+N2-2,i-N2,-1):
   pos[iN,0] = j*a1[0]+i*a2[0]
   pos[iN,1] = j*a1[1]+i*a2[1]
   print('{} {}'.format(i,j))
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

if __name__ == "__main__":
 input_cluster = sys.argv[1]
 X0 = 0.0
 Y0 = 0.0
 if (len(sys.argv)>2):
  X0 = sys.argv[2]
  Y0 = sys.argv[3]
 create_cluster(input_cluster, X0, Y0)
