#!/usr/bin/env python3

import json, sys
import numpy as np
import pandas as pd
from time import time
from tempfile import NamedTemporaryFile
from tool_create_circles import create_cluster_circle
from tool_create_hexagons import create_cluster_hex
from MD_rigid_rototrasl import MD_rigid_rototrasl

def cluster_inhex_Nl(N1, N2,  a1 = np.array([4.45, 0]), a2 = np.array([-4.45/2, 4.45*np.sqrt(3)/2]),
                     clgeom_fname = "input_circ.hex", cluster_f = create_cluster_circle):
    """Create input file in EP hex format for a cluster of Bravais size Nl"""

    clgeom_file = open(clgeom_fname, 'w') # Store it somewhere for MD function
    with NamedTemporaryFile(prefix=clgeom_fname, suffix='tmp', delete=True) as tmp: # write input on tempfile
        tmp.write(bytes("%i %i\n" % (N1, N2), encoding = 'utf-8'))
        tmp.write(bytes("%15.10g %15.10g\n" % (a1[0], a1[1]), encoding = 'utf-8'))
        tmp.write(bytes("%15.10g %15.10g\n" % (a2[0], a2[1]), encoding = 'utf-8'))
        tmp.seek(0) # Reset 'reading needle'
        pos = cluster_f(tmp.name, clgeom_file)[:,:2] # Pass name of tempfile to create function
    clgeom_file.close() # Close file so MD function can read it

    return pos, clgeom_fname

#------------------------------------------
# INPUTS
#------------------------------------------
# Reads inputs from json file. We will then modify suff we are interested in.
with open(sys.argv[1]) as inj:
    inputs = json.load(inj)

Nl = 3 # 'Bravais' size

#------------------------------------------
# SETUP SYSTEM
#------------------------------------------

#------- CLUSTER GEOM -----
# define cluster shape
clt_shape = 'hexagon'
if clt_shape == 'circle':
    create_cluster = create_cluster_circle
elif clt_shape == 'hexagon':
    create_cluster = create_cluster_hex
else:
    raise ValueError("Symmetry %s not implemented" % symmetry)

# Cluster lattice
Rcl = 4.45
a1 = np.array([Rcl, 0])
a2 = np.array([-Rcl/2, Rcl*np.sqrt(3)/2])

clgeom_fname = "input_circ-Nl_%i.hex" % Nl
pos, _ = cluster_inhex_Nl(Nl, Nl, a1=a1, a2=a2,
                              clgeom_fname=clgeom_fname, cluster_f=create_cluster)
N = pos.shape[0]
inputs['cluster_hex'] = clgeom_fname

# Estimate of free cluster moving
theta1 = inputs['T']/N**2 * inputs['Nsteps']*inputs['dt']

#------- SUBSTRATE --------
# We want to define barrier
inputs['epsilon'] = 105

#------- MD PARAMS --------
# Torque enought to make it move
inputs['T'] = 10*N # fN*Micron
inputs['Nsteps'] = int(2e4)

#--------------------------
# RUN
#--------------------------
print("Start run of %i steps" % inputs['Nsteps'])
print("Free cl should rotate %.2f deg" % (theta1))
c_outf = "out-N_%i.dat" % N
t0 = time() # Start global clock
with open(c_outf, 'w') as c_out:
    MD_rigid_rototrasl([inputs], outstream=c_out, info_fname="info-N_%i.json" % N)

#--------------------------
# ANALYSIS
#--------------------------
data = pd.read_fwf(c_outf, infer_nrows=1e3)
theta_fin, omega_fin = data['06)angle'].tail(1), data['07)omega'].tail(1)
print("Final config"
print("N %10i (Nl %10i) theta fin %20.15g omega fin %20.15g" % (N, Nl, theta_fin, omega_fin))

#------------------------------
out_data.close()
t1 = time()
print("Done in %is (%.2fmin)" % (t1-t0, (t1-t0)/60))
