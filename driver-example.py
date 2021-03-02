#!/usr/bin/env python3

import json, sys
import numpy as np
import pandas as pd
from time import time
from tempfile import NamedTemporaryFile
from tool_create_circles import create_cluster_circle
from tool_create_hexagons import create_cluster_hex
from MD_rigid_rototrasl import MD_rigid_rototrasl

def create_cluster_sizeN(N1, N2, a1 = np.array([4.45, 0]), a2 = np.array([-2.225, 3.85381304684075]),
                         clgeom_fname = "input_circ.hex", cluster_f = create_cluster_circle):
    """What am I doing here?"""

    clgeom_file = open(clgeom_fname, 'w') # Store it somewhere for MD function
    with NamedTemporaryFile() as tmp: # write input on tempfile
        tmp.write(bytes("%i %i\n" % (N1, N2), encoding = 'utf-8'))
        tmp.write(bytes("%15.10g %15.10g\n" % (a1[0], a1[1]), encoding = 'utf-8'))
        tmp.write(bytes("%15.10g %15.10g\n" % (a2[0], a2[1]), encoding = 'utf-8'))
        tmp.seek(0) # Reset 'reading needle'
        pos = create_cluster(tmp.name, clgeom_file)[:,:2] # Pass name of tempfile to create function
    clgeom_file.close() # Close file so MD function can read it

    return pos, clgeom_fname

#------------------------------------------
# INPUTS
#------------------------------------------
# AS: Reads inputs from json file, easier for simul driver in python
with open(sys.argv[1]) as inj:
    inputs = json.load(inj)

#------------------------------------------
# SETUP SYSTEM
#------------------------------------------
# define cluster shape
clt_shape = 'hexagon'
if clt_shape == 'circle':
    create_cluster = create_cluster_circle
elif clt_shape == 'hexagon':
    create_cluster = create_cluster_hex
else:
    raise ValueError("Symmetry %s not implemented" % symmetry)

# Cluster lattice
R = 4.45
a1 = np.array([R, 0])
a2 = np.array([-R/2, R*np.sqrt(3)/2])

# Free system no barrier
inputs['epsilon'] = 0

# Torque and final angle
inputs['T'] = 1000 # fN*Micron
inputs['Nsteps'] = int(1e3)

#------------------------------------------
# RUN SCALING
#------------------------------------------
t0 = time() # Start global clock
out_data = open("free_scaling.dat", 'w')
for Ni in range(2,10):

    #--------------------------
    # SETUP
    #--------------------------
    print("On N", Ni)

    #------- CLUSTER GEOM -----
    clgeom_fname = "input_circ-Nl_%i.hex" % Ni
    pos, _ = create_cluster_sizeN(Ni, Ni, a1=a1, a2=a2,
                                  clgeom_fname=clgeom_fname, cluster_f=create_cluster)
    N = pos.shape[0]
    inputs['cluster_hex'] = clgeom_fname

    # ------ LANGEVIN ---------
    eta = 1 # [fKg/ms]
    # CM translational viscosity
    etat_eff = eta*N
    # CM rotational viscosity.
    etar_eff = eta*N**2

    theta1 = inputs['T']/etar_eff * inputs['Nsteps']*inputs['dt']

    # ------ SETUP INFO -------
    input_fname = 'inputs-N_%i.json' % N
    with open(input_fname, 'w') as outj:
        json.dump(inputs, outj)

    #--------------------------
    # RUN
    #--------------------------
    print("Start run of %i steps" % inputs['Nsteps'])
    print("Free cl should rotate %.2f deg" % (theta1))
    c_outf = "out-N_%i.dat" % N
    with open(c_outf, 'w') as c_out:
        MD_rigid_rototrasl([input_fname], outstream=c_out, info_fname="info-N_%i.json" % N)

    #--------------------------
    # ANALYSIS
    #--------------------------
    data = pd.read_fwf(c_outf, infer_nrows=1e3)
    theta_fin, omega_fin = data['06)angle'].tail(1), data['07)omega'].tail(1)
    print("%10i %10i %20.15g %20.15g" % (N, Ni, theta_fin, omega_fin), file=out_data)
    out_data.flush()
    print("-------------------------------------\n") # Separate a bit steps

#------------------------------
out_data.close()
t1 = time()
print("Done in %is (%.2fmin)" % (t1-t0, (t1-t0)/60))
