#!/usr/bin/env python

import sys
import re
import os
import os.path
import math
import json
import numpy as np
from time import time

# calcolates simple matrix for mapping clusters colloids into primitive cell and viceversa.
# !! AS: a,b are relics of an ancent code?
def calc_matrices_square(R,a,b):
 area = R*R

 u     = np.array([[1,0], [0,1]])*R/area
 u_inv = np.array([[1,0], [0,1]])*R

 return u, u_inv

def calc_matrices_triangle(R,a,b):
 area = R*R*np.sqrt(3)/2.

 #    Along y
 #    u     = np.array([[1,0], [-1./2, sqrt(3)/2]])*R/area
 #    u_inv = np.array([[sqrt(3)/2,0], [1/2,1]])*R

 #   Along x, like tool_create does
 u =     np.array([[np.sqrt(3)/2.,0.5], [0,1]])*R/area
 u_inv = np.array([[1,-0.5],            [0.0, np.sqrt(3)/2.]])*R

 return u, u_inv

# takes input positions, some data describing the substrate well (a,b,ww)
# and some data for mapping cluster colloids into primitive cell of substrate.
def calc_en_tan(pos, a, b, ww, epsilon, u, u_inv):
 F = np.zeros((2))
 # map to substrate cell
 Xp = pos[:,0]*u[0,0] + pos[:,1]*u[0,1]
 #!! AS: Took a while to notice this...
 #Yp = pos[:,0]*u[0,1] + pos[:,1]*u[1,1]
 Yp = pos[:,0]*u[1,0] + pos[:,1]*u[1,1]
 Xp -= np.floor(Xp + 0.5)
 Yp -= np.floor(Yp + 0.5)
 X = Xp*u_inv[0, 0] + Yp*u_inv[0, 1]
 Y = Xp*u_inv[1, 0] + Yp*u_inv[1, 1]
 R  = np.sqrt(X*X+Y*Y)
 # energy inside flat bottom region
 en = -np.sum(R <= a)*epsilon
 # colloids inside the curve region
 # mask and relative R
 inside = np.logical_and(R<b, R>a)
 Xin = X[inside]
 Yin = Y[inside]
 Rin = R[inside]
 # calculation of energy and force. See X. Cao Phys. Rev. E 103, 1 (2021)
 xx = (Rin-a)/(b-a) # Reduce coordinate rho in [0,1]
 # energy
 en += np.sum(epsilon/2.*(np.tanh((xx-ww)/xx/(1-xx))-1.))
 # force F = - grad(E)
 ff = (xx-ww)/xx/(1-xx)
 #!! AS: Doesn't matter, as it's squared, but there's a sign error
 #ass = (np.cosh(ff)*(xx-1)*xx)*(np.cosh(ff)*(xx-1)*xx)
 ass = (np.cosh(ff)*(1-xx)*xx)*(np.cosh(ff)*(1-xx)*xx)
 vecF = -epsilon/2*(xx*xx+ww-2*ww*xx)/ass
 #!! AS: dV(xi)/dr=dxi/dr dV/xi, here is dxi/dr, no?
 vecF /= (b-a)
 #!! AS to EP: Why multiply for x/r? Just to project total force on x,y components of CM?
 F[0] = np.sum(vecF*Xin/Rin)
 F[1] = np.sum(vecF*Yin/Rin)
 return en, F

#!! AS: Not needed technically, no?
# remaps X,Y into primitive cell of substrate.
def remapping(X,Y,u,u_inv):
 Xp = X*u[0][0] + Y*u[0][1]
 Yp = X*u[1][0] + Y*u[1][1]
 print(Xp, Yp)
 Xp = Xp - np.floor(Xp+0.5)
 Yp = Yp - np.floor(Yp+0.5)
 print(Xp, Yp)
 X  = Xp*u_inv[0][0] + Yp*u_inv[0][1]
 Y  = Xp*u_inv[1][0] + Yp*u_inv[1][1]
 R  = np.sqrt(X*X+Y*Y)
 return X, Y, R, Xp, Yp

# create clusters taking as input the two primitive vectors a1 and a2
# and the integer couples describing their positions.
# center of mass in zero.
def create_cluster_hex(input_cluster, angle):
 file = open(input_cluster, 'r')
 N = int(file.readline())
 a1 = [float(x) for x in file.readline().split()]
 a2 = [float(x) for x in file.readline().split()]
 pos = np.zeros((N,6))
 for i in range(N):
  index = [float(x) for x in file.readline().split()]
  pos[i,0] = index[0]*a1[0]+index[1]*a2[0]
  pos[i,1] = index[0]*a1[1]+index[1]*a2[1]
 pos -= np.average(pos,axis=0)
 pos = rotate(pos, angle)
 return pos

# rotates pos vector (the first two rows are X,Y) by an angle in rad
def rotate(pos, angle):
 for i in range(pos.shape[0]):
  #newx = pos[i,0] * np.cos(angle/180*np.pi) - pos[i,1] * np.sin(angle/180*np.pi)
  #newy = pos[i,0] * np.sin(angle/180*np.pi) + pos[i,1] * np.cos(angle/180*np.pi)
  newx = pos[i,0] * np.cos(angle) - pos[i,1] * np.sin(angle)
  newy = pos[i,0] * np.sin(angle) + pos[i,1] * np.cos(angle)
  pos[i,0] = newx
  pos[i,1] = newy
 return pos

def MD_rigid_rototrasl(argv, outstream=sys.stdout, info_fname='info.json'):
    """Add description"""

    # TODO:
    # - add logger instead of print on stderr.
    # - clean weird printing and stuff
    # - clean code a bit
    # - correct indentation: it's 4 spaces according to PEP8!
    # - IMPLEMENTE OMEGA_STOP: omega < tol interrupt

    #------------------------------------------
    # INPUTS
    #------------------------------------------
    # AS: Reads inputs from json file, easier for simul driver in python
    with open(argv[0]) as inj:
       inputs = json.load(inj)

    #print(inputs, file=sys.stderr)
    omega_tol, minNsteps = 1e30, 1e30
    try:
     omega_tol = inputs['omega_tol']
     minNsteps = inputs['minNsteps']
     print("omega tol", omega_tol, "min N", minNsteps)
    except KeyError:
     print("No omega-stop and min N steps given, do full run")

    # Cluster
    input_cluster = inputs['cluster_hex'] # Geom as lattice and Bravais points
    angle = inputs['angle']*np.pi/180 # Starting angle [deg] -> [rad]

    # Substrate
    symmetry = inputs['sub_symm'] # Substrate symmetry (hex or square)
    a = inputs['a'] # Well end radius [micron]
    b = inputs['b'] # Well slope radius [micron]
    R = inputs['R'] # Well lattice spacing [micron]
    wd = inputs['wd'] # Well asymmetry. 0.29 is a good value
    epsilon =  inputs['epsilon'] # Well depth [zJ]

    # External torque T
    T = inputs['T'] # [zJ]
    # External force Fx, Fy
    Fx = inputs['Fx'] # [fN]
    Fy = inputs['Fy'] # [fN]

    # MD params
    Nsteps = inputs['Nsteps']
    dt = inputs['dt'] # [ms]
    dtheta = inputs['dtheta']*np.pi/180 # [deg] -> [rad]

    #------------------------------------------
    # SETUP SYSTEM
    #------------------------------------------
    # create cluster
    pos = create_cluster_hex(input_cluster, angle)[:,:2]
    np.savetxt('pos_rotate.dat', pos)
    N = pos.shape[0]

    # define substrate metric
    if symmetry == 'square':
     calc_matrices = calc_matrices_square
    elif symmetry == 'triangle':
     calc_matrices = calc_matrices_triangle
    else:
     raise ValueError("Symmetry %s not implemented" % symmetry)
    u, u_inv = calc_matrices(R,a,b)

    # initialise variable
    forces = np.zeros(2)
    torque = 0.
    pos_cm = np.zeros(2)

    # -------LANGEVIN----------
    # At T=0 this just scales the time, energy barriers are untouched.
    # Dissipation quantities. Assumes rotation and translation indipendent.
    # Translational damping of sigle colloid. Mass = 1 fKg
    eta = 1 # [fKg/ms]
    # CM translational viscosity
    etat_eff = eta*N
    # CM rotational viscosity.
    # Prop to N^(3/2)
    #!! AS: Not sure it makes much sense, but needed to match units: viscous FORCE applied somewhere to get a torque
    #fulcrum = 1 # [micron]
    #etar_eff = fulcrum*eta*np.sum(np.linalg.norm(pos, axis=1)) # [fKg*micron^2/ms]
    # Prop to N+N^2
    #!! AS: Xin derivation. Assume etar/eta=1. Include Stokes derivation
    #etar_eff = eta*(N+*np.sum(np.linalg.norm(pos, axis=1)**2)) [fKg*micron^2/ms]
    #!! AS: I think we could skip the Stokes treatment, no?
    # sigma = 4.45 # micron. Colloid radius.
    sigma = 1 # fake-micron. Ignore colloid radius
    Ic = 3/N/sigma**2*np.sum(pos**2) # Shape factor prop to N
    etar_eff = N*np.pi*eta*sigma**3*(1+Ic) # Prot to N^2

    print("Number of particles %i Eta trasl %.5g" % (N, eta), file=sys.stderr)
    print("Eta trasl eff %.5g Eta roto eff %.5g" % (etat_eff, etar_eff),
          "Ratio roto/tras %.5g" % (etar_eff/etat_eff), file=sys.stderr)

    # TODO: random forces at finite temperature
    #!! AS: How do they scale with number of particles? With sqrt(N), see notes with CentraLimTheo
    #!! AS: Do I need a "rotational noise" as well?

    # Print system info, for debug, analysis and reproducibility
    with open(info_fname, 'w') as infof:
      infod = {'eta': eta, 'etat_eff': etat_eff, 'etar_eff': etar_eff, 'N': N}
      infod.update(inputs)
      json.dump(infod, infof, indent=True)

    #------------------------------------------
    # OUTPUT HEADER
    #------------------------------------------
    num_space = 25 # Width printed numerical values
    indlab_space = 2 # Header index width
    lab_space = num_space-indlab_space-1 # Match width of printed number, including parenthesis
    header_labels = ['dt*it', 'e_pot',
                     'pos_cm[0]', 'pos_cm[1]', 'Vcm[0]', 'Vcm[1]',
                     'angle', 'angle%360', 'omega',
                     'forces[0]', 'forces[1]', 'torque-T']
    print('#'+"".join(['{i:0{ni}d}){s: <{n}}'.format(i=il, s=lab, ni=indlab_space, n=lab_space,c=' ')
                       for il, lab in zip(range(len(header_labels)), header_labels)]
                     ),
          file=outstream
    )

    #------------------------------------------
    # STARTS MD
    #------------------------------------------
    printerr_skip = int(Nsteps/25)
    print_skip = 50
    omega_runavg = np.zeros(printerr_skip)
    stop_count = 0
    stop_count_lim = 5
    omega_doubleavg = 0

    t0 = time() # Start clock
    for it in range(Nsteps):

        # energy is -epsilon inside well, 0 outside well.
        e_pot, forces = calc_en_tan(pos + pos_cm, a, b, wd, epsilon, u, u_inv)
        # torque is T = - dE / dTheta = - (E(theta+) - E(theta-)) / 2dTheta
        en_plus, _ = calc_en_tan(rotate(pos, dtheta) + pos_cm, a, b, wd, epsilon, u, u_inv)
        en_minus, _ = calc_en_tan(rotate(pos, -dtheta) + pos_cm, a, b, wd, epsilon, u, u_inv)
        torque = -(en_plus - en_minus)/dtheta/2

        # center of mass follows local forces.
        Vcm = (forces + [Fx, Fy])/etat_eff
        pos_cm[:] = pos_cm[:] + dt * Vcm

        # angle of cluster follows local torque.
        omega = (torque+T)/etar_eff
        dangle = dt*omega
        angle += dangle
        # positions are further rotated
        pos = rotate(pos,dangle)
        omega_runavg[int(it%print_skip)] = omega

        # Print progress
        if it % printerr_skip == 0:
         print("t=%10.4g of %.4g " % (it*dt, Nsteps*dt),
                 "(%2i%%)" % (100.*it/Nsteps),
                 "x=%10.3g y=%10.3g" % (pos_cm[0], pos_cm[1]),
                 "theta=%8.2f" % (angle*180/np.pi),
                 "omega=%8.2g" % omega,
                 "|Vcm|=%8.2g" % (np.sqrt(Vcm[0]**2+Vcm[1]**2)),
                 file=sys.stderr)

        # Print step results
        if it % print_skip == 0:
         data = [dt*it, e_pot,
                 pos_cm[0], pos_cm[1], Vcm[0], Vcm[1],
                 angle*180/np.pi, ((angle*180/np.pi)%360), omega,
                 forces[0], forces[1], torque]
         print("".join(['{n:<{nn}.16g}'.format(n=val, nn=num_space)
                        for val in data]),
               file=outstream
         )
         omega_avg = np.abs(np.average(omega_runavg))
         if omega_avg < omega_tol and it > minNsteps:
          print("Omega stuck strike", stop_count)
          omega_doubleavg += omega_avg
          stop_count += 1
         else:
          stop_count = 0
          omega_doubleavg = 0
         if stop_count >= stop_count_lim:
          print("System stuck for %i times in a row (omega=%.2g) at step %i. Exit." % (stop_count_lim,
                                                                                       omega_doubleavg/stop_count_lim,
                                                                                       it), file=sys.stderr)
          break
         omega_runavg = np.zeros(printerr_skip)

    # Always print the last point
    data = [dt*it, e_pot, pos_cm[0], pos_cm[1], Vcm[0], Vcm[1],
            angle*180/np.pi, ((angle*180/np.pi)%360), omega,
            forces[0], forces[1], torque]
    print("".join(['{n:<{nn}.16g}'.format(n=val, nn=num_space) for val in data]), file=outstream)
    t1 = time() # Stop clock
    t_exec = t1-t0
    print("Done in %is or %.2fmin or %.2fh" % (t_exec, t_exec/60, t_exec/3600), file=sys.stderr)
    print("Speed %5.3f s/step" % (t_exec/Nsteps), file=sys.stderr)

if __name__ == "__main__":
   MD_rigid_rototrasl(sys.argv[1:])
