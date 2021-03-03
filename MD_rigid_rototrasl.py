#!/usr/bin/env python

import sys
import json
import logging
import numpy as np
from time import time

# calcolates simple matrix for mapping clusters colloids into primitive cell and viceversa.
def calc_matrices_square(R):
    """Metric matrices of square lattice of spacing R"""
    area = R*R
    u     = np.array([[1,0], [0,1]])*R/area
    u_inv = np.array([[1,0], [0,1]])*R
    return u, u_inv

def calc_matrices_triangle(R):
    """Metric matrices of triangular lattice of spacing R"""
    area = R*R*np.sqrt(3)/2.
    # NN along y
    #    u     = np.array([[1,0], [-1./2, sqrt(3)/2]])*R/area
    #    u_inv = np.array([[sqrt(3)/2,0], [1/2,1]])*R
    # NN along x (like tool_create_hex/circ)
    u =     np.array([[np.sqrt(3)/2.,0.5], [0,1]])*R/area
    u_inv = np.array([[1,-0.5],            [0.0, np.sqrt(3)/2.]])*R
    return u, u_inv

def calc_en_tan(pos, a, b, ww, epsilon, u, u_inv):
    """Calculate energy and forces. Well is approximated with tanh function."""
    en = 0
    F = np.zeros((2))
    # map to substrate cell
    Xp = pos[:,0]*u[0,0] + pos[:,1]*u[0,1]
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
    # calculation of energy and force. See X. Cao Phys. Rev. E 103, 1 (2021).
    xx = (Rin-a)/(b-a) # Reduce coordinate rho in [0,1]
    # energy
    en += np.sum(epsilon/2.*(np.tanh((xx-ww)/xx/(1-xx))-1.))
    # force F = - grad(E)
    ff = (xx-ww)/xx/(1-xx)
    ass = (np.cosh(ff)*(1-xx)*xx)*(np.cosh(ff)*(1-xx)*xx)
    vecF = -epsilon/2*(xx*xx+ww-2*ww*xx)/ass
    # Go from rho to r again
    vecF /= (b-a)
    # Project to x and y
    F[0] = np.sum(vecF*Xin/Rin)
    F[1] = np.sum(vecF*Yin/Rin)
    return en, F

def create_cluster(input_cluster, angle):
    """create clusters taking as input the two primitive vectors a1 and a2
    and the integer couples describing their positions.
    center of mass in zero."""

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

def rotate(pos, angle):
    """rotates pos vector (the first two rows are X,Y) by an angle in degrees"""
    for i in range(pos.shape[0]):
        newx = pos[i,0] * np.cos(angle/180*np.pi) - pos[i,1] * np.sin(angle/180*np.pi)
        newy = pos[i,0] * np.sin(angle/180*np.pi) + pos[i,1] * np.cos(angle/180*np.pi)
        pos[i,0] = newx
        pos[i,1] = newy
    return pos

def MD_rigid_rototrasl(argv, outstream=sys.stdout, info_fname='info.json', pos_fname='pos_rotate.dat', debug=False):
    """Overdamped Langevin Molecular Dynamics of rigid cluster over a substrate"""

    t0 = time() # Start clock

    #-------- SET UP LOGGER -------------
    c_log = logging.getLogger("MD_rigid_rototrasl") # Set name of the function
    # Adopted format: level - current function name - message. Width is fixed as visual aid.
    logging.basicConfig(format='[%(levelname)5s - %(funcName)10s] %(message)s')
    c_log.setLevel(logging.INFO)
    if debug: c_log.setLevel(logging.DEBUG)

    #-------- READ INPUTS -------------
    if type(argv[0]) == dict: # Inputs passed as python dictionary
        inputs = argv[0]
    elif type(argv[0]) == str: # Inputs passed as path to json file
        with open(argv[0]) as inj:
            inputs = json.load(inj)
    else:
        raise TypeError('Unrecognized input structure (no dict or filename str)', inputs)
    c_log.debug("Input dict \n%s", "\n".join(["%10s: %10s" % (k, str(v)) for k, v in inputs.items()]))

    # Cluster
    input_cluster = inputs['cluster_hex'] # Geom as lattice and Bravais points
    try:
        angle = inputs['angle'] # Starting angle [deg]
    except KeyError:
        angle = 0 # If not given, start aligned
        pass
    try:
        pos_cm = np.array(inputs['pos_cm']) # Start pos [micron]
    except KeyError:
        pos_cm = np.zeros(2) # If not given, start from centre

    # Substrate
    symmetry = inputs['sub_symm'] # Substrate symmetry (triangle or square)
    a = inputs['a'] # Well end radius [micron]
    b = inputs['b'] # Well slope radius [micron]
    R = inputs['R'] # Well lattice spacing [micron]
    wd = inputs['wd'] # Well asymmetry. 0.29 is a good value
    epsilon =  inputs['epsilon'] # Well depth [zJ]

    # External torque T and external forces Fx, Fy
    T = inputs['T'] # [zJ]
    Fx, Fy = inputs['Fx'], inputs['Fy'] # [fN]

    # MD params
    Nsteps = inputs['Nsteps']
    dt = inputs['dt'] # [ms]
    dtheta = inputs['dtheta'] # [deg]
    print_skip = 100 # Timesteps to skip between prints
    try:
       print_skip = inputs['print_skip']
    except KeyError:
        pass
    printprog_skip = int(Nsteps/20) # Progress output frequency
    c_log.debug("Print every %i timesteps. Status update every %i." % (print_skip, printprog_skip))

    # Stuck config exit
    omega_avg = 0 # store average of omega over given timesteps
    omega_avglen = 20 # timesteps
    try:
        min_Nsteps, omega_tol = inputs['min_Nsteps'], inputs['omega_tol']
    except KeyError:
        # If not given, continue indefinitely: min steps huge.
        min_Nsteps = 1e30 # min steps for average. E.g. 1e5
        omega_tol =  1e-10 # tolerance to consider the system stuck. Careful with sign!
        pass
    c_log.info("Stuck val: N %g avglen %i omega tol %.4g deg/ms" % (min_Nsteps, omega_avglen, omega_tol))

    if min_Nsteps < omega_avglen:
        raise ValueError("Omega average length larger them minimum number of steps")

    # Spinning config exit
    try:
        theta_max = inputs['theta_max']+angle
    except KeyError:
        # If not given, continue indefinitely: theta_max huge.
        theta_max = 1e30 # Exit if cluster rotates more than this
        pass
    c_log.info("Theta max =  %.4g deg" % theta_max)

    #-------- SETUP SYSTEM  -----------
    # create cluster
    pos = create_cluster(input_cluster, angle)[:,:2]
    np.savetxt(pos_fname, pos)
    N = pos.shape[0] # Size of the cluster

    # define substrate metric
    if symmetry == 'square':
        calc_matrices = calc_matrices_square
    elif symmetry == 'triangle':
        calc_matrices = calc_matrices_triangle
    else:
        raise ValueError("Symmetry %s not implemented" % symmetry)
    u, u_inv = calc_matrices(R)

    # initialise variable
    forces, torque = np.zeros(2), 0.

    #-------- LANGEVIN ----------------
    # Assumes rotation and translation indipendent. We just care about the scaling, not exact number.
    eta = 1   # Translational damping of single colloid
    # CM translational viscosity
    etat_eff = eta*N # [fKg/ms]
    # CM rotational viscosity.
    #etar_eff = etat*np.sum(pos**2) # CM rotational viscosity. Prop to N^2. Varying with shape as well.
    etar_eff = eta*N**2 # [micron^2*fKg/ms]

    c_log.info("Number of particles %i Eta trasl %.5g" % (N, eta))
    c_log.info("Eta tras eff %.5g Eta roto eff %.5g Ratio roto/tras %.5g" % (etat_eff, etar_eff, etar_eff/etat_eff))
    c_log.info("T = %.4g fN*micron (T/N=%.4g)" % (T, T/N))
    c_log.debug("Free cluster would rotate %.2f deg", T/etar_eff * Nsteps * dt)
    c_log.info("Fx = %.4g fN (Fx/N=%.4g), Fy = %.4g fN (Fy/N=%.4g), |F| = %.4g fN (|F|/N=%.4g)",
               Fx, Fx/N, Fy, Fy/N, np.sqrt(Fx**2+Fy**2), np.sqrt(Fx**2+Fy**2)/N)
    c_log.debug("Free cluster would translate %.2f micron", np.sqrt(Fx**2+Fy**2)/etat_eff * Nsteps * dt)

    #-------- INFO FILE ---------------
    with open(info_fname, 'w') as infof:
        infod = {'eta': eta, 'etat_eff': etat_eff, 'etar_eff': etar_eff, 'N': N, 'theta_max': theta_max, 'print_skip': print_skip,
                 'min_Nsteps': min_Nsteps, 'omega_avglen': omega_avglen, 'omega_tol': omega_tol, 'pos_cm': pos_cm.tolist()
        }
        infod.update(inputs)
        if debug: c_log.debug("Info dict\n %s" % ("\n".join(["%s: %s" % (k, type(v)) for k, v in infod.items()])))
        json.dump(infod, infof, indent=True)

    #-------- OUTPUT HEADER -----------
    # !! Labels and print_status data structures must be coherent !!
    num_space = 30 # Width printed numerical values
    indlab_space = 2 # Header index width
    lab_space = num_space-indlab_space-1 # Match width of printed number, including parenthesis
    header_labels = ['e_pot', 'pos_cm[0]', 'pos_cm[1]', 'Vcm[0]', 'Vcm[1]',
                     'angle', 'omega', 'forces[0]', 'forces[1]', 'torque-T']
    # Gnuplot-compatible (leading #) fix-width output file
    first = '#{i:0{ni}d}){s: <{n}}'.format(i=0, s='dt*it', ni=indlab_space, n=lab_space-1,c=' ')
    print(first+"".join(['{i:0{ni}d}){s: <{n}}'.format(i=il+1, s=lab, ni=indlab_space, n=lab_space,c=' ')
                       for il, lab in zip(range(len(header_labels)), header_labels)]), file=outstream)

    # Inner-scope shortcut for printing
    def print_status():
        data = [dt*it, e_pot, pos_cm[0], pos_cm[1], Vcm[0], Vcm[1],
                angle, omega, forces[0], forces[1], torque-T]
        print("".join(['{n:<{nn}.16g}'.format(n=val, nn=num_space)
                       for val in data]), file=outstream)

    # Print setup time
    t_exec = time()-t0
    c_log.info("Setup in %is (%.2fmin or %.2fh)", t_exec, t_exec/60, t_exec/3600)

    #-------- START MD ----------------
    t0 = time() # Start clock
    for it in range(Nsteps):

        # SOLVE DYNAMICS
        # energy is -epsilon inside well, 0 outside well. Continuous function in between.
        e_pot, forces = calc_en_tan(pos + pos_cm, a, b, wd, epsilon, u, u_inv)
        # torque is T = - dE / dTheta = - (E(theta+) - E(theta-)) / 2dTheta
        en_plus, _ = calc_en_tan(rotate(pos, dtheta) + pos_cm, a, b, wd, epsilon, u, u_inv)
        en_minus,_ = calc_en_tan(rotate(pos,-dtheta) + pos_cm, a, b, wd, epsilon, u, u_inv)
        # add external torque
        torque = -(en_plus - en_minus)/dtheta/2

        Vcm = (forces + [Fx, Fy])/etat_eff

        omega = (torque + T)/etar_eff

        # Print progress
        if it % printprog_skip == 0:
            c_log.info("t=%10.3g of %5.2g (%2i%%) E=%15.7g  x=%9.3g y=%9.3g theta=%9.3g omega=%9.3g |Vcm|=%9.3g",
                       it*dt, Nsteps*dt, 100.*it/Nsteps, e_pot, pos_cm[0], pos_cm[1], angle, omega, np.sqrt(Vcm[0]**2+Vcm[1]**2))

        # Print step results
        if it % print_skip == 0: print_status()

        # UPDATE DEGREES OF FREEDOM
        # center of mass follows local forces.
        pos_cm[:] = pos_cm[:] + dt * Vcm
        # angle of cluster follows local torque.
        dangle = dt*omega
        angle += dangle
        # positions are further rotated
        pos = rotate(pos,dangle)

        # Compute omega average
        omega_avg += omega # Average omega to check if system is stuck. See omega_avglen.
        if it % omega_avglen == 0: omega_avg /= omega_avglen
        # If system is stuck, exit
        if np.abs(omega_avg) < omega_tol and it >= min_Nsteps:
            c_log.info("System is stuck (omega=%10.4g). Exit." % omega_avg)
            break

        # If system is rotating without stopping, exit
        if angle >= theta_max and it >= min_Nsteps:
            c_log.info("System is rotating freely (angle=%10.4g). Exit." % angle)
            break
    #-----------------------

    # Print last step, if needed
    c_log.info("t=%10.3g of %5.2g (%2i%%) E=%15.7g  x=%9.3g y=%9.3g theta=%9.3g omega=%9.3g |Vcm|=%9.3g",
               it*dt, Nsteps*dt, 100.*it/Nsteps, e_pot, pos_cm[0], pos_cm[1], angle, omega, np.sqrt(Vcm[0]**2+Vcm[1]**2))
    print_status()

    # Print execution time
    t_exec = time()-t0
    c_log.info("Done in %is (%.2fmin or %.2fh)" % (t_exec, t_exec/60, t_exec/3600))

# Stand-alone scripting
if __name__ == "__main__":
    MD_rigid_rototrasl(sys.argv[1:], debug=False)
