#!/usr/bin/env python

import sys
import json
import logging
import numpy as np
from numpy.random import normal
from numpy.linalg import norm as npnorm
from time import time
from tool_create_cluster import create_cluster, rotate, calc_cluster_langevin
from tool_create_substrate import gaussian, calc_matrices_triangle, calc_matrices_square, calc_en_gaussian, calc_en_tan

def MD_rigid_rototrasl(argv, outstream=sys.stdout, name=None, info_fname=None, pos_fname=None, logger=None, debug=False):
    """Overdamped Langevin Molecular Dynamics of rigid cluster over a substrate"""

    t0 = time() # Start clock

    if name == None: name = 'MD_rigid_rototrasl'
    if info_fname == None: info_fname = "info-%s.json" % name
    if pos_fname == None: pos_fname = "pos-%s.dat" % name

    #-------- SET UP LOGGER -------------
    if logger == None:
        c_log = logging.getLogger("MD_rigid_rototrasl") # Set name of the function
        # Adopted format: level - current function name - message. Width is fixed as visual aid.
        logging.basicConfig(format='[%(levelname)5s - %(funcName)10s] %(message)s')
        c_log.setLevel(logging.INFO)
        if debug: c_log.setLevel(logging.DEBUG)
    else:
        c_log = logger

    #-------- INPUTS -------------
    if type(argv[0]) == dict: # Inputs passed as python dictionary
        inputs = argv[0]
    elif type(argv[0]) == str: # Inputs passed as path to json file
        with open(argv[0]) as inj:
            inputs = json.load(inj)
    else:
        raise TypeError('Unrecognized input structure (no dict or filename str)', inputs)
    c_log.debug("Input dict \n%s", "\n".join(["%10s: %10s" % (k, str(v)) for k, v in inputs.items()]))

    # -- Cluster --
    input_cluster = inputs['cluster_hex'] # Geom as lattice and Bravais points
    pos_cm = np.zeros(2, dtype=float) # If not given, start from centre
    angle = 0 # If not given, start aligned
    try: angle = inputs['angle'] # Starting angle [deg]
    except KeyError: pass
    try: pos_cm = np.array(inputs['pos_cm'], dtype=float) # Start pos [micron]
    except KeyError: pass
    # create cluster
    pos = create_cluster(input_cluster, angle)[:,:2]

    np.savetxt(pos_fname, pos)
    N = pos.shape[0] # Size of the cluster
    c_log.info("Cluster N=%i start at (x,y,theta)=(%.3g,%.3g,%.3g)" % (N, *pos_cm, angle))

    # -- Substrate --
    # define substrate metric
    sub_symmetry = inputs['sub_symm'] # Substrate symmetry (triangle or square)
    well_shape = inputs['well_shape'] # Substrate well shape (Gaussian or Tanh)
    R = inputs['R'] # Well lattice spacing [micron]
    epsilon =  inputs['epsilon'] # Well depth [zJ]

    if sub_symmetry == 'square':
        calc_matrices = calc_matrices_square
    elif sub_symmetry == 'triangle':
        calc_matrices = calc_matrices_triangle
    else:
        raise ValueError("Symmetry %s not implemented" % sub_symmetry)
    u, u_inv = calc_matrices(R)

    if well_shape == 'tanh':
        # Realistic well energy landscape
        calc_en_f = calc_en_tan
        a = inputs['a'] # Well end radius [micron]
        b = inputs['b'] # Well slope radius [micron]
        wd = inputs['wd'] # Well asymmetry. 0.29 is a good value
        en_params = [a, b, wd, epsilon, u, u_inv]
    elif well_shape == 'gaussian':
        # Gaussian energy landscape
        #sigma = inputs['sigma'] # Width of Gaussian
        #en_params = [sigma, epsilon, u, u_inv]
        #calc_en_f = calc_en_gaussian
        # Gaussian energy landscape
        #a = R/2*inputs['at'] # Tempered tail as fraction of R
        #b = R/2*inputs['bt'] # Flat end as fraction of R
        a = inputs['a'] # Well end radius [micron]
        b = inputs['b'] # Well slope radius [micron]
        sigma = inputs['sigma'] # Width of Gaussian
        en_params = [a, b, sigma, epsilon, u, u_inv]
        calc_en_f = calc_en_gaussian
    else:
        raise ValueError("Form %s not implemented" % well_shape)
    c_log.info("%s substrate R=%.3g %s well shape depth=%.3g" % (sub_symmetry, R, well_shape, epsilon))

    # -- MD params --
    T = inputs['T'] # kBT [zJ]
    Tau = inputs['Tau'] # [fN*micron]
    Fx, Fy = inputs['Fx'], inputs['Fy'] # [fN]
    F = np.array([Fx, Fy])
    Nsteps = inputs['Nsteps']
    dt = inputs['dt'] # [ms]
    print_skip = 100 # Timesteps to skip between prints
    try: print_skip = inputs['print_skip']
    except KeyError: pass
    printprog_skip = int(Nsteps/20) # Progress output frequency
    c_log.debug("Print every %i timesteps. Status update every %i." % (print_skip, printprog_skip))
    # initialise variable
    forces, torque = np.zeros(2), 0.

    # -- Simulation break conditions --
    both_breaks = True # Break if both V and omega satisfy conditions
    try: both_breaks = bool(inputs['both_breaks'])
    except KeyError: pass
    break_omega, break_V = False, False
    omega_avg, vel_avg = 0, 0 # store average of omega and velox over given timesteps
    avglen = 100 # timesteps
    min_Nsteps = 1e30 # min steps for average. E.g. 1e5
    omega_min, omega_max = -1, 1e30 # tolerance (>0) to consider the system stuck or depinned.
    vel_min, vel_max = -1, 1e30     # If not given, continue indefinitely: max huge, min <0
    rcm_max, theta_max = 1e30, 1e30

    # Set Stuck config exit
    try: min_Nsteps = inputs['min_Nsteps']
    except KeyError: pass # If not given, continue indefinitely: min steps huge.
    try: omega_min = inputs['omega_min']
    except KeyError: pass
    try: vel_min = inputs['vel_min']
    except KeyError: pass

    c_log.debug("Stuck %s: Nmin=%g (tmin=%g) avglen %i omega_min=%.4g deg/ms velox_min=%.4g micron/ms" % ('both' if both_breaks else 'single', min_Nsteps, min_Nsteps*dt, avglen, omega_min, vel_min))
    if min_Nsteps < avglen: raise ValueError("Omega/Velocity average length larger them minimum number of steps!")

    # Set pinning config exit
    try: theta_max = inputs['theta_max']+angle # Exit if cluster rotates more than this
    except KeyError: pass
    try: rcm_max = inputs['rcm_max']+npnorm(pos_cm) # Exit if cluster slides more than this
    except KeyError: pass
    try: omega_max = inputs['omega_max'] # Exit if cluster rotates faster than this
    except KeyError: pass
    try: vel_max = inputs['vel_max'] # Exit if cluster rotates more than this
    except KeyError: pass
    c_log.debug("Depin: theta_max=%.4g omega_max=%.4g rcm_max=%.4g vel_max = %.4g" % (theta_max, rcm_max, omega_max, vel_max))

    #-------- LANGEVIN ----------------
    # Assumes rotation and translation indipendent. We just care about the scaling, not exact number.
    eta = 1   # Translational damping of single colloid
    try: eta = inputs['eta']
    except KeyError: pass
    # CM translational viscosity [fKg/ms], CM rotational viscosity [micron^2*fKg/ms]
    etat_eff, etar_eff = calc_cluster_langevin(eta, pos)
    # Aplitude of random numbers. Translational follows nicely from CLT, rotational is assumed from A. E. Filippov, M. Dienwiebel, J. W. M. Frenken, J. Klafter, and M. Urbakh, Phys. Rev. Lett. 100, 046102 (2008).
    brandt, brandr = np.sqrt(2*T*etat_eff), np.sqrt(2*T*etar_eff)
    kBTroom = 4.069767441860465 #zj
    c_log.info("Number of particles %i Eta trasl %.5g Eta tras eff %.5g Eta roto eff %.5g Ratio roto/tras %.5g kBT=%.3g (kBT/kBTroom=%.3g)" % (N, eta, etat_eff, etar_eff, etar_eff/etat_eff, T, T/kBTroom))

    c_log.info("Tau = %.4g fN*micron (Tau/N=%.4g)" % (Tau, Tau/N))
    c_log.info("Fx=%.4g fN (Fx/N=%.4g), Fy=%.4g fN (Fy/N=%.4g), |F| = %.4g fN (|F|/N=%.4g)",
               Fx, Fx/N, Fy, Fy/N, np.sqrt(Fx**2+Fy**2), np.sqrt(Fx**2+Fy**2)/N)
    c_log.info("Omega free=%.4g  Vfree=(%.4g,%.4g) |Vfree|=%.4g" % (Tau/etar_eff, Fx/etat_eff, Fy/etat_eff, np.sqrt(Fx**2+Fy**2)/etat_eff))
    c_log.debug("Free cluster would rotate %.2f deg", Tau/etar_eff * Nsteps * dt)
    c_log.debug("Free cluster would translate %.2f micron", np.sqrt(Fx**2+Fy**2)/etat_eff * Nsteps * dt)
    c_log.debug("Amplitude of random number trasl %.2g and roto %.2g" % (brandt, brandr))

    #-------- INFO FILE ---------------
    with open(info_fname, 'w') as infof:
        infod = {'eta': eta, 'etat_eff': etat_eff, 'etar_eff': etar_eff, 'brandt': brandt, 'brandr': brandr,
                 'N': N, 'theta_max': theta_max, 'print_skip': print_skip,
                 'min_Nsteps': min_Nsteps, 'avglen': avglen, 'omega_min': omega_min, 'omega_max': omega_max,
                 'pos_cm': pos_cm.tolist()
        }
        infod.update(inputs)
        #if debug: c_log.debug("Info dict\n %s" % ("\n".join(["%s: %s" % (k, type(v)) for k, v in infod.items()])))
        json.dump(infod, infof, indent=True)

    #-------- OUTPUT SETUP -----------
    # !! Labels and print_status data structures must be coherent !!
    num_space = 30 # Width printed numerical values
    indlab_space = 2 # Header index width
    lab_space = num_space-indlab_space-1 # Match width of printed number, including parenthesis
    header_labels = ['e_pot', 'pos_cm[0]', 'pos_cm[1]', 'Vcm[0]', 'Vcm[1]',
                     'angle', 'omega', 'forces[0]', 'forces[1]', 'torque']
    # Gnuplot-compatible (leading #) fix-width output file
    first = '#{i:0{ni}d}){s: <{n}}'.format(i=0, s='dt*it', ni=indlab_space, n=lab_space-1,c=' ')
    print(first+"".join(['{i:0{ni}d}){s: <{n}}'.format(i=il+1, s=lab, ni=indlab_space, n=lab_space,c=' ')
                       for il, lab in zip(range(len(header_labels)), header_labels)]), file=outstream)

    # Inner-scope shortcut for printing
    def print_status():
        data = [dt*it, e_pot, pos_cm[0], pos_cm[1], Vcm[0], Vcm[1],
                angle, omega, forces[0], forces[1], torque]
        print("".join(['{n:<{nn}.16g}'.format(n=val, nn=num_space)
                       for val in data]), file=outstream)

    # Print setup time
    t_exec = time()-t0
    c_log.debug("Setup in %is (%.2fmin or %.2fh)", t_exec, t_exec/60, t_exec/3600)

    #-------- START MD ----------------
    t0 = time() # Start clock
    for it in range(Nsteps):

        # ENERGY LANDSCAPE
        e_pot, forces, torque = calc_en_f(pos + pos_cm, pos_cm, *en_params)

        # UPDATE VELOCITIES
        # First order Langevin equation
        noise = normal(0, 1, size=3)
        Vcm = (forces + F + brandt*noise[0:2])/etat_eff
        omega = (torque + Tau + brandr*noise[2])/etar_eff

        # Print progress
        if it % printprog_skip == 0:
            c_log.info("t=%8.3g of %5.2g (%2i%%) E=%10.3g  x=%9.3g y=%9.3g theta=%9.3g omega=%10.3g |Vcm|=%8.3g",
                       it*dt, Nsteps*dt, 100.*it/Nsteps, e_pot, pos_cm[0], pos_cm[1], angle, omega, npnorm(Vcm))
            #c_log.debug("Noise %.2g %.2g %.2g, thermal kick Fxy=(%.2g %.2g) torque=%.2g" % (noise[0],noise[1], noise[2], *(brandt*noise[0:2]),brandr*noise[2]))
            #c_log.debug("Thermal displ scaled (Fx, Fy)/etat=(%.2g %.2g) torque/etar=%.2g" % (*(brandt*noise[0:2]/etat_eff),brandr*noise[2]/etar_eff))
            c_log.debug("Break: v %.5g (vmin %.5g vmax %.5g); omega %.5g (omegamin %.5g omegamax %.5g)" % (npnorm(Vcm), vel_min, vel_max,
                                                                                               omega, omega_min, omega_max))
        # Print step results
        if it % print_skip == 0: print_status()

        # UPDATE DEGREES OF FREEDOM
        # center of mass follows local forces.
        pos_cm += dt * Vcm
        # angle of cluster follows local torque.
        dangle = dt*omega
        angle += dangle
        # positions are further rotated
        pos = rotate(pos,dangle)

        # CHECK FOR STOPPING CONDITIONS
        # Compute omega average and check for exit conditions
        omega_avg += omega # Average omega to check if system is stuck. See avglen.
        vel_avg += npnorm(Vcm) # Average omega to check if system is stuck. See avglen.
        if it % avglen == 0:
            omega_avg /= avglen
            vel_avg /= avglen
            rcm = npnorm(pos_cm)

            # If system is stuck, set flag to exit
            if np.abs(omega_avg) < omega_min and it >= min_Nsteps: break_omega = True
            if vel_avg < vel_min and it >= min_Nsteps: break_V = True

            # If system is rotating or sliding without stopping, set flag to exit
            if (angle >= theta_max or np.abs(omega_avg) >= omega_max) and it >= min_Nsteps: break_omega = True
            if (rcm >= rcm_max or vel_avg >= vel_max) and it >= min_Nsteps: break_V = True

            # Break if either or both condistions are satisfied
            if ((break_omega and break_V) and both_breaks) or ((break_omega or break_V) and not both_breaks):
                # Values
                c_log.info("Rotational: angle=%10.4g <omega>_%i=%10.4g" % (angle, avglen, omega_avg))
                c_log.info("Translationa: rcm=%10.4g <v>_%i=%10.4g" % (rcm, avglen, vel_avg))
                # Pinned check
                if np.abs(omega_avg) < omega_min: c_log.info("System is roto-pinned (omega min=%.4g)." % (omega_min))
                if np.abs(vel_avg) < vel_min: c_log.info("System is trasl-pinned (vel_min=%.4g)." % (vel_min))
                # Depinned check
                if (angle >= theta_max or np.abs(omega_avg) >= omega_max):
                    c_log.info("System is roto-depinned (theta max=%10.4g omega max=%10.4g)" % (theta_max, omega_max))
                if (rcm >= rcm_max or vel_avg >= vel_max):
                    c_log.info("System is trasl-depinned (rcm max=%10.4g vel max=%10.4g)" % (rcm_max, vel_max))
                # Exit conditions
                c_log.info("Breaking condition (%s) satisfied. Exit" % ('both omega and Vcm' if both_breaks else 'single'))
                break

            omega_avg = 0 # Reset average
            vel_avg = 0
    #-----------------------

    # Print last step, if needed
    c_log.info("t=%7.3g of %5.2g (%2i%%) E=%10.7g  x=%9.3g y=%9.3g theta=%9.3g omega=%8.3g |Vcm|=%8.3g",
               it*dt, Nsteps*dt, 100.*it/Nsteps, e_pot, pos_cm[0], pos_cm[1], angle, omega, npnorm(Vcm))
    print_status()

    # Print execution time
    t_exec = time()-t0
    c_log.info("Done in %is (%.2fmin or %.2fh)" % (t_exec, t_exec/60, t_exec/3600))

# Stand-alone scripting
if __name__ == "__main__":
    MD_rigid_rototrasl(sys.argv[1:], debug=True)
