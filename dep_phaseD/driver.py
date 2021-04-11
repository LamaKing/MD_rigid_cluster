#!/usr/bin/env python3

# System stuff
import json, sys, logging, os
from functools import partial
from time import time
from tempfile import NamedTemporaryFile
import multiprocessing
# Science stuff
import numpy as np
from numpy.linalg import norm as npnorm
import pandas as pd
# Model source
from tool_create_cluster import rotate, create_cluster, calc_cluster_langevin
from tool_create_substrate import calc_matrices_square, calc_matrices_triangle, calc_en_tan, calc_en_gaussian, gaussian
from MD_rigid_rototrasl import MD_rigid_rototrasl

def ramp_F_fixTau(driving_FsT, MD_inputs, ramp_inputs,
                  name=None, debug=False):
    """        """

    t0 = time()

    # Unpack esternal forces
    F_range, Tau = driving_FsT
    NF = len(F_range)

    if name == None: name = 'ramp_F-Tau_%.4g' % Tau

    #-------- SET UP LOGGER -------------
    # For this threads and children
    c_log = logging.getLogger(name)
    c_log.setLevel(logging.INFO)
    if debug: c_log.setLevel(logging.DEBUG)
    # Adopted format: level - current function name - message. Width is fixed as visual aid.
    log_format = logging.Formatter('[%(levelname)5s - %(funcName)10s] %(message)s')
    console = open('console-%s.log' %name, 'w')
    handler = logging.StreamHandler(console)
    handler.setFormatter(log_format)
    c_log.addHandler(handler)

    #-------- READ INPUTS -------------
    if type(MD_inputs) == str: # Inputs passed as path to json file instead of dictionary
        with open(MD_inputs) as inj:
            MD_inputs = json.load(inj)
    else:
        MD_inputs = MD_inputs.copy() # Get a copy so all threads can edit
    MD_inputs['Tau'] = Tau # Set current torque in all MD of this process
    Nsteps = MD_inputs['Nsteps']
    # First run might be longer
    Nsteps0 = Nsteps # Default
    try:
        Nsteps0 = ramp_inputs['Nsteps0'] # Update if found
        c_log.debug('Update Nstep0: %i -> %i' % (Nsteps, Nsteps0))
    except KeyError: pass
    # Angle of the driving force in xy
    Fphi = ramp_inputs['Fphi']
    # To be extra careful, make MD breking conditions stricter
    MD_min_mobfrac, MD_max_mobfrac = 0.08, 0.5

    iF = 0
    for F in F_range:
        c_log.info("On F=%.3g (%i of %i %.2f%%)" % (F, iF, NF, iF/NF*100))

        if F==0 and Tau==0:
            c_log.info("Skip not driven config")
            continue

        # Adjust Nsteps
        if iF == 0: MD_inputs['Nsteps'] = Nsteps0
        else: MD_inputs['Nsteps'] = Nsteps

        # Set min/max vel and omega to exit MD run at current torque and force
        MD_inputs['vel_min'], MD_inputs['vel_max'] = F*tmob0*MD_min_mobfrac, F*tmob0*MD_max_mobfrac
        MD_inputs['omega_min'], MD_inputs['omega_max'] = Tau*rmob0*MD_min_mobfrac, Tau*rmob0*MD_max_mobfrac
        # Set correct logical function to check break
        if F == 0:
            c_log.debug("Check roto dep only")
            break_check = lambda rmob, tmob: rmob > rmob0*rmob_frac
            MD_inputs['vel_min'], MD_inputs['vel_max'] = -1, 1e30
        elif Tau == 0:
            c_log.debug("Check trasl dep only")
            break_check = lambda rmob, tmob: tmob > tmob0*tmob_frac
            MD_inputs['omega_min'], MD_inputs['omega_max'] = -1, 1e30
        else:
            c_log.debug("Check both dep")
            break_check = lambda rmob, tmob: np.logical_or(rmob > rmob0*rmob_frac, tmob > tmob0*tmob_frac)

        # Set current force in MD inputs
        MD_inputs['Fx'], MD_inputs['Fy'] = F*np.cos(Fphi*np.pi/180), F*np.sin(Fphi*np.pi/180)

        # ------ Start MD ------
        c_log.propagate = False # Don't print update in driver logger
        c_log.info('-'*50)
        c_outf = "out-%s-F_%.4g.dat" % (name, F)
        c_outinfo = "info-%s-F_%.4g.json" % (name, F)
        with open(c_outf, 'w') as c_out:
            MD_rigid_rototrasl([MD_inputs], outstream=c_out, info_fname=c_outinfo, logger=c_log, debug=debug)

        c_log.propagate = True

        # ------ Get stationary Omega(T) ------
        pos_cm0, th0 = MD_inputs['pos_cm'], MD_inputs['angle']
        Rcm0 = npnorm(pos_cm0)
        data = pd.read_fwf(c_outf, infer_nrows=1e30) # Pandas has the tendency of underestimate column width
        tail_len = 100 # Average over this prints. Oscillates in a depinned config makes this tricky.
        # Careful with the labels here, if they change, this breaks. Weird syntax needed from Panda. Am I doing something wrong?
        pos_cm1 = np.reshape([data['02)pos_cm[0]'].tail(1), data['03)pos_cm[1]'].tail(1)], newshape=(2))
        Rcm1 = npnorm(pos_cm1)
        th1, omega1 = data['06)angle'].tail(1).mean(), data['07)omega'].tail(tail_len).mean()
        Vx1, Vy1 = data['04)Vcm[0]'].tail(1).mean(), data['05)Vcm[1]'].tail(tail_len).mean()
        V1 = npnorm([Vx1, Vy1])

        rmob, tmob = omega1/Tau, V1/F
        c_log.debug("rmob %10.4g (thold %10.4g, break: %s) tmob %10.4g (thold %10.4g, break: %s)." % (rmob, rmob0*rmob_frac, rmob > rmob0*rmob_frac,
                                                                                                    tmob, tmob0*tmob_frac, tmob > tmob0*tmob_frac))
        c_log.debug("Break condition %s" % break_check(rmob, tmob))

        if break_check(rmob, tmob):
            c_log.info("Above treshold. Exit.")
            if (Rcm1-Rcm0 < MD_inputs['R']):
                c_log.warning('Cluster tranlate less than R: %.3g < %.3g' % (Rcm1-Rcm0, MD_inputs['R']))
            th_warn = 3 # [deg] Arbitrary
            if (th1-th0 < th_warn):
                c_log.warning('Cluster rotated: %.3g < %.3g' % (th1-th0,th_warn))
            break

        # ------ UPDATE MD INPUTS ------
        # Update angle and CM pos for next run. Only variable determining the sys config in Overdamped Langevin.
        MD_inputs['angle'] = float(th1)
        c_log.debug(pos_cm1)
        MD_inputs['pos_cm'] = pos_cm1.tolist() # Json doesn't like numpy arrays
        iF += 1

    # Print execution time
    t_exec = time()-t0
    c_log.info("Done in %is (%.2fmin or %.2fh)" % (t_exec, t_exec/60, t_exec/3600))

    console.close()
    return 0

if __name__ == '__main__':

    t0 = time()
    debug = True

    #-------- SET UP LOGGER -------------
    c_log = logging.getLogger('driver') # Set name identifying the logger.
    # Adopted format: level - current function name - message. Width is fixed as visual aid.
    logging.basicConfig(format='[%(levelname)5s - %(name)15s] %(message)s')
    c_log.setLevel(logging.INFO)
    if debug: c_log.setLevel(logging.DEBUG)

    with open(sys.argv[1]) as inj:
        MD_inputs = json.load(inj)

    # Cluster
    pos = create_cluster(MD_inputs['cluster_hex'])
    N = pos.shape[0]

    Tau0, Tau1, nTau = float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
    F0, F1, nF = float(sys.argv[5]), float(sys.argv[6]), int(sys.argv[7])
    Tau_range = np.linspace(Tau0, Tau1, nTau, endpoint=True)
    F_range = np.linspace(F0, F1, nF, endpoint=True)
    Fphi = 0
    try: Fphi = float(sys.argv[8])
    except: pass

    c_log.info("Grid: increase forces (%i) at fix torques (%i). Force angle %.3g" % (len(F_range), len(Tau_range), Fphi))
    # Set up system for multiprocess
    ncpu = os.cpu_count()
    nworkers = 1
    try: nworkers = int(sys.argv[9])
    except: pass
    c_log.info("Running %i elements on %i processes  (%i-core machine)" % (len(Tau_range), nworkers, ncpu))

    #-------- MOBILITIES -------------
    eta = 1   # Translational damping of single colloid
    etat_eff, etar_eff = calc_cluster_langevin(eta, pos)
    # Free cluster under a torque T moves at omega = T/etar_eff. Define 1/etar_eff as "roto-mobility" of the free cluster
    # Same for translation
    rmob0, tmob0 = 1/etar_eff, 1/etat_eff
    # Fractions to consider cluster roto/trasl depinned
    rmob_frac, tmob_frac = 0.2, 0.2
    c_log.info("Roto_mob0=%.3g (thold: %.3g) trasl_mob0=%.3g (thold: %.3g)" % (rmob0, rmob0*rmob_frac, tmob0, tmob0*tmob_frac))

    ramp_inputs={'rmob0': rmob0, 'rmob_frac': rmob_frac,
                 'tmob0': tmob0, 'tmob_frac': tmob_frac,
                 'Fphi': Fphi, 'Nsteps0': 2*MD_inputs['Nsteps']}

    # Info dictionary for post-processing
    with open('info-driver-N_%i.json' % N, 'w') as infof:
        infod = {'F_range': F_range.tolist(), 'Tau_range': Tau_range.tolist(), 'N': N, 'eta': eta, 'MD_inputs': MD_inputs}
        infod.update(ramp_inputs)
        json.dump(infod, infof, indent=True)

    # Auxiliary function with single positional argument, required by pool
    p_rampF = partial(ramp_F_fixTau,
                      MD_inputs=MD_inputs,ramp_inputs = ramp_inputs, **{'debug': debug})

    # Launch all simulations with Pool of workers
    pool = multiprocessing.Pool(processes=nworkers)
    results = pool.map(p_rampF, [(F_range, Tau) for Tau in Tau_range])
    pool.close() # Pool doesn't accept new jobs
    pool.join() # Wait for all processes to finish
    c_log.debug("Results: %s" % str(results))

    # Print execution time
    t_exec = time()-t0
    c_log.info("Done in %is (%.2fmin or %.2fh)" % (t_exec, t_exec/60, t_exec/3600))
