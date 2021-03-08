#!/usr/bin/env python3

# System stuff
import json, sys, logging, os
from functools import partial
from time import time
from tempfile import NamedTemporaryFile
import multiprocessing
# Science stuff
import numpy as np
import pandas as pd
# Personal things
from tool_create_circles import create_cluster_circle
from tool_create_hexagons import create_cluster_hex
from driver_T_ramping import driver_Tau_ramp

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

def sizeNl_Taucrit(Nl, TauN_range, MD_inputs, run_simul=True, run_anal=True,
                   clt_shape = 'hexagon', angle=0, theta1=30, theta_min='auto', rmobility_frac=0.1,
                   outfname=None, name=None, debug=False):
        """Launch T ramp with given geometry and calclate Tau crit from results.

        Limits of Tau range to explore are given per particle and tranformed within this function.
        """

        if name == None:
            name = 'Nl_%i_Taucrit' % Nl

        if outfname == None:
            outfname = 'Nl_%i_Taucrit.dat' % Nl

        #-------- SET UP LOGGER -------------
        # For this threads and children
        c_log = logging.getLogger(name)
        c_log.setLevel(logging.INFO)
        if debug: c_log.setLevel(logging.DEBUG)
        # Adopted format: level - current function name - message. Width is fixed as visual aid.
        log_format = logging.Formatter('[%(levelname)5s - %(funcName)10s] %(message)s')
        console = open('console-Nl_%i.log' % Nl, 'w')
        handler = logging.StreamHandler(console)
        handler.setFormatter(log_format)
        c_log.addHandler(handler)

        #-------- READ INPUTS -------------
        if type(MD_inputs) == str: # Inputs passed as path to json file instead of dictionary
            with open(MD_inputs) as inj:
                MD_inputs = json.load(inj)
        else:
            MD_inputs = MD_inputs.copy() # Get a copy so all threads can edit

        # ------ CLUSTER ----------
        Rcl = 4.45 # Lattice spacing of cluster. Fixed by experiments.
        # define cluster shape
        if clt_shape == 'circle':
            create_cluster = create_cluster_circle
        elif clt_shape == 'hexagon':
            create_cluster = create_cluster_hex
        else:
            raise ValueError("Shape %s not implemented" % clt_shape)
        a1 = np.array([Rcl, 0])
        a2 = np.array([-Rcl/2, Rcl*np.sqrt(3)/2])
        clgeom_fname = "input_hex-Nl_%.i.hex" % Nl
        pos, _ = cluster_inhex_Nl(Nl, Nl, a1=a1, a2=a2, clgeom_fname=clgeom_fname, cluster_f=create_cluster)
        N = pos.shape[0]
        MD_inputs['cluster_hex'] = clgeom_fname
        #MD_inputs['angle'] = np.arctan(a/Rcl/(Nl-1))*180/np.pi # Put yourself near the end of flat bit in commensurate case.
        MD_inputs['angle'] = angle
        c_log.info("%s cluster size %i (Nl %i). Starting at angle %.2g" % (clt_shape, N, Nl, MD_inputs['angle']))

        # ------ LANGEVIN ---------
        # Be sure it's the same as in MD_rigid_rototrasl
        eta = 1 # [fKg/ms]
        # CM translational viscosity
        etat_eff = eta*N # [fKg/ms]
        # CM rotational viscosity.
        etar_eff = eta*N**2 # [micron^2*fKg/ms]

        # ------ DRIVER INPUT -----
        #if Tau_range == 'auto':
        #    # Check static maps to get an idea. Can't scale faster than sqrt(N), only multiplier prefactor offset
        #    Tau0, Tau1 = offset/2*N, 2*offset*N**(3/2)
        #    # Set uncertainty on dTau...
        #    #dTau = 1*N/5 # Allowed uncertainty on Tcritic
        #    #Ntau = int((Tau1-Tau0)/dTau)
        #    #if Ntau > 5e3: c_log.warning("realtively large Tau steps %i" % Ntau)
        #    # ... or total number of Tau steps
        #    Ntau = 300
        #    dTau = (Tau1-Tau0)/Ntau
        #else:
        #    Tau0, Tau1, dTau = Tau_range
        Tau0, Tau1, dTau = N*np.array(TauN_range)
        c_log.debug("Tau range %.2f %.2f %.2f" % (Tau0, Tau1, dTau))

        # Free cluster under a torque T moves at omega = T/etar_eff
        # Define 1/etar_eff as "roto-mobility" of the free cluster
        rmobility_free = 1/etar_eff

        # Free cluster would rotate this much at given Tau
        Nsteps_scale = theta1*etar_eff/MD_inputs['dt'] # Add to base Nsteps: Nsteps_scale/T. Decrease for larger torques.
        c_log.info("Base run is %i steps + Nsteps(T) for free cluster to rotate %.2f deg" % (MD_inputs['Nsteps'], theta1))

        # If it start by rotatinig freerly, let first simul go at least angular width of well in commensurate case tan(th)=a/((Nl-1)*R)
        # !! CAREFUL WITH DIRECTIONAL LOCKING !!
        if theta_min == 'auto':
            theta_min = np.arctan((2*MD_inputs['a'])/MD_inputs['R']/(Nl-1))*180/np.pi
        MD_inputs['min_Nsteps'] = theta_min*etar_eff/MD_inputs['dt']/Tau0
        c_log.info("Minimum number of steps (%i): free cluster under Tau=%.2g rotates of %.2f deg" % (MD_inputs['min_Nsteps'], Tau0, theta_min))

        # Input dict for driver Tau ramp function
        inputs = {
            'Nsteps_scale': Nsteps_scale,
            'Tau0': Tau0,
            'Tau1': Tau1,
            'dTau': dTau,
            'rmobility': rmobility_free,
            'rmobility_frac': rmobility_frac,
            'out_bname': "N_%i" % N,
            'MD_inputs': MD_inputs,
        }

        out_fname = "Tau_ramp-N_%i.dat" % N
        # ------ LAUNCH TAU RAMP --
        if run_simul:
            c_log.info("Running MD. Turn off logger propagate, see specific console given")
            c_log.propagate = False
            # TODO: SOLVE PROBLEM OF NOT KNOWING WHERE THE ANGLE MINIUM IS (DIRLOCK).
            # Sol 1
            #   FIRST RUN A LONG SIMUL AT MINIMUM TAU, TO SEE WHERE MINIMUM ANGLE IS. THEN START RAMPING UP. OTHERWISE YOU WILL HAVE SUPER LONG SIMUL FOR ALL, NO?
            # Sol 2
            #   USE A LESS STRICT OMEGA MAX. INSTEAD OF ALREADY 10%, PUT IT AT 90%. IF IT'S STUCK, IT WILL EXIT, IF IT KEEPS ROTATING, WILL FINISH THE SIMULATION...
            #   BUT THEN ALL TAU RANGE WOULD BE EXPLORED.
            #   LET'S THINK ABOUT IT TOMORROW...
            #
            # The brute force, but probably most stable solution is to force the a long "equilibration/exloration" initial step. Keep mobility thold at 0.1, but force
            # the minimum number of steps to be enough for a free cluster to rotate of 15deg, enought to encounter any barrier...

            with open(out_fname, 'w') as out_data:
                driver_Tau_ramp([inputs], outstream=out_data, info_fname='info-rampTau-N_%i.json' % N, logger=c_log, debug=debug)

            c_log.propagate = True
            c_log.info("MD done, turn on logging propagate.")

        # ------ ESTIMATE TAU C ---
        if run_anal:
            c_log.debug("Run analysis")
            rmobility_free, rmobility_frac = inputs['rmobility'], inputs['rmobility_frac']
            # Read data, output from above section
            data = pd.read_fwf(out_fname, infer_nrows=1e30) # Load output file
            omega, Tau = data['01)omega'].to_numpy(), data['#00)T'].to_numpy() # Converto to array for convenience
            rmobility = omega/Tau
            # Always unpinned
            if rmobility[0]/rmobility_free > rmobility_frac:
                c_log.warning('First config is unpinned: rmobil (%.3g) > rmobil_free*frac (%.3g*%.3g)' % (rmobility[0], rmobility_free, rmobility_frac))
                Tauc = 0
                err_Tauc = Tau[0]
            # Alwasy pinned
            elif rmobility[-1]/rmobility_free < rmobility_frac:
                c_log.warning('Last config is pinned: rmobil (%.3g) < rmobil_free*frac (%3.g*%.3g)' % (rmobility[-1], rmobility_free, rmobility_frac))
                Tauc = Tau[-1] * 2 # !!! ABSOLUTELY ARBITRARY !!!
                err_Tauc = Tau[-1]*1.1 # Lower bound must be higher than last force measured
            else:
                # Linear interpolatee between last-pinned and first-depinned
                c_log.debug("Interpolating")
                rotating_mask = rmobility > rmobility_free*rmobility_frac  # Divide spinning and non-spinning
                y2, y1 = rmobility[rotating_mask][0], rmobility[~rotating_mask][-1] # last sub-critical and first over-critical rmobility
                x2, x1 = Tau[rotating_mask][0], Tau[~rotating_mask][-1] # last sub-critical and first over-critical torquee
                m = (y2-y1)/(x2-x1) # Slope
                Tauc = x1 + (rmobility_frac*rmobility_free-y1)/m # Tau at critical rmobility
                err_Tauc = (x2-x1)/2 # Uncertanty as half the distance between pinned and unpinned.

            # Print size, bravais size, Tc, sigma_Tc, omega
            results = N, Nl, Tauc, err_Tauc, float(omega[-1])
            c_log.info(" ".join("%25.20g" % r for r in results))
            outstream.flush()
        #------------------------------
        console.close()
        return results

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

    # Substrate
    Rcl = 4.45 # Lattice spacing of cluster. Fixed by experiments.
    MD_inputs['epsilon'] = 105
    MD_inputs['R'] = Rcl*np.sqrt(7)/2
    c_log.info("%s substrate: R=%.6g (bottom a=%.2g end curved b=%.2g) depth eps=%.4g" % tuple([MD_inputs[k]
                                                                                                for k in ['sub_symm', 'R', 'a', 'b', 'epsilon']]))
    # MD params
    MD_inputs['Nsteps'] = 1e6
    MD_inputs['omega_min'] = 1e-10
    MD_inputs['theta_max'] = 60 # If you rotate this far, you won't really stop, cluster symmetry.
    theta_min = 10 # 'auto'
    angle = 0

    # Set up system for multiprocess
    ncpu = os.cpu_count()
    nworkers = 2
    Nl_range = range(2,10) # Sizes to explore, in parallels
    c_log.info("Running %i elements on %i processes on %i cores" % (len(Nl_range), nworkers, ncpu))

    # Fix the all arguments a part from Nl, so that we can use pool.map
    # Collect results as they come in. Backup in case we don't reach the ordered bit.
    tau0, tau1, ntau = 0.2, 10, 100 # Check static results to choose a meaningful range
    partial_sizeNl_Taucrit = partial(sizeNl_Taucrit,
                                     TauN_range=[tau0, tau1, (tau1-tau0)/ntau], MD_inputs=MD_inputs,
                                     **{'run_simul': True, 'run_anal': True,
                                        'theta_min': theta_min, # set min N steps so that free cluster would rotate this much
                                        'angle': angle, 'rmobility_frac': 0.1, # defintion of "unpinned" config
                                        'debug':debug})

    # Launch all simulations with Pool of workers
    c_log.debug("Starting pool")
    pool = multiprocessing.Pool(processes=nworkers)
    c_log.debug("Mapping processes")
    results = pool.map(partial_sizeNl_Taucrit, Nl_range)
    c_log.debug("Results: %s" % str(results))

    c_log.debug("Close pool and wait for joining")
    pool.close() # Pool doesn't accept new jobs
    pool.join() # Wait for all processes to finish
    out_async.close() # Processes are done now

    # Print ordered results
    print(" ".join(["%25s" % header for header in ['#N', 'Nl', 'Tauc', 'err_Tauc', 'omega_fin']]), file=sys.stdout)
    for l in results:
        print(" ".join(["%25.20g" % f for f in l]), file=sys.stdout)

    # Print execution time
    t_exec = time()-t0
    c_log.info("Done in %is (%.2fmin or %.2fh)" % (t_exec, t_exec/60, t_exec/3600))
