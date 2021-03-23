#!/usr/bin/env python3

# System stuff
import json, sys, logging
from time import time
from tempfile import NamedTemporaryFile
# Science stuff
import numpy as np
import pandas as pd
# Personal things
from MD_rigid_rototrasl import MD_rigid_rototrasl

def driver_Tau_ramp(argv, outstream=sys.stdout, name=None, info_fname=None, logger=None, debug=False):
    """Gradually increase external torque Tau until cluster spins."""

    # We're in a pickle if either the first or last is not spinning. Should do a check and restart with
    # and change the extremal in that case, i.e. halving the start or doubling the end.
    # A simple yet inefficient way around it is to start from small, surely pinned config and go up in
    # small, fixed steps to a huge, definitely depinning Torque, e.g. T=1e30.

    if name == None: name = 'Tau_ramp'
    if info_fname == None: info_fname = "info-%s.json" % name

    #-------- SET UP LOGGER -------------
    if logger == None:
        c_log = logging.getLogger(name) # Set name of the function
        # Adopted format: level - current function name - message. Width is fixed as visual aid.
        logging.basicConfig(format='[%(levelname)5s - %(funcName)10s] %(message)s')
        c_log.setLevel(logging.INFO)
        if debug: c_log.setLevel(logging.DEBUG)
    else:
        c_log = logger

    #-------- READ INPUTS -------------
    if type(argv[0]) == dict: # Inputs passed as python dictionary
        inputs = argv[0]
    elif type(argv[0]) == str: # Inputs passed as path to json file
        with open(argv[0]) as inj:
            inputs = json.load(inj)
    else:
        raise TypeError('Unrecognized input structure (no dict or filename str)', inputs)
    c_log.debug("Input dict \n%s", "\n".join(["%10s: %10s" % (k, str(v)) for k, v in inputs.items()]))

    Tau0, Tau1, dTau = inputs['Tau0'], inputs['Tau1'], inputs['dTau']
    Ntau = np.floor((Tau1-Tau0)/dTau)
    # Mobility of a free cluster and fraction to consider deppinned
    rmobility, rmobility_frac = inputs['rmobility'], inputs['rmobility_frac']

    #-------- READ MD INPUTS -------------
    # Template input for MD_rigid_rototrasl
    if type(inputs['MD_inputs']) == dict:
        MD_inputs = inputs['MD_inputs']
    elif type(inputs['MD_inputs']) == str:
        with open(inputs['MD_inputs']) as inj:
            MD_inputs = json.load(inj)
    else:
        raise TypeError('Unrecognized MD input structure (no dict or filename str)', MD_inputs)

    #-------- SETUP SYSTEM -------------
    # Extend run for different torque N=Nscale/|T|. Set to 0 to use only Nsteps in input dictionary
    Nsteps_scale = inputs['Nsteps_scale']
    Nsteps0 = MD_inputs['Nsteps'] # Always use this as starting point.

    c_log.info("Ramping from tau0=%.2g to tau1=%.2g, steps of %.2g fN*micron (%i steps)" % (Tau0, Tau1, dTau, Ntau))
    c_log.info("Scaling base MD steps N0= (%i) by factor f=%.2g: N'=N0+f/T" % (Nsteps0, Nsteps_scale))
    c_log.info("Stopping if rotational mobility 1/etar_eff=%.4g exiceeds %.4g" % (rmobility, rmobility_frac))

    #-------- INFO FILE ---------------
    with open(info_fname, 'w') as infof:
        infod = {'Ntau': Ntau, 'rmobility': rmobility, 'rmobility_frac': rmobility_frac}
        infod.update(inputs)
        if debug: c_log.debug("Info dict\n %s" % ("\n".join(["%s: %s" % (k, type(v)) for k, v in infod.items()])))
        json.dump(infod, infof, indent=True)


    #-------- OUTPUT HEADER -----------
    # !! Labels and print_status data structures must be coherent !!
    num_space = 30 # Width printed numerical values
    indlab_space = 2 # Header index width
    lab_space = num_space-indlab_space-1 # Match width of printed number, including parenthesis
    header_labels = ['Fx', 'Fy', 'omega', 'V_cm[0]', 'V_cm[1]', 'theta', 'pos_cm[0]', 'pos_cm[1]']
    # Gnuplot-compatible (leading #) fix-width output file
    first = '#{i:0{ni}d}){s: <{n}}'.format(i=0, s='T', ni=indlab_space, n=lab_space-1,c=' ')
    print(first+"".join(['{i:0{ni}d}){s: <{n}}'.format(i=il+1, s=lab, ni=indlab_space, n=lab_space,c=' ')
                       for il, lab in zip(range(len(header_labels)), header_labels)]), file=outstream)

    # Inner-scope shortcut for printing
    def print_status():
        data = [T, Fx, Fy, omega_fin, Vx_fin, Vy_fin, theta_fin, *pos_cm_fin]
        c_log.debug('T Fx Fy omega Vx Vy theta pos_cm[0] pos_cm[1]')
        c_log.debug(data)
        c_log.debug([type(d) for d in data])
        print("".join(['{n:<{nn}.16g}'.format(n=val, nn=num_space)
                       for val in data]), file=outstream)

    #-------- RAMP UP! -----------------
    t0 = time()
    it = 0

    for T in np.arange(Tau0, Tau1, dTau):

        MD_inputs['T'] = float(T) # Json doesn't like <class 'numpy.int64'>

        # Extend MD steps according to current torque
        N_extra = Nsteps_scale/np.abs(T)
        MD_inputs['Nsteps'] = int(Nsteps0 + N_extra)
        omega_threshold = np.abs(T)*rmobility*rmobility_frac # Stop if |omega| exceeds fraction (e.g. 10%) of free cluster
        MD_inputs['omega_max'] =  omega_threshold
        if MD_inputs['omega_min'] > MD_inputs['omega_max']:
            c_log.warning("Breaking omegamin (%.3g) > omegamax (%.3g)" % (MD_inputs['omega_min'], MD_inputs['omega_max']))

        # ------ RUN  -------------
        c_log.info("On T=%.4g (%2i%%). omega_free/10=%.3g. Start run of %i steps" % (T, 100*it/Ntau, omega_threshold, MD_inputs['Nsteps']))
        c_log.info("-----------------------------------------------------") # Separate MD runs infos a bit

        c_outf = "out-%s-T_%.4g.dat" % (inputs['out_bname'], T)
        c_outinfo = "info-%s-T_%.4g.json" % (inputs['out_bname'], T)
        c_posfile = 'pos-%s.dat' % inputs['out_bname']
        with open(c_outf, 'w') as c_out:
            MD_rigid_rototrasl([MD_inputs], outstream=c_out, info_fname=c_outinfo, pos_fname=c_posfile, logger=c_log, debug=debug)

        # ------ GET stationary Omega(T) ------
        data = pd.read_fwf(c_outf, infer_nrows=1e30) # Pandas has the tendency of underestimate column width
        tail_len = 20 # Average over this prints. Omega oscillates in a spinning config, so this is not 100% accurate.
        # Careful with the labels here, if they change, this breaks
        pos_cm_fin = np.reshape([data['02)pos_cm[0]'].tail(1), data['03)pos_cm[1]'].tail(1)], newshape=(2))
        theta_fin, omega_fin = data['06)angle'].tail(1).mean(), data['07)omega'].tail(tail_len).mean()
        Vx_fin, Vy_fin = data['04)Vcm[0]'].tail(1).mean(), data['05)Vcm[1]'].tail(tail_len).mean()
        V_fin = np.linalg.norm([Vx_fin, Vy_fin])
        Rcm_fin = np.linalg.norm(pos_cm_fin)
        print_status()
        outstream.flush() # Print progress as you go.

        if np.abs(omega_fin) > np.abs(omega_threshold):
            c_log.info("Omega (%.4g) above treshold (%.4g). Exit." % (omega_fin, omega_threshold))
            break

        # ------ UPDATE MD INPUTS ------
        # Update angle and CM pos for next run. Only variable determining the sys in Overdamped Langevin.
        MD_inputs['angle'] = float(theta_fin)
        c_log.debug(pos_cm_fin)
        MD_inputs['pos_cm'] = pos_cm_fin.tolist() # Json doesn't like numpy arrays

        it += 1
        c_log.info("-----------------------------------------------------\n") # Separate MD runs infos a bit
    #------------------------------

    # Print execution time
    t_exec = time()-t0
    c_log.info("Done in %is (%.2fmin or %.2fh)" % (t_exec, t_exec/60, t_exec/3600))
    c_log.info("#####################################################\n") # Separate Taus infos a bit

    if np.abs(omega_fin) < omega_threshold:
        c_log.warning("Omega (%.4g) never overcame threshold (%.4g)!" % (omega_fin, omega_threshold))
        return 1

    return 0

if __name__ == "__main__":
    driver_Tau_ramp(sys.argv[1:], debug=False)
