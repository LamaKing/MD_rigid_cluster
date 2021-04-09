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

def driver_F_ramp(argv, outstream=sys.stdout, info_fname='info-rampF.json', logger=None, debug=False):
    """Gradually increase external force F=(Fx,Fy) until cluster slides."""

    #-------- SET UP LOGGER -------------
    if logger == None:
        c_log = logging.getLogger("driver_F_ramp") # Set name of the function
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
    c_log.info("Input dict \n%s", "\n".join(["%10s: %10s" % (k, str(v)) for k, v in inputs.items()]))

    Fx0, Fx1, dFx = inputs['Fx0'], inputs['Fx1'], inputs['dFx']
    Fy0, Fy1, dFy = inputs['Fy0'], inputs['Fy1'], inputs['dFy']
    NFx = np.floor((Fx1-Fx0)/dFx)
    NFy = np.floor((Fy1-Fy0)/dFy)
    # Mobility of a free cluster and fraction to consider deppinned
    tmobility, tmobility_frac = inputs['tmobility'], inputs['tmobility_frac']

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

    c_log.info("Ramping from Fx0=%.2g to Fx1=%.2g, steps of %.2g fN (%i steps)" % (Fx0, Fx1, dFx, NFx))
    c_log.info("Ramping from Fy0=%.2g to Fy1=%.2g, steps of %.2g fN (%i steps)" % (Fy0, Fy1, dFy, NFy))
    c_log.info("Scaling base MD steps N0= (%i) by factor f=%.2g: N'=N0+f/T" % (Nsteps0, Nsteps_scale))
    c_log.info("Stopping if translational mobility 1/etar_eff=%.4g exiceeds %.4g" % (tmobility, tmobility_frac))

    #-------- INFO FILE ---------------
    with open(info_fname, 'w') as infof:
        infod = {'NFx': NFx, 'NFy': NFy, 'tmobility': tmobility, 'tmobility_frac': tmobility_frac}
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

    for Fy in np.arange(Fy0, Fy1, dFy):
        for Fx in np.arange(Fx0, Fx1, dFx):

            MD_inputs['Fx'] = float(Fx) # Json doesn't like <class 'numpy.int64'>
            MD_inputs['Fy'] = float(Fy) # Json doesn't like <class 'numpy.int64'>

            # Extend MD steps according to current torque
            N_extra = Nsteps_scale/np.sqrt(Fx**2+Fy**2)
            MD_inputs['Nsteps'] = int(Nsteps0 + N_extra)
            vel_threshold = np.sqrt(Fx**2+Fy**2)*tmobility*tmobility_frac # Stop if |vel| exceeds given fraction of free cluster
            MD_inputs['vel_max'] =  vel_threshold
            if MD_inputs['vel_min'] > MD_inputs['vel_max']:
                c_log.warning("Breaking vel_min (%.3g) > vel_max (%.3g)" % (MD_inputs['vel_min'], MD_inputs['vel_max']))

            # ------ RUN  -------------
            c_log.info("On Fx=%.4g Fy=%.4g  (%2i%%). vel_free/10=%.3g. Start run of %i steps" % (Fx, Fy, 100*it/NFx/NFy,
                                                                                                 vel_threshold, MD_inputs['Nsteps']))
            c_log.info("-----------------------------------------------------") # Separate MD runs infos a bit

            c_outf = "out-%s-Fx_%.4g-Fy_%.4g.dat" % (inputs['out_bname'], Fx, Fy)
            c_outinfo = "info-%s-Fx_%.4g-Fy_%.4g.json" % (inputs['out_bname'], Fx, Fy)
            c_posfile = 'pos-%s.dat' % inputs['out_bname']
            with open(c_outf, 'w') as c_out:
                MD_rigid_rototrasl([MD_inputs], outstream=c_out, info_fname=c_outinfo, pos_fname=c_posfile, logger=c_log, debug=debug)

            # ------ GET stationary Omega(T) ------
            data = pd.read_fwf(c_outf, infer_nrows=1e30) # Pandas has the tendency of underestimate column width
            tail_len = 20 # Average over this prints.
            # Careful with the labels here, if they change, this breaks
            pos_cm_fin = np.reshape([data['02)pos_cm[0]'].tail(1), data['03)pos_cm[1]'].tail(1)], newshape=(2))
            theta_fin, omega_fin = data['06)angle'].tail(1).mean(), data['07)omega'].tail(tail_len).mean()
            Vx_fin, Vy_fin = data['04)Vcm[0]'].tail(1).mean(), data['05)Vcm[1]'].tail(tail_len).mean()
            V_fin = np.linalg.norm([Vx_fin, Vy_fin])
            Rcm_fin = np.linalg.norm(pos_cm_fin)
            print_status()
            outstream.flush() # Print progress as you go.

            if V_fin > np.abs(vel_threshold):
                c_log.info("|Vcm| (%.4g) above treshold (%.4g). Exit." % (V_fin , vel_threshold))
                break

            # ------ UPDATE MD INPUTS ------
            # Update angle and CM pos for next run. Only variable determining the sys in Overdamped Langevin.
            MD_inputs['angle'] = float(theta_fin)
            MD_inputs['pos_cm'] = pos_cm_fin.tolist() # Json doesn't like numpy arrays

            it += 1
            c_log.info("-----------------------------------------------------\n") # Separate MD runs infos a bit
    #------------------------------

    # Print execution time
    t_exec = time()-t0
    c_log.info("Done in %is (%.2fmin or %.2fh)" % (t_exec, t_exec/60, t_exec/3600))
    c_log.info("#####################################################\n") # Separate Taus infos a bit

    if np.abs(V_fin) < vel_threshold:
        c_log.warning("|Vcm| (%.4g) never overcame threshold (%.4g)!" % (V_fin, vel_threshold))
        return 1

    return 0

if __name__ == "__main__":
    driver_F_ramp(sys.argv[1:], debug=False)
