#!/usr/bin/env python

import sys, os, json, logging, multiprocessing
import numpy as np
from time import time
from functools import partial
from tool_create_cluster import create_cluster_hex, create_cluster_circle, rotate, cluster_inhex_Nl
from tool_create_substrate import gaussian, calc_matrices_triangle, calc_matrices_square, calc_en_gaussian, calc_en_tan
from string_method import PotentialPathAnalyt, Path

def static_barrier_Nl(Nl, inputs, calc_en_f, name=None, out_fname=None, info_fname=None, debug=False):

    if name == None:
        name = 'traslbar_Nl_%i' % Nl
    if out_fname == None: out_fname = 'out-%s.dat' % name
    if info_fname == None: info_fname = 'info-%s.json' % name

    #-------- SET UP LOGGER -------------
    # For this threads and children
    c_log = logging.getLogger(name)
    c_log.setLevel(logging.INFO)
    if debug: c_log.setLevel(logging.DEBUG)
    # Adopted format: level - current function name - message. Width is fixed as visual aid.
    log_format = logging.Formatter('[%(levelname)5s - %(funcName)10s] %(message)s')
    console = open('console-%s.log' % name, 'w')
    handler = logging.StreamHandler(console)
    handler.setFormatter(log_format)
    c_log.addHandler(handler)

    #-------- READ INPUTS -------------
    if type(inputs) == str: # Inputs passed as path to json file
        with open(inputs) as inj:
            inputs = json.load(inj)
    else:
        inputs = inputs.copy() # Copy it so multiple threads can access it
    c_log.debug("Input dict \n%s", "\n".join(["%10s: %10s" % (k, str(v)) for k, v in inputs.items()]))

    # Cluster
    clt_shape = inputs['cluster_shape'] # Select function associate to different cluster shape. Now: circle or hexagon
    # Rotations of cluster
    angle = inputs['angle'] # [deg]

    # ------ CLUSTER ----------
    Rcl = 4.45 # Lattice spacing of cluster. Fixed by experiments.
    # define cluster shape
    if clt_shape == 'circle':
        create_cluster = create_cluster_circle
    elif clt_shape == 'hexagon':
        create_cluster = create_cluster_hex
    else:
        raise ValueError("Shape %s not implemented" % clt_shape)
    # Initial shift
    try:
        pos_cm = np.array(inputs['pos_cm'], dtype=float) # Start pos [micron]
    except KeyError:
        pos_cm = np.zeros(2, dtype=float) # If not given, start from centre
    # Cluster symmetry fixed by experiments
    a1 = np.array([Rcl, 0])
    a2 = np.array([-Rcl/2, Rcl*np.sqrt(3)/2])
    clgeom_fname = "input_hex-Nl_%.i.hex" % Nl
    pos, _ = cluster_inhex_Nl(Nl, Nl, a1=a1, a2=a2, clgeom_fname=clgeom_fname, cluster_f=create_cluster)
    pos = rotate(pos, angle) + pos_cm
    N = pos.shape[0]
    c_log.info("%s cluster size %i (Nl %i) at angle %.3g deg" % (clt_shape, N, Nl, angle))
    inputs['cluster_hex'] = clgeom_fname

    # initialise variable
    try:
        pos_cm = np.array(inputs['pos_cm']) # Start pos [micron]
    except KeyError:
        pos_cm = np.zeros(2) # If not given, start from centre
    forces = np.zeros(2)
    torque = 0

    Nsteps = 3000
    Npt = 100                          # number of subdivisions of the path connecting a and b
    L = Path(inputs['p0'], inputs['p1'], pos, Npt, fix_ends=False)               # initalise the path
    V = PotentialPathAnalyt(L, calc_en_f, en_params)      # potential along the path
    c_log.info("Relax string of %i points in %i stesp" % (Npt, Nsteps))

    #-------- INFO FILE ---------------
    with open(info_fname, 'w') as infof:
        infod = {'N': N, 'Nsteps': Nsteps, 'path_phi': phi, 'Npt': Npt}
        infod.update(inputs)
        if debug: c_log.debug("Info dict\n %s" % ("\n".join(["%s: %s" % (k, type(v)) for k, v in infod.items()])))
        json.dump(infod, infof, indent=True)

    #-------- OUTPUT SETUP -----------
    print_skip = 1 # Timesteps to skip between prints
    try:
       print_skip = inputs['print_skip']
    except KeyError:
        pass
    printerr_skip = int(Nsteps/10)
    outstream = open(out_fname, 'w')
    # !! Labels and print_status data structures must be coherent !!
    num_space = 30 # Width printed numerical values
    indlab_space = 2 # Header index width
    lab_space = num_space-indlab_space-1 # Match width of printed number, including parenthesis
    header_labels = ['ly', 'e_pot', 'forces[0]', 'forces[1]', 'torque']
    # Gnuplot-compatible (leading #) fix-width output file
    first = '#{i:0{ni}d}){s: <{n}}'.format(i=0, s='lx', ni=indlab_space, n=lab_space-1,c=' ')
    print(first+"".join(['{i:0{ni}d}){s: <{n}}'.format(i=il+1, s=lab, ni=indlab_space, n=lab_space,c=' ')
                        for il, lab in zip(range(len(header_labels)), header_labels)]), file=outstream)

    # Inner-scope shortcut for printing
    def print_status():
        data = [lx, ly, e_pot, forces[0], forces[1], torque]
        print("".join(['{n:<{nn}.16g}'.format(n=val, nn=num_space)
                       for val in data]), file=outstream)

    #-------- STRING MINIMISATION ----------------
    t0 = time() # Start clock
    c_log.info("computing map. Turn off logger propagate, see specific console given")
    c_log.propagate = False

    infoskip = 20
    for i in range(Nsteps):
        if i % int(Nsteps/infoskip) == 0: c_log.info("On it %i (%.2f%%)" % (i, i/Nsteps*100))
        L.eulerArc(V, dt=1e-6)
        V.update(L)

    i = 0
    for lx, ly in zip(L.x, L.y):
        e_pot, forces, torque = calc_en_f(pos+[lx, ly], [lx, ly], *en_params)
        print_status()
    outstream.close()

    c_log.propagate = True
    t_exec = time() - t0
    c_log.info("Done in %is (%imin)" % (t_exec, t_exec/60))
    return L, V

if __name__ == "__main__":
    t0 = time()
    debug = False

    #-------- SET UP LOGGER -------------
    c_log = logging.getLogger('driver') # Set name identifying the logger.
    # Adopted format: level - current function name - message. Width is fixed as visual aid.
    logging.basicConfig(format='[%(levelname)5s - %(name)15s] %(message)s')
    c_log.setLevel(logging.INFO)
    if debug: c_log.setLevel(logging.DEBUG)

    with open(sys.argv[1]) as inj:
        inputs = json.load(inj)

    N0, N1 = int(sys.argv[2]), int(sys.argv[3])
    dN = 1
    try: dN = int(sys.argv[4])
    except: pass
    c_log.info("From Nl %i to %i steps %i" % (N0, N1, dN))

    # Substrate
    Rcl = 4.45 # Lattice spacing of cluster. Fixed by experiments.
    symmetry = inputs['sub_symm']
    well_shape = inputs['well_shape']
    R = inputs['R'] # Well lattice spacing [micron]
    epsilon =  inputs['epsilon'] # Well depth [zJ]
    # define substrate metric
    if symmetry == 'square':
        calc_matrices = calc_matrices_square
    elif symmetry == 'triangle':
        calc_matrices = calc_matrices_triangle
    else:
        raise ValueError("Symmetry %s not implemented" % symmetry)
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
        #a = R/2*inputs['at'] # Tempered tail as fraction of R
        #b = R/2*inputs['bt'] # Flat end as fraction of R#
        a = inputs['a'] # Well end radius [micron]
        b = inputs['b'] # Well slope radius [micron]
        sigma = inputs['sigma'] # Width of Gaussian
        en_params = [a, b, sigma, epsilon, u, u_inv]
        calc_en_f = calc_en_gaussian
    else:
        raise ValueError("Form %s not implemented" % well_shape)

    c_log.info("%s substrate: R=%.6g depth eps=%.4g. " % (symmetry, R, epsilon))
    c_log.info("%s sub parms: " % well_shape + " ".join(["%s" % str(i) for i in en_params[:-2]]))

    # Path
    llp = np.linspace(-0.8,0.8)*R
    phi = 0
    try: phi = float(sys.argv[5])
    except: pass
    llx, lly = llp*np.cos(phi*np.pi/180), llp*np.sin(phi*np.pi/180)
    p0 = [llx[0], lly[0]]
    p1 = [llx[-1], lly[-1]]
    inputs['p0'], inputs['p1'] = p0, p1
    c_log.info("String from %.2f %.2f to %.2f %.2f (phi=%.2f)" % (*p0, *p1, phi))

    # Set up system for multiprocess
    ncpu = os.cpu_count()
    nworkers = 8
    try: nworkers = int(sys.argv[6])
    except: pass
    Nl_range = range(N0, N1, dN) # Sizes to explore, in parallels
    c_log.info("Running %i elements on %i processes (%i cores machine)" % (len(Nl_range), nworkers, ncpu))

    # Fix the all arguments a part from Nl, so that we can use pool.map
    partial_traslmap_Nl = partial(static_barrier_Nl, inputs=inputs, calc_en_f=calc_en_f, debug=debug) # defaut should be fine for most

    # Launch all simulations with Pool of workers
    c_log.debug("Starting pool")
    pool = multiprocessing.Pool(processes=nworkers)
    c_log.debug("Mapping processes")
    results = pool.map(partial_traslmap_Nl, Nl_range)

    c_log.debug("Close pool and wait for joining")
    pool.close() # Pool doesn't accept new jobs
    pool.join() # Wait for all processes to finish
    c_log.debug("Results: %s" % str(results))

    # Print execution time
    t_exec = time()-t0
    c_log.info("Done in %is (%.2fmin or %.2fh)" % (t_exec, t_exec/60, t_exec/3600))
