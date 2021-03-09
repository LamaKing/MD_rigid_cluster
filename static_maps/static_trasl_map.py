#!/usr/bin/env python

import sys, os, json, logging, multiprocessing
import numpy as np
from time import time
from functools import partial
from tempfile import NamedTemporaryFile
from tool_create_circles import create_cluster_circle
from tool_create_hexagons import create_cluster_hex

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

def calc_en_gaussian(pos, sigma, epsilon, u, u_inv):
    Xp = pos[:,0]*u[0,0] + pos[:,1]*u[0,1]
    Yp = pos[:,0]*u[1,0] + pos[:,1]*u[1,1]
    Xp -= np.floor(Xp + 0.5)
    Yp -= np.floor(Yp + 0.5)
    X = Xp*u_inv[0, 0] + Yp*u_inv[0, 1]
    Y = Xp*u_inv[1, 0] + Yp*u_inv[1, 1]
    R  = np.sqrt(X*X+Y*Y)
    off = R != 0
    en = -epsilon*np.sum(gaussian(R[off], 0, sigma))
    FX = epsilon*gaussian(R[off],0,sigma) * (R[off] / np.power(sigma, 2.)) * X[off] / R[off]
    FY = epsilon*gaussian(R[off],0,sigma) * (R[off] / np.power(sigma, 2.)) * Y[off] / R[off]
    return en, [np.sum(FX), np.sum(FY)]

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))) / (2. * np.pi * np.power(sig, 2.))

def cluster_inhex_Nl(N1, N2,  a1 = np.array([4.45, 0]), a2 = np.array([-4.45/2, 4.45*np.sqrt(3)/2]),
                     clgeom_fname = "input_pos.hex", cluster_f = create_cluster_circle):
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

# rotates pos vector (the first two rows are X,Y) by an angle in degrees
def rotate(pos, angle):
    for i in range(pos.shape[0]):
        newx = pos[i,0] * np.cos(angle/180*np.pi) - pos[i,1] * np.sin(angle/180*np.pi)
        newy = pos[i,0] * np.sin(angle/180*np.pi) + pos[i,1] * np.cos(angle/180*np.pi)
        pos[i,0] = newx
        pos[i,1] = newy
    return pos

def static_traslmap_Nl(Nl, inputs, calc_en_f, name=None, out_fname=None, info_fname=None, debug=False):

    if name == None:
        name = 'traslmap_Nl_%i' % Nl
    if out_fname == None: out_fname = 'out-%s.dat' % name
    if info_fname == None: info_fname = 'info-%s.txt' % name

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
    forces = np.zeros(2)
    torque = 0.
    da1 = np.zeros(2)
    da2 = np.zeros(2)

    # Map params
    nbin = inputs['nbin'] # Reduce coordinate fraction
    Nsteps = nbin**2
    dtheta = inputs['dtheta'] # Reduce coordinate fraction

    #-------- INFO FILE ---------------
    with open(info_fname, 'w') as infof:
        infod = {'N': N,'Nsteps': Nsteps, 'pos_cm': pos_cm.tolist()}
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
    header_labels = ['pos_cm[1]', 'e_pot', 'forces[0]', 'forces[1]', 'torque']
    # Gnuplot-compatible (leading #) fix-width output file
    first = '#{i:0{ni}d}){s: <{n}}'.format(i=0, s='pos_cm[0]', ni=indlab_space, n=lab_space-1,c=' ')
    print(first+"".join(['{i:0{ni}d}){s: <{n}}'.format(i=il+1, s=lab, ni=indlab_space, n=lab_space,c=' ')
                        for il, lab in zip(range(len(header_labels)), header_labels)]), file=outstream)

    # Inner-scope shortcut for printing
    def print_status():
        data = [pos_cm[0], pos_cm[1], e_pot, forces[0], forces[1], torque]
        print("".join(['{n:<{nn}.16g}'.format(n=val, nn=num_space)
                       for val in data]), file=outstream)

    #-------- ENERGY TRASL MAP ----------------
    t0 = time() # Start clock
    c_log.info("computing map. Turn off logger propagate, see specific console given")
    c_log.propagate = False
    it = 0
    for dda1 in np.linspace(0, 1, nbin, endpoint=True):
        for dda2 in np.linspace(0, 1, nbin, endpoint=True):
            # This is so damn inefficient
            delta = np.array([dda1, dda2])
            r =  np.dot(delta, u_inv)
            pos_cm[:] = r
            #pos_cm[:] =  np.array([dda1, dda2])
            # energy is -epsilon inside well, 0 outside well. Continuous function in between.
            e_pot, forces = calc_en_f(pos + pos_cm, *en_params)
            # torque is T = - dE / dTheta = - (E(theta+) - E(theta-)) / 2dTheta
            en_plus, _ = calc_en_f(rotate(pos, dtheta) + pos_cm, *en_params)
            en_minus,_ = calc_en_f(rotate(pos,-dtheta) + pos_cm, *en_params)
            torque = -(en_plus - en_minus)/dtheta/2

            # Print progress
            if it % printerr_skip == 0:
                c_log.info("it=%10.4g of %.4g (%2i%%) da1=%10.3g da2=%10.3g en=%10.6g x=%9.3g y=%9.3g" % (it, Nsteps, 100.*it/Nsteps, delta[0], delta[1], e_pot, pos_cm[0], pos_cm[1]))

            # Print step results
            if it % print_skip == 0: print_status()

            it += 1
    #-----------------------------
    outstream.close()

    c_log.propagate = True
    c_log.info("map done, turn on logging propagate.")
    t_exec = time() - t0 # Stop clock
    c_log.info("Done in %is (%.2fmin or %.2fh)" % (t_exec, t_exec/60, t_exec/3600))
    c_log.info("Speed %5.3f s/step" % (t_exec/Nsteps))

    return 0

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

    # Substrate
    Rcl = 4.45 # Lattice spacing of cluster. Fixed by experiments.
    symmetry = inputs['sub_symm']
    en_form = inputs['en_form']
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

    if en_form == 'tanh':
        # Realistic well energy landscape
        calc_en_f = calc_en_tan
        a = inputs['a'] # Well end radius [micron]
        b = inputs['b'] # Well slope radius [micron]
        wd = inputs['wd'] # Well asymmetry. 0.29 is a good value
        en_params = [a, b, wd, epsilon, u, u_inv]
    elif en_form == 'gaussian':
        # Gaussian energy landscape
        sigma = inputs['sigma'] # Width of Gaussian
        en_params = [sigma, epsilon, u, u_inv]
        calc_en_f = calc_en_gaussian
    else:
        raise ValueError("Form %s not implemented" % en_form)

    c_log.info("%s substrate: R=%.6g depth eps=%.4g. " % (symmetry, R, epsilon))
    c_log.info("%s sub parms: " % en_form + " ".join(["%s" % str(i) for i in en_params[:-2]]))

    # Set up system for multiprocess
    ncpu = os.cpu_count()
    nworkers = 4
    Nl_range = [3, 5, 10, 30] # Sizes to explore, in parallels
    c_log.info("Running %i elements on %i processes on %i cores" % (len(Nl_range), nworkers, ncpu))

    # Fix the all arguments a part from Nl, so that we can use pool.map
    partial_traslmap_Nl = partial(static_traslmap_Nl, inputs=inputs, calc_en_f=calc_en_f, debug=debug) # defaut should be fine for most

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
