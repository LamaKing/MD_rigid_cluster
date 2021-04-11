#!/usr/bin/env python3

# System stuff
import json, sys, logging, os
from time import time
# Science stuff
import numpy as np
from numpy import array as npa
from numpy.linalg import norm as npnorm
import pandas as pd

if __name__ == '__main__':

    t0 = time()
    debug = False

    #-------- SET UP LOGGER -------------
    c_log = logging.getLogger('driver') # Set name identifying the logger.
    # Adopted format: level - current function name - message. Width is fixed as visual aid.
    logging.basicConfig(format='[%(levelname)5s - %(name)15s] %(message)s')
    c_log.setLevel(logging.INFO)

    N = int(sys.argv[1])
    with open('info-driver-N_%i.json' % N, 'r') as inj:
        inputs = json.load(inj)

    rmob0, tmob0 = inputs['rmob0'], inputs['tmob0']
    # Fractions to consider cluster roto/trasl depinned
    rmob_frac, tmob_frac = inputs['rmob_frac'], inputs['tmob_frac']
    Tau_range, F_range = npa(inputs['Tau_range']), npa(inputs['F_range'])

    nTau, nF = len(Tau_range), len(F_range)
    dTau, dF = Tau_range[1]-Tau_range[0], F_range[1]-F_range[0]

    # -1=not computed, 0=pinned, 1=roto, dep 2=trasl dep, 3=full dep
    TauF_grid = -1*np.ones((nTau, nF), dtype=int)

    results = []
    for i, exTau in enumerate(Tau_range):
        for j, exF in enumerate(F_range):
            if exF==0 and exTau==0: continue

            c_log.info('On Tau=%.4g F=%.4g' % (exTau, exF))
            # Output of simulations
            c_outf = "out-ramp_F-Tau_%.4g-F_%.4g.dat" % (exTau, exF)
            #print(c_outf)
            try:
                # Read data, output from above section
                data = pd.read_fwf(c_outf, infer_nrows=1e30) # Load output file
            except FileNotFoundError:
                c_log.info("Not computed")
                c_log.info("-"*30+'\n')
                continue
                pass
            transient = int(np.floor(len(data)/3)) # Discard first third of simulation

            omega, Tau, theta = [data[label].to_numpy()
                                 for label in ['07)omega', '10)torque', '06)angle']
            ]
            Vx, Vy, Fx, Fy, x, y = [data[label].to_numpy()
                                    for label in ['04)Vcm[0]', '05)Vcm[1]',
                                                  '08)forces[0]', '09)forces[1]',
                                                  '02)pos_cm[0]', '03)pos_cm[1]']
            ]
            V = npnorm([Vx, Vy], axis=0)
            F = npnorm([Fx, Fy], axis=0)
            R = npnorm([x, y], axis=0)
#            print(V.shape)
            rmobility = (omega/Tau)[transient:].mean()*bool(exTau)
            tmobility = (V/F)[transient:].mean()*bool(exF)
            rdep, tdep = rmobility>rmob0*rmob_frac, tmobility>tmob0*tmob_frac
            c_log.info("Rmob %.5g (thold %.5g), Tmob %.5g (thold %.5g)", rmobility, rmob0*rmob_frac, tmobility, tmob0*tmob_frac)
            if R[-1] > inputs['MD_inputs']['rcm_max']: tdep = 1
            if theta[-1] > inputs['MD_inputs']['theta_max']: rdep = 1
            TauF_grid[i,j] = rdep + 2*tdep # 0=pinned, 1=roto, dep 2=trasl dep, 3=full dep

            if TauF_grid[i,j] == 0: c_log.info('Pinned')
            elif TauF_grid[i,j] == 1: c_log.info('R dep')
            elif TauF_grid[i,j] == 2: c_log.info('T dep')
            elif TauF_grid[i,j] == 3: c_log.info('RT dep')
            else: c_log.info('this should NOT happen')
            c_log.info("-"*30+'\n')
    np.savetxt('TauF_grid-N_%i.npdat' % N, TauF_grid)
    exit()


    # Always unpinned
    if rmobility[0]/rmob0 > rmobility_frac:
        c_log.warning('First config is unpinned: rmobil (%.3g) > rmobil_free*frac (%.3g*%.3g)' % (rmobility[0], rmob0, rmobility_frac))
        Tauc = 0
        err_Tauc = Tau[0]
    # Alwasy pinned
    elif rmobility[-1]/rmob0 < rmobility_frac:
        c_log.warning('Last config is pinned: rmobil (%.3g) < rmobil_free*frac (%3.g*%.3g)' % (rmobility[-1], rmob0, rmobility_frac))
        Tauc = Tau[-1] * 2 # !!! ABSOLUTELY ARBITRARY !!!
        err_Tauc = Tau[-1]*1.1 # Lower bound must be higher than last force measured
    else:
        # Linear interpolatee between last-pinned and first-depinned
        c_log.debug("Interpolating")
        rotating_mask = rmobility > rmob0*rmobility_frac  # Divide spinning and non-spinning
        y2, y1 = rmobility[rotating_mask][0], rmobility[~rotating_mask][-1] # last sub-critical and first over-critical rmobility
        x2, x1 = Tau[rotating_mask][0], Tau[~rotating_mask][-1] # last sub-critical and first over-critical torquee
        m = (y2-y1)/(x2-x1) # Slope
        Tauc = x1 + (rmobility_frac*rmob0-y1)/m # Tau at critical rmobility
        err_Tauc = (x2-x1)/2 # Uncertanty as half the distance between pinned and unpinned.

    # Print size, bravais size, Tc, sigma_Tc, omega
    results.append([N, Nl, Tauc, err_Tauc, float(omega[-1]), float(theta[-1])])
    c_log.info(" ".join("%25.20g" % r for r in results[-1]))

    # Print ordered results
    print(" ".join(["%25s" % header for header in ['#N', 'Nl', 'Tauc', 'err_Tauc', 'omega_fin', 'theta_fin']]), file=sys.stdout)
    for l in results:
        print(" ".join(["%25.20g" % f for f in l]), file=sys.stdout)

    # Print execution time
    t_exec = time()-t0
    c_log.info("Done in %is (%.2fmin or %.2fh)" % (t_exec, t_exec/60, t_exec/3600))
