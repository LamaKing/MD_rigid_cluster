# Overdamped dynamics of CM of rigid cluster of colloids

Define the geometry of substrate and cluster, system properties and then solve the equation of motion for the Centre of Mass (CM) of the cluster.
```MD_angle.sh``` creates the cluster and calls the Langevin solver.

### Substrate
The lithographic patterns are approximated by Tanh function in X. Cao, E. Panizon, A. Vanossi, N. Manini, E. Tosatti, and C. Bechinger, Phys. Rev. E 103, 1 (2021).
The substrate is defined by:
  - the spacing ```R``` between the wells
  - the well depththe ```epsilon```
  - the well width ```a``` 
  - the transition region width ```b```
  - The ideal well is correctd by a shape factor ```wd``` (0.29 of R=5micron is a good bet).

The symmetry implemented, as of now, are ```triangular``` and ```square```.
There should be a Jupyter NB showing a bit how this substrate looks like and how the forces scale with the 

### Cluster
Hexagonal clusters are created by giving to ```tool_create_hexagons.py``` the lattice vectors a and b and the repetitions along each of them. 
See the script description.

## Units
Base units of are:
  - E in zj
  - L in micron
  - mass in fKg

It follows:
  - F in fN
  - torque in fN*ms
  - t in ms
  - gamma in fKg/ms

## Equations of motion
The ```script MD_rigid_rototrasl.py``` solves first order equations, i.e. fully overdamped dynamics with no interial term:
  
```dr/dt = (F_ext - grad U)/gamma_trasl```

```dtheta/dt = (T_ext - dU/dtheta)/gamma_rot```
  
In this picture energy is not conserved (fully dissipated in the Langevin bath between timesteps) and the value of the dissipation constants, gamma_trasl and gamma_rot, effectively sets how quickly the time flows. Thus ms is the formal unit of time, but by lowering gamma we should be able to explore timescales similar to experiments.

### The problem of gamma scaling
How do we scale from the dissipation gamma of a single colloid to the one of the CM?
For translations is easy: gamma_trasl = N gamma, where N is the number of colloids in the cluster.
For the rotation is a bit more complicated. And we lack a clear understanding of how the rotation actuated by the magnetic field in the experiments (or at least I do). So we should just reproduce the experimental scaling of omega = dtheta/dt for increasing torque T_ext over a flat substrate, dU/dtheta=0.

It looked like it scaled as N^-3/2, but it's now closer N^-2 (or N^-1/2 - N^-1 for T_ext/N). There are some LaTex notes somewhere deriving how you could get the different scaling. The N^-2 is the most "physically sound" one: integrate of the torque generated by the dissipation of each colloid ```sum_i r_i*(-gamma rdot_i)```.


# TODO
  - Redo testing with new scaling factor:
    * Translation only: F1s
    * Flat substrate omega scaling
    * Critical torque scaling with N
    * PRE directional locking: thetao=-3.4 thetad=26.6
  - Add testing folder. Possibly only inputs and analysis if outputs are too big.
  - Roto-trasl coupling: take PRE dir locking and drive the sys in perpendicular direction. Measure Fs. Fix F<Fs and add torque. How does the response change?
  - Critical torque behaviour for different lattice symmetry and mismatch.
