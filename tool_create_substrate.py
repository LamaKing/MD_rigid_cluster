import sys, os, json, logging
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

def calc_en_tan(pos, pos_cm, a, b, ww, epsilon, u, u_inv):
    """Calculate energy and forces. Well is approximated with tanh function."""
    from numpy.linalg import norm as npnorm
    en = 0
    F = np.zeros(shape=(pos.shape[0],2))
    # map to substrate cell
    posp = np.dot(u, pos.T).T # Fast numpy dot with different convention on row/cols
    posp -= np.floor(posp + 0.5)
    # back to real space
    pospp = np.dot(u_inv, posp.T).T
    posR = npnorm(pospp, axis=1)
    # colloids inside the curve region
    # mask and relative R
    inside = np.logical_and(posR<b, posR>a)
    Rin = posR[inside]
    # colloids inside the curve region
    inside = np.logical_and(posR<b, posR>a) # numpy mask vector
    # energy inside flat bottom region
    en = -epsilon*np.sum(posR <= a)
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
    F[inside,0] = vecF*pospp[inside,0]/Rin
    F[inside,1] = vecF*pospp[inside,1]/Rin
    # Torque = r vec F. Applied to CM pos
    tau = np.cross(pos-pos_cm, F)
    # Return energy, F and torque on CM
    return en, np.sum(F, axis=0), np.sum(tau)

def calc_en_gaussian(pos, pos_cm, sigma, epsilon, u, u_inv):
    from numpy.linalg import norm as npnorm
    F = np.zeros(shape=(pos.shape[0],2))
    # Unit cell mapping
    posp = np.dot(u, pos.T).T
    posp -= np.floor(posp + 0.5)
    pospp = np.dot(u_inv, posp.T).T
    posR = npnorm(pospp, axis=1)
    off = posR != 0 # Forces at min are 0, but non analytic projecton.
    # Energy, forces and torque on CM
    en = -epsilon*np.sum(gaussian(posR, 0, sigma)) # Do include all for energy!
    F[off, 0] = -epsilon*gaussian(posR[off],0,sigma) * (posR[off] / np.square(sigma)) * pospp[off,0] / posR[off]
    F[off, 1] = -epsilon*gaussian(posR[off],0,sigma) * (posR[off] / np.square(sigma)) * pospp[off,1] / posR[off]
    tau = np.cross(pos-pos_cm, F)
    return en, np.sum(F, axis=0), np.sum(tau)

def gaussian(x, mu, sig):
    return np.exp(-np.square(x - mu) / (2 * np.square(sig)))
