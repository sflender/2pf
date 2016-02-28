import math
import numpy as np
from numpy import log, exp, sqrt, sin, cos
from numpy import size
import camb
from camb import model, initialpower

def H(H0, Omega_M, z): # returns the Hubble rate at redshift z (in units km/s/Mpc)
    return H0*sqrt(Omega_M*(1+z)**3 + (1-Omega_M) )

def Hred(Omega_M, z): # returns the reduced Hubble rate at redshift z (in units km/s/(Mpc/h))
    return 100.0*sqrt(Omega_M*(1+z)**3 + (1-Omega_M) )

def Hmod(H0, Omega_M, Omega_L, w, z):
    return H0*sqrt(Omega_M*(1+z)**3 + Omega_L*(1+z)**(3*(1+w)) )

def a(z):
    return 1.0/(1.0+z)

def window(x): #top-hat window function
    return 3*(sin(x)-x*cos(x))/x**3

def Gwindow(x): #Gaussian window function
    return exp(-0.5*x**2)

def sinc(x):
    return sin(x)/x

def j1(x):
    return (sin(x)-x*cos(x))/x**2

def alpha(w,Omega_M):
    return(3.0/(5.0-w/(1.0-w)) + 3.0/125.0 * (1.0-w) * (1.0-3.0*w/2.0)/(1.0-6.0*w/5.0)**3 * (1-Omega_M) )

def growth_rate(Omega_M, z):
    gamma = 0.55 #GR prediction
    Omega_M_z = Omega_M*(1+z)**3 # Omega_M at redshift z (it was higher in the past)
    f = Omega_M_z**gamma
    return f

def rho_crit(H0, Omega_M, z): #returns the critical density at redshift z in proper units 
    m_sun = 1.98892e30;
    G = 6.67e-11/1.0e9; #in km^3 kg^-1 s^-2
    mpc = 3.0857e19; #in km
    gnewt = G*m_sun/mpc; #for cosm params (in km^2 Mpc msun^-1 s^-2)
    return H(H0, Omega_M, z)**2 *3.0/(8.0*math.pi*gnewt);

def rho_crit_red(Omega_M, z): #same as above in reduced units
    m_sun = 1.98892e30;
    G = 6.67e-11/1.0e9; #in km^3 kg^-1 s^-2
    mpc = 3.0857e19; #in km
    gnewt = G*m_sun/mpc; #for cosm params (in km^2 Mpc msun^-1 s^-2)
    return Hred(Omega_M, z)**2 *3.0/(8.0*math.pi*gnewt);

def calc_M200(Omega_M,R200,z):
    return (4.0/3.0) * R200**3 * 200.0 * rho_crit_red(Omega_M,z)

def calc_R200(Omega_M,M200,z):
    return (3.0*M200/(4.0*math.pi*rho_crit_red(Omega_M,z)))**(1.0/3.0)

def xi_dv(r, z, h, Omega_DM, ombh2, w, mnu, smoothing_scale):
    # comoving matter-velocity correlation function in km/s at comoving distance r (in units Mpc/h) and redshift z
    # the power spectrum is smoothed with a top-hat window function with the specified smoothing scale in Mpc/h.
    # w is the dark energy eq. of state
    # mnu is the sum of the neutrino masses in eV

    H0 = h*100
    omch2 = Omega_DM * h**2
    Omega_b = ombh2 * h**(-2)
    Omega_M = Omega_DM + Omega_b
    
    pars = camb.CAMBparams() #Set up a new set of parameters for CAMB
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=0, tau=0.06)
    pars.InitPower.set_params(ns=0.965, r=0)
    pars.set_dark_energy(w)
    pars.set_for_lmax(2500, lens_potential_accuracy=0);
    results = camb.get_results(pars)
    pars.set_matter_power(redshifts=[z], kmax=2.0)
    pars.NonLinear = model.NonLinear_none #compute the linear power spectrum (i.e. w/o halofit)
    #pars.NonLinear = model.NonLinear_both #using halofit
    results = camb.get_results(pars) #compute the power spectra
    kh, dummy, pk = results.get_matter_power_spectrum(minkh=1e-3, maxkh=2, npoints = 10000)
    curlyP = kh**3 * pk / (2.0*math.pi**2)
    # pk in Mpc^3/h^3
    # kh in h/Mpc
    # curlyP is dimensionless
    
    dlnk = log(kh[1])-log(kh[0])
    temp = 0
    norm = -a(z)*Hred(Omega_M,z)*growth_rate(Omega_M,z)
    # units of norm: km/s/(Mpc/h)
    for i in range(0,size(kh)):
        temp = temp + dlnk * j1(kh[i]*r) * curlyP[0][i]/kh[i] * window(kh[i]*smoothing_scale)**2
    return norm*temp

def xi_dd(r,z,h,Omega_DM,ombh2,w,mnu,smoothing_scale):
    # returns the matter-matter correlation function (dimensionless)
    # at comoving separation r in Mpc/h, and using the given smoothing scale in Mpc/h

    H0 = h*100
    omch2 = Omega_DM * h**2
    Omega_b = ombh2 * h**(-2)
    Omega_M = Omega_DM + Omega_b
    
    pars = camb.CAMBparams() #Set up a new set of parameters for CAMB
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=0, tau=0.06)
    pars.InitPower.set_params(ns=0.965, r=0)
    pars.set_dark_energy(w)
    pars.set_for_lmax(2500, lens_potential_accuracy=0);
    results = camb.get_results(pars)
    pars.set_matter_power(redshifts=[z], kmax=2.0)
    pars.NonLinear = model.NonLinear_none #compute the linear power spectrum (i.e. w/o halofit)
    #pars.NonLinear = model.NonLinear_both #using halofit
    results = camb.get_results(pars) #computes the power spectra
    kh, dummy, pk = results.get_matter_power_spectrum(minkh=1e-3, maxkh=2, npoints = 10000)
    curlyP = kh**3 * pk / (2.0*math.pi**2) 
    
    dlnk = log(kh[1])-log(kh[0])

    temp=0
    for i in range(0,size(kh)):
        temp = temp + dlnk * sinc(kh[i]*r) * curlyP[0][i] * window(kh[i]*smoothing_scale)**2
    return temp


def v12(r, z, b, h,Omega_DM,ombh2,w,mnu,smoothing_scale):
    # returns the comoving pairwise velocity using the specified mass-averaged bias b
    answer = 2*b*xi_dv(r,z,h,Omega_DM,ombh2,w,mnu,smoothing_scale)/(1+b**2 * xi_dd(r,z,h,Omega_DM,ombh2,w,mnu,smoothing_scale)) #see Soergel et al
    return answer

def v12_lin(r, z, b, h,Omega_DM,ombh2,w,mnu,smoothing_scale):
    # returns the comoving linear pairwise velocity using the specified mass-averaged bias b
    answer = 2*b*xi_dv(r,z,h,Omega_DM,ombh2,w,mnu,smoothing_scale) #see e.g. Soergel et al
    return answer
