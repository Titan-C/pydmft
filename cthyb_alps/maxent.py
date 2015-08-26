import argparse
import pyalps.mpi as mpi          # MPI library
from pyalps.hdf5 import archive       # hdf5 interface
import pyalps.maxent as maxent    # the maxent module
import matplotlib.pyplot as plt   # for plotting results
from numpy import sin,cos,exp,log,sqrt,pi,ndarray,array,zeros # some math
from scipy.special import lpn as legendre
from scipy.special import sph_jn,jn as besselJ
from scipy.optimize import curve_fit


parser = argparse.ArgumentParser()
parser.add_argument('--PH',action='store_true')
parser.add_argument('--TEXT_OUTPUT',action='store_true')
parser.add_argument('--VERBOSE',action='store_true')
parser.add_argument('--DATA_SPACE',default="time")
parser.add_argument('--FREQUENCY_GRID',default="log")
parser.add_argument('--KERNEL',default="fermionic")
parser.add_argument('--COVARIANCEMATRIX',action='store_true')
parser.add_argument('--NFREQ',type=int,default=500)
parser.add_argument('--NALPHA',type=int,default=300)
parser.add_argument('--NORM',type=float,default=1.0)
parser.add_argument('--OMEGA_MAX',type=float,default=5.0)
parser.add_argument('basename')
parser.add_argument('orb', metavar='orb',type=int)
parser.add_argument('alpha',nargs=2)
default_model = parser.add_subparsers(dest='default_model_name')
flat = default_model.add_parser('flat')
gaussian = default_model.add_parser('gaussian')
gaussian.add_argument('--SIGMA',type=float,default=1.0)
double_gaussian = default_model.add_parser('twogaussians')
double_gaussian.add_argument('SHIFT1',type=float)
double_gaussian.add_argument('SIGMA1',type=float)
#double_gaussian.add_argument('NORM1',type=float)
double_gaussian.add_argument('SHIFT2',type=float)
double_gaussian.add_argument('SIGMA2',type=float)
file = default_model.add_parser('file')
file.add_argument('filename')
file.add_argument('HDF5',action='store_true')
args = parser.parse_args()

ar=archive(args.basename+".out.h5",'r')

B=ar['parameters/BETA']
U=ar['parameters/U']
N_TAU=ar['parameters/N_TAU']


orb=args.orb


parms = {
    # solver parameters
    'N_ALPHA' : args.NALPHA,
    'ALPHA_MIN' : args.alpha[0],
    'ALPHA_MAX' : args.alpha[1],
    'NORM' : args.NORM,
    'OMEGA_MAX' : args.OMEGA_MAX,
    'BASENAME' : args.basename+"_%i_maxent"%orb,
    'KERNEL' : args.KERNEL,
    'TEXT_OUTPUT' : args.TEXT_OUTPUT,
    'VERBOSE' : args.VERBOSE,
    # file names
    'DATA' : "DATA.h5",
    'DATA_IN_HDF5' : 1,
    # physical parameters
    'BETA' : B,
    'NFREQ' : args.NFREQ,
    'NDAT' : N_TAU+1,
    'MAX_IT' : 2000,
    # measurements
    'DATA_SPACE' : args.DATA_SPACE,
    'FREQUENCY_GRID' : args.FREQUENCY_GRID,
    'PARTICLE_HOLE_SYMMETRY' : args.PH
}

if args.default_model_name != "file":
  parms['DEFAULT_MODEL'] = args.default_model_name
  if args.default_model_name == "gaussian":
    parms["SIGMA"] = args.SIGMA
  if args.default_model_name == "twogaussians":
    parms["SIGMA1"] = args.SIGMA1
    parms["SIGMA2"] = args.SIGMA2
    parms["SHIFT1"] = args.SHIFT1
    parms["SHIFT2"] = args.SHIFT2
    parms["NORM1"] = ar['simulation/results/density_%i/mean/value'%orb]
else:
  if not args.HDF5:
    parms['DEFAULT_MODEL'] = args.filename
  else:
    old_data = archive(args.filename,'r')
    om = array(old_data['spectrum/omega'])
    A = array(old_data['spectrum/average'])
    f = open("Default_Model.dat","w")
    parms['NFREQ'] = len(om)
    for i in range(len(om)):
      f.write("%e %e\n"%(om[i],A[i]))
    f.close()
    parms['DEFAULT_MODEL'] = "Default_Model.dat"




if args.COVARIANCEMATRIX:
    parms['COVARIANCE_MATRIX'] = 1

Gt = array(ar['G_tau/%i/mean/value'%orb])
dGt = array(ar['G_tau/%i/mean/error'%orb])
CovG = array(ar['G_tau/%i/mean/covariance'%orb])
if args.PH:
  Gt = (Gt[range(0,N_TAU+1,1)]+Gt[range(N_TAU,-1,-1)])/2.0
  dGt = sqrt(dGt[range(0,N_TAU+1,1)]**2+dGt[range(N_TAU,-1,-1)]**2)/2.0


ar=archive("DATA.h5",'w')
ar["/Data"] = Gt
ar["/Error"] = dGt
ar["/Covariance"] = CovG


del ar

maxent.AnalyticContinuation(parms)

ar=archive(args.basename+"_%i_maxent.out.h5"%orb,"r")

alpha = array(ar["/alpha/values"])
prob = array(ar["/alpha/probability"])

omega = array(ar["/spectrum/omega"])
av_spec = array(ar["/spectrum/average"])
max_spec = array(ar["/spectrum/maximum"])

del ar

plt.figure()
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$P(\alpha)$')
plt.plot(alpha,prob)
plt.xscale('log')
plt.show()

plt.figure()
plt.xlabel(r'$\omega$')
plt.ylabel(r'$A(\omega)$')
plt.plot(omega,av_spec,label="average")
plt.plot(omega,max_spec,label="maximum")
plt.legend()
plt.show()
