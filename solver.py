import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
from tqdm import tqdm

from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from scipy.interpolate import interp1d
from scipy.ndimage import map_coordinates

from joblib import Parallel, delayed

import contextlib
import joblib
from tqdm import tqdm

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

class PadeApproximation():

    """
       Class to solve stationary advection-diffusion equation
       in polar coordinates.
       With the acquired solution 
       one can find the aggregation coefficients
       by integrating the flux over the center particle.
       Finally one can plot aggregation coefs approximation graphs
       and stationary advection-diffusion equation solution. 
    """

    def __init__(self):

        self.N = 1000
        self.T = 50
        self.r = np.linspace(1.0, 10.5, self.N)
        self.dr = self.r[1]-self.r[0]

        self.dt = math.pi/self.T
        self.theta = np.linspace(self.dt/2, math.pi-self.dt/2, self.T, dtype=np.float32)
        self.theta_half = np.linspace(0, math.pi, self.T+1, dtype=np.float32)
        self.last = len(self.theta)-1

        #self.muR = muR
        self.sections = np.array([0, 0.86, 1.47])

        #self.small_ap = np.zeros(muR.shape[0], dtype=np.float32)
        #self.big_ap = np.zeros(muR.shape[0], dtype=np.float32)
        #self.numerical = np.zeros(muR.shape[0], dtype=np.float32)
        #self.pade_ap = np.zeros(muR.shape[0], dtype=np.float32)

        self.x_ = np.linspace(0., 5., 200)
        self.y_ = np.linspace(-2., 8., 400)
        self.t = np.linspace(0, math.pi, self.T)
        #self.data = []

        return

    @staticmethod
    def polar2cartesian(r, t, grid, x, y, order=3):
        X, Y = np.meshgrid(x, y)
        new_r = np.sqrt(X*X+Y*Y)
        new_t = np.arctan2(X, Y)

        ir = interp1d(r, np.arange(len(r)), bounds_error=False)
        it = interp1d(t, np.arange(len(t)))

        new_ir = ir(new_r.ravel())
        new_it = it(new_t.ravel())

        new_ir[new_r.ravel() > r.max()] = len(r)-1
        new_ir[new_r.ravel() < r.min()] = 0

        return map_coordinates(grid, np.array([new_ir, new_it]),
                               order=order).reshape(new_r.shape)

    def solve_equation(self, muR: np.ndarray):
        """
            muR - array of analyzed Peclet numbers.
        """
        N, theta, r, t, dr, dt = self.N, self.theta, self.r, self.t, self.dr, self.dt
        theta_half, last, sections = self.theta_half, self.last, self.sections
        x_, y_ = self.x_, self.y_
        data = []
        
        matrix = np.zeros((N*len(theta), N*len(theta)), dtype = np.float32)
        for i in range(1, len(r)-1):
            for j in range(1, len(theta)-1):
                matrix[j+i*len(theta)][j+i*len(theta)] = \
                    -r[i+1]*np.exp(-muR*2*np.cos(theta[j])*dr/2) \
                     /(r[i]*dr*dr) \
                    -r[i-1]*np.exp(muR*2*np.cos(theta[j])*dr/2) \
                     /(r[i]*dr*dr) \
                    -(np.sin(theta_half[j+1]) \
                      *np.exp(-muR*2*(np.cos(theta[j+1])-np.cos(theta_half[j+1]) )*r[i])) \
                     /(np.sin(theta[j])*dt*dt*r[i]*r[i]) \
                    -(np.sin(theta_half[j]) \
                      *np.exp(-muR*2*(np.cos(theta[j])-np.cos(theta_half[j+1]) )*r[i])) \
                     /(np.sin(theta[j])*dt*dt*r[i]*r[i])
                matrix[j+i*len(theta)][(j+1)+i*len(theta)] = \
                     (np.sin(theta_half[j+1]) \
                      *np.exp(-muR*2*(np.cos(theta[j+1])-np.cos(theta_half[j+1]) )*r[i])) \
                     /(np.sin(theta[j])*dt*dt*r[i]*r[i])
                matrix[j+i*len(theta)][(j-1)+i*len(theta)] = \
                     (np.sin(theta_half[j]) \
                      *np.exp(-muR*2*(np.cos(theta[j])-np.cos(theta_half[j+1]) )*r[i])) \
                     /(np.sin(theta[j])*dt*dt*r[i]*r[i])
                matrix[j+i*len(theta)][j+(i+1)*len(theta)] = \
                     r[i+1]*np.exp(-muR*2*np.cos(theta[j])*dr/2)/(r[i]*dr*dr)
                matrix[j+i*len(theta)][j+(i-1)*len(theta)] = \
                     r[i-1]*np.exp(muR*2*np.cos(theta[j])*dr/2)/(r[i]*dr*dr)
        for i in range(1, len(r)-1):
            matrix[0+i*len(theta)][0+i*len(theta)] = \
                     -r[i+1]*np.exp(-muR*2*np.cos(theta[0])*dr/2)/(r[i]*dr*dr) \
                     -r[i-1]*np.exp(muR*2*np.cos(theta[0])*dr/2)/(r[i]*dr*dr) \
                     -(np.sin(theta_half[0+1]) \
                       *np.exp(-muR*2*(np.cos(theta[0+1])-np.cos(theta_half[0+1]) )*r[i])) \
                      /(np.sin(theta[0])*dt*dt*r[i]*r[i]) \
                     -(np.sin(theta_half[0]) \
                       *np.exp(-muR*2*(np.cos(theta[0])-np.cos(theta_half[0+1]) )*r[i])) \
                      /(np.sin(theta[0])*dt*dt*r[i]*r[i])
            matrix[0+i*len(theta)][1+i*len(theta)] = \
                      (np.sin(theta_half[0+1]) \
                       *np.exp(-muR*2*(np.cos(theta[1])-np.cos(theta_half[1]) )*r[i])) \
                      /(np.sin(theta[0])*dt*dt*r[i]*r[i])
            matrix[0+i*len(theta)][0+(i+1)*len(theta)] = \
                      r[i+1]*np.exp(-muR*2*np.cos(theta[0])*dr/2)/(r[i]*dr*dr)
            matrix[0+i*len(theta)][0+(i-1)*len(theta)] = \
                      r[i-1]*np.exp(muR*2*np.cos(theta[0])*dr/2)/(r[i]*dr*dr)

            matrix[last+i*len(theta)][last+i*len(theta)] = \
                      -r[i+1]*np.exp(-muR*2*np.cos(theta[last])*dr/2)/(r[i]*dr*dr) \
                      -r[i-1]*np.exp(muR*2*np.cos(theta[last])*dr/2)/(r[i]*dr*dr) \
                      -(np.sin(theta_half[last+1]) \
                        *np.exp(-muR*2*(np.cos(theta[last]+dt)-np.cos(theta_half[last+1]))*r[i])) \
                       /(np.sin(theta[last])*dt*dt*r[i]*r[i]) \
                      -(np.sin(theta_half[last]) \
                        *np.exp(-muR*2*(np.cos(theta[last])-np.cos(theta_half[last+1]))*r[i])) \
                       /(np.sin(theta[last])*dt*dt*r[i]*r[i])
            matrix[last+i*len(theta)][last-1+i*len(theta)] = \
                       (np.sin(theta_half[last]) \
                        *np.exp(-muR*2*(np.cos(theta[last])-np.cos(theta_half[last+1]) )*r[i])) \
                       /(np.sin(theta[last])*dt*dt*r[i]*r[i])
            matrix[last+i*len(theta)][last+(i+1)*len(theta)] = \
                       r[i+1]*np.exp(-muR*2*np.cos(theta[last])*dr/2)/(r[i]*dr*dr)
            matrix[last+i*len(theta)][last+(i-1)*len(theta)] = \
                       r[i-1]*np.exp(muR*2*np.cos(theta[last])*dr/2)/(r[i]*dr*dr)

        for j in range(0, len(theta)):
            matrix[j+0*len(theta)][j+0*len(theta)] = 1.0
            matrix[j+(len(r)-1)*len(theta)][j+(len(r)-1)*len(theta)] = 1.0

        b = np.zeros((len(r)*len(theta),1))

        for j in range(len(theta)):
            b[j+(len(r)-1)*len(theta)] = 1.
        A = csc_matrix(matrix)
        B = csc_matrix(b)
        solution = spsolve(A,B)
        solution = solution.reshape((len(r), len(theta)))

        cartezian_solution = self.polar2cartesian(r, t, solution, x_, y_, order=3)

        if any(abs(np.log10(muR) - sections)<0.05):
            data = cartezian_solution.copy()

        formula_der = (solution[1] - solution[0])/dr    
        formula_num = np.zeros(len(theta))
        formula_flux_adv = np.zeros(len(theta))
        numerical = 0
        for j in range(formula_flux_adv.shape[0]):
            formula_flux_adv[j] = -(- np.cos(theta[j]))
            formula_num[j] = -formula_der[j]
            if formula_num[j]<=0.0:
                numerical -= (formula_num[j])*np.sin(theta[j])*dt
        pade_ap = (2 + 10/3*muR*r[0] + 2/3*(muR*r[0])**2)/(1 + 2/3*muR*r[0])
        big_ap = muR*r[0]
        small_ap = ( 2.+ 2*muR*r[0] - 2/3*(muR*r[0])**2)
        return [data, small_ap, big_ap, pade_ap, numerical]

    def plot_solution(self, data):
        x_, y_ = self.x_, self.y_
        f, axarr = plt.subplots(1,3, figsize=(10, 8))
        axarr[0].imshow(data[0], extent = [x_[0], x_[-1], y_[-1], y_[0]])
        axarr[0].set_title("$\\mu = 1$", fontsize=20)
        axarr[1].imshow(data[1], extent = [x_[0], x_[-1], y_[-1], y_[0]])
        axarr[1].set_title("$\\mu = 7$", fontsize=20)
        im = axarr[2].imshow(data[2], extent = [x_[0], x_[-1], y_[-1], y_[0]])
        axarr[2].set_title("$\\mu = 30$", fontsize=20)
        axarr[0].set_ylabel("z", fontsize=20)
        axarr[0].tick_params(labelsize=15)
        axarr[1].set_yticks([])
        axarr[1].tick_params(labelsize=15)
        axarr[2].set_yticks([])
        axarr[2].tick_params(labelsize=15)
        axarr[0].set_xlabel("y", fontsize=20) 
        axarr[1].set_xlabel("y", fontsize=20) 
        axarr[2].set_xlabel("y", fontsize=20) 
        cbar_ax = f.add_axes([0.11, 0.1, 0.8, 0.03])
        cbar = f.colorbar(im, cax=cbar_ax, orientation="horizontal") 	
        cbar.ax.tick_params(labelsize=15)
        cbar.set_label("n", fontsize=20)
        f.savefig("cartezian.pdf")
        plt.close()

    def plot_pade_results(self, muR, small_ap, big_ap, pade_ap, numerical):

        plt.figure(figsize=(18,14))
        plt.plot(muR, small_ap/2., label = "asymptotics $\\mu \\rightarrow 0$", lw=4)
        plt.plot(muR, big_ap/2., label = "asymptotics $\\mu \\rightarrow \\infty$", lw=4)
        plt.plot(muR, numerical/2., label = "numerical solution", lw=4)
        plt.plot(muR, pade_ap/2., "o", label = r'Pad$\acute{e}$ approximation', markersize=8, markevery=5)
        plt.xlabel("$\\mu$", fontsize=55)
        plt.ylabel("$\\frac{K(\\mu)}{4 \\pi RD} $", fontsize=55)
        plt.xticks(fontsize=50)
        plt.yticks(fontsize=50)
        plt.xscale("log")
        plt.legend(fontsize = 40, loc='best')
        plt.ylim(bottom = -1, top = 55)
        plt.savefig("final.pdf")
        plt.close()

if __name__ == "__main__":
    muR = np.logspace(-2, 2, 100, base=10)
    solver = PadeApproximation()
    with tqdm_joblib(tqdm(desc="progress", total=100)) as progress_bar:
        res = Parallel(n_jobs=4)(delayed(solver.solve_equation)(mu) for mu in muR)
    res = np.array(res, dtype=object)
    data = np.array([x for x in res[:, 0] if len(x) > 0])
    solver.plot_solution(data)
    solver.plot_pade_results(muR, res[:,1], res[:,2], res[:,3], res[:,4])




