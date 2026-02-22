import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import os
import cupyx.scipy.ndimage as ndimage

from propagation import Propagation
from shift import Shift
from utils import *

class Rec:
    def __init__(self, args):

        # copy args to elements of the class
        for key, value in vars(args).items():
            setattr(self, key, value)        

        # list of functionals, gradients, differentials, and second-order differentials
        self.F = [self.F0, self.F1, self.F2, self.F3]
        self.gF = [self.gF0, self.gF1, self.gF2, self.gF3]
        self.dF = [self.dF0, self.dF1, self.dF2, self.dF3]
        self.d2F = [self.d2F0, self.d2F1, self.d2F2, self.d2F3]
        self.d2F_dF = [self.d2F_dF0, self.d2F_dF1, self.d2F_dF2,self.d2F_dF3]

        # X-ray propagation and magnification parameters for classes
        # 1. sample - detector
        wavelength = 1.24e-09/self.energy  # [m] wave length
        z2 = args.focusToDetectorDistance-self.z1
        distances = (self.z1 * z2) / self.focusToDetectorDistance
        magnifications = self.focusToDetectorDistance / self.z1
        norm_magnifications = magnifications / magnifications[0]
        voxelsize = np.abs(self.detector_pixelsize / magnifications[0])
        # scaled propagation distances due to magnified probes
        distances = distances*norm_magnifications**2

        # 2. probe - sample 
        z1p = self.z1[0]  # positions of the probe for reconstruction
        z2p = self.z1-np.tile(z1p, len(self.z1))
        magnificationsp = (z1p+z2p)/z1p
        distances_prb = (z1p*z2p)/(z1p+z2p)
        norm_magnifications_prb = magnificationsp/(z1p/self.z1[0])  # normalized magnifications
        distances_prb = distances_prb*norm_magnifications_prb**2
        distances_prb = distances_prb*(z1p/self.z1)**2

        print(f'Adjusted distances sample-detector {distances}')
        print(f'Adjusted distances probe-sample {distances_prb}')
        print(f'Sums adjusted {distances+distances_prb}')
        print(f'Voxelsize for reconstruction {voxelsize}')

        
        # initialize processing classes per gpu
        self.cl_prop = Propagation(self.n, self.nz, self.ndist, wavelength, voxelsize, distances)
        self.cl_prop_prb = Propagation(self.n, self.nz, self.ndist, wavelength, voxelsize, distances_prb)
        self.cl_shift = Shift(self.n, self.nobj, self.nz, self.nzobj, 1.0 / norm_magnifications,self.obj_dtype)
        self.grads = {'proj':cp.empty([self.ntheta,self.nzobj,self.nobj],dtype='complex64'),
                            'prb':cp.empty([self.n,self.n],dtype='complex64'),
                            'pos':cp.empty([self.ntheta, self.ndist, 2],dtype='float32')
                            }
        self.etas = {'proj':cp.empty([self.ntheta,self.nzobj,self.nobj],dtype='complex64'),
                      'prb':cp.empty([self.n,self.n],dtype='complex64'),
                      'pos':cp.empty([self.ntheta, self.ndist, 2],dtype='float32')
                      }
        
        self.voxelsize = voxelsize
        # sizes for normalization
        self.data_size = self.ntheta * self.ndist * self.nz * self.n
        self.prb_size = self.nz * self.n
        self.proj_size = self.ntheta * self.nzobj * self.nobj
        
    def gen_data(self, vars, out):
        """Generate synthetic data"""

        x = [vars['prb'], vars['proj'], vars['pos'] ]
        y = x  # forming output
        # compute functional by applying operators in reverse order
        for id in range(1, len(self.F))[::-1]:  
            y = self.F[id](y)        
        out[:] = cp.abs(y)**2
        
    def gen_ref(self, prb, out):
        """Generate synthetic reference"""
        out[:] = cp.abs(self.cl_prop.D(prb,0))**2
                            
    def BH(self, data, ref, vars):
        # keep data and initial shifts in class
        self.data = np.sqrt(data)
        self.ref = np.sqrt(ref)
        self.pos_init = vars["pos"].copy()        

        # refs to preallocated memory for gradients
        grads = self.grads
        etas = self.etas
                
        # calc init error
        self.error_debug(vars, -1)
        
        for i in range(self.niter):        
            self.gradients(vars, grads)      
                
            # scale
            for v in ["proj", "pos", "prb"]:                
                grads[v] *= self.rho[v]**2

            if i >= 0:
                # initial search direction (negative gradient)
                for v in ["proj", "pos", "prb"]:                
                    etas[v][:] = -grads[v]
            else:
                # calc beta using Hessian-weighted inner products                
                top = self.hessian(vars, grads, etas)                         
                bottom = self.hessian(vars, etas, etas)
                beta = top/bottom

                # update search direction: eta = beta * previous_eta - grad
                for v in ["proj", "pos", "prb"]:                
                    etas[v][:] = etas[v]*beta - grads[v]
            
            # calc alpha (step length)            
            top = 0
            for v in ["proj", "pos", "prb"]:                
                top -= redot(grads[v], etas[v]).get() / self.rho[v]**2     
            bottom = self.hessian(vars, etas, etas)                                        
            alpha = top/bottom

            # check approximation with the Hessian
            # self.check_approximation(vars, etas, top, bottom, alpha, i)
            
            # update variables: var = var+alpha*eta
            for v in ["proj", "pos", "prb"]:       
                vars[v]+=alpha*etas[v]         
            
            # error and visualization debug
            self.error_debug(vars, i)
            
            self.vis_debug(vars, i)
                                
        return vars

    def hessian(self, vars, grads, etas):
        """Hessian for the full functional, is a sum of 3 terms:
        1. main data fit term calcuated with the cascade rule,
        2. probe fit term,
        3. regularization term"""
        
        w = self.hessian_cascade(vars, grads, etas)
        w += self.hessian_prbfit(vars["prb"], grads["prb"], etas["prb"])
        w += self.hessian_reg(vars["proj"], grads["proj"], etas["proj"])
        return w.get()    

    def hessian_cascade(self, vars, grads, etas):
        """"Cascade computation of the hessian for the main term,
            following the composition rule (Carlsson, 2025):
            For f = F1 ◦ F2 the hessian is 
                d2f = dF1 ◦ d2F2 + d2F1 ◦ dF2
                where dF are differentials, 
                d2F are second order terms.
            The function implements it for f = F0 ◦ F1 ◦ F2 ◦ F3 ...
            parameters to functions are unified as (x,y,z,w)
        """
        
        # reorganize inputs into ordered lists for cascade traversal
        x = vars["prb"], vars["proj"], vars["pos"]
        y = grads["prb"], grads["proj"], grads["pos"]
        z = etas["prb"], etas["proj"], etas["pos"]
        w = [None,None,None]
        
        # check whether y and z share same object (avoid duplicate work)
        y_is_z = y[0] is z[0]

        for id in range(1,len(self.F))[::-1]:
            # compute d2F(dFy,dFz)+dF(d2F(y,z))                                
            w = self.d2F_dF[id](x, y, z, w)                        
            
            # propagate differentials to the next level: fx, dF(x)(y)
            fx, y = self.dF[id](x, y)  # returns (fx, dfx(y))
            if y_is_z:
                z = y
            else:
                z = self.dF[id](x, z, return_x=False)  # returns dfx(z)
            x = fx

        # outer functional
        out = self.d2F_dF[0](x, y, z, w, self.data)
        
        return out

    def gradients(self, vars, grads):
        """Full gradient, consists of 3 terms: 
        1. main data fit term calcuated with the cascade rule,
        2. probe fit term,
        """        
        
        self.gradients_cascade(vars,grads)        
        self.gradient_prbfit(grads["prb"], vars["prb"])
        self.gradient_reg(grads["proj"], vars["proj"])
        
    def gradients_cascade(self, vars, grads):
        """Cascade gradient for the main term
            following the composition rule (Carlsson, 2025):
            For f = F1 ◦ F2 the gradient is 
                gradf = dF_2^*(\nabla F_1)),
                where dF_2^* is the adjoint to the differential
            The function implements it for f = F0 ◦ F1 ◦ F2 ◦ F3 ...
            parameters to functions are unified as (x,y,z)
        """
        
        x = [vars['prb'], vars['proj'], vars['pos']]
        y = self.data
        # compute gradient by applying operators in forward order
        for id in range(len(self.gF)):  #last one computed separately because of different chunking
            y = self.gF[id](x,y)              
        grads['prb'][:] = y[0]
        grads['proj'][:] = y[1]
        grads['pos'][:] = y[2]                
            
    #### probe fit term
    def gradient_prbfit(self, grad_prb, prb):
        """Gradient with respect to the term 
        lam_prbfit|||Dprb|-ref||_2^2"""
        
        if self.lam_prbfit == 0:
            return
        tmp = self.cl_prop.D(prb, 0)
        td = self.ref * (tmp / (cp.abs(tmp)))
        grad_prb[:] += self.lam_prbfit / self.prb_size * self.cl_prop.DT((2 * (tmp - td)), 0)

    def gradient_reg(self, grad_proj, proj):
        """Gradient with respect to the regularization term (laplace)
        lam_reg ||Delta proj||_2^2"""
        
        if self.lam_reg == 0:
            return
        # process by chunk to save memory
        gg = self.laplacian_sq(proj)
        grad_proj[:] += 2 * self.lam_reg / self.proj_size * gg
        
    def hessian_prbfit(self, prb, dprb1, dprb2):
        """Hessian with respect to the term 
        lam_prbfit|||Dprb|-ref||_2^2"""

        if self.lam_prbfit == 0:
            return 0

        Dprb = self.cl_prop.D(prb,0)
        Ddprb1 = self.cl_prop.D(dprb1,0)
        Ddprb2 = self.cl_prop.D(dprb2,0)
        l0 = Dprb / (cp.abs(Dprb))
        d0 = self.ref / (cp.abs(Dprb))
        v1 = cp.sum((1 - d0) * reprod(Ddprb1, Ddprb2))
        v2 = cp.sum(d0 * reprod(l0, Ddprb1) * reprod(l0, Ddprb2))
        out = 2 * (v1 + v2)
        out = self.lam_prbfit * out / self.prb_size
        return out
    
    def hessian_reg(self, proj, dproj1, dproj2):
        """Hessian with respect to the regularization term (laplace)
        lam_reg ||Delta proj||_2^2"""
        
        if self.lam_reg == 0:
            return 0

        Lobj1 = self.laplacian(dproj1)
        Lobj2 = self.laplacian(dproj2)
        out = redot(Lobj1, Lobj2)
        return 2 * self.lam_reg * out / self.proj_size

    def F0(self, x, d):
        """In: (x0), Out: const"""
        return cp.linalg.norm(cp.abs(x) - d) ** 2 / self.data_size

    def dF0(self, x, y, d, return_x=False):
        """In: (x0,y0), Out: const"""
        
        x0 = x
        y0 = y

        tmp = 2 * (x - d * (x0 / cp.abs(x0)))
        return redot(tmp, y0) / self.data_size
    
    def d2F0(self, x, y, z, d):
        """In: (x0,y0,z0), Out: const"""
        
        x0 = x
        y0 = y
        z0 = z

        l0 = x0 / (cp.abs(x0))
        d0 = d / (cp.abs(x0))
        v1 = cp.sum((1 - d0) * reprod(y0, z0))
        v2 = cp.sum(d0 * reprod(l0, y0) * reprod(l0, z0))
        return 2 * (v1 + v2)/ self.data_size

    def d2F_dF0(self, x, y, z, w, d):
        """In: (x0,y0,z0,w0), Out: const"""
        
        x0 = x
        y0 = y
        z0 = z
        w0 = w
        
        l0 = x0 / (cp.abs(x0))
        d0 = d / (cp.abs(x0))
        v = cp.sum((1 - d0) * reprod(y0, z0))
        v += cp.sum(d0 * reprod(l0, y0) * reprod(l0, z0))        
        if w is not None:
            v +=  redot(x - d * l0, w0)
        
        return v * 2 / self.data_size
     
    def gF0(self, x, y):
        """In: x, y = F0(F1(..(x)))), Out: y0"""
        
        # calc fwd starting from 1
        for id in range(1, 4)[::-1]: 
            x = self.F[id](x)
               
        td = y * (x / (cp.abs(x)))
        y0 = (2 / self.data_size) * (x - td) 
        
        return y0
    
    ####### x0 = F1(x11,x12) = D(D(x11)\cdot x12)
    def F1(self, x):
        """In: (x11,x12), Out: x0"""

        x11, x12 = x

        
        x0 = cp.empty([x12.shape[0], self.ndist, self.nz, self.n], dtype="complex64")
        for j in range(self.ndist):
            dx11 = self.cl_prop_prb.D(x11,j)
            x0[:, j] = self.cl_prop.D(dx11 * x12[:, j], j)
        
        return x0
    
    def dF1(self, x, y, return_x=True):
        """In: (x11,x12),(y11,y12) Out: y0"""

        x11, x12 = x
        y11, y12 = y

        y0 = cp.empty([x12.shape[0], self.ndist, self.nz, self.n], dtype="complex64")
        x0 = cp.empty_like(y0) if return_x else None

        
        for j in range(self.ndist):
            dx11 = self.cl_prop_prb.D(x11,j)
            dy11 = self.cl_prop_prb.D(y11,j)
            y0[:, j] = self.cl_prop.D(dy11 * x12[:, j] + dx11 * y12[:, j], j)
            if return_x:
                x0[:, j] = self.cl_prop.D(dx11 * x12[:, j], j)

        return (x0, y0) if return_x else y0
    
    def d2F1(self, x, y, z):
        """In: (x11,x12),(y11,y12),(z11,z12) Out: y0"""

        x11, x12 = x
        y11, y12 = y
        z11, z12 = z
        y0 = cp.empty([len(y12), self.ndist, self.nz, self.n], dtype="complex64")
        
        for j in range(self.ndist):            
            dy11 = self.cl_prop_prb.D(y11,j)
            if y12 is z12:                
                y0[:, j] = 2 * self.cl_prop.D(dy11 * y12[:, j], j)
            else:
                dz11 = self.cl_prop_prb.D(z11,j)
                y0[:, j] = self.cl_prop.D(dy11 * z12[:, j], j) + self.cl_prop.D(dz11 * y12[:, j], j)

        return y0

    def d2F_dF1(self, x, y, z, w):
        """In: (x11,x12),(y11,y12),(z11,z12) Out: y0"""

        x11, x12 = x
        y11, y12 = y
        z11, z12 = z
        w11, w12 = w
        
        
        y0 = cp.empty([len(y12), self.ndist, self.nz, self.n], dtype="complex64")
        
        for j in range(self.ndist):
            dy11 = self.cl_prop_prb.D(y11,j)                           
            if y12 is z12:                
                y0[:, j] = 2 * dy11 * y12[:, j]
            else:
                dz11 = self.cl_prop_prb.D(z11,j)
                y0[:, j] = dy11 * z12[:, j] + dz11 * y12[:, j]
            
            if w11 is not None:
                dw11 = self.cl_prop_prb.D(w11,j)
                y0[:, j] += dw11 * x12[:, j]
            
            if w12 is not None:
                dx11 = self.cl_prop_prb.D(x11,j)
                y0[:, j] += dx11 * w12[:, j]

            y0[:, j] = self.cl_prop.D(y0[:, j],j)                

        return y0
    
    def gF1(self, x, y):
        """In: x=(x01,x02,x03),(y0) Out: y11,y12"""

        y0 = y
        
        # calc fwd starting from 2        
        for id in range(2, 4)[::-1]: 
            x = self.F[id](x)

        x11, x12 = x                    
        
        y11 = cp.zeros([self.nz,self.n], dtype="complex64")
        y12 = cp.empty([y0.shape[0], self.ndist, self.nz, self.n], dtype="complex64")
        for j in range(self.ndist):
            tmp = self.cl_prop.DT(y0[:, j], j)             
            tmp *= np.conj(x12[:,j])
            y11 += cp.sum(self.cl_prop_prb.DT(tmp,j), axis=0)
            dx11 = self.cl_prop_prb.D(x11,j)
            y12[:, j] = tmp * np.conj(dx11)
        return y11, y12        

    ######## (x11,x12) = F2(x21,x22) = (D(x21),e^{1j x22})
    def F2(self, x):
        """In: (x21,x22) Out: (x11,x12)"""

        x21, x22 = x
        x11 = x21
        x12 = cp.exp(1j*x22)

        return x11, x12

    def dF2(self, x, y, return_x=True):
        """In: (x21,x22),(y21,y22) Out: (x11,x12),(y11,y12)"""

        x21, x22 = x
        y21, y22 = y

        x12 = cp.exp(1j*x22)             
        y12 = x12 * 1j * y22

        x11 = x21
        y11 = y21       

        return ([x11, x12], [y11, y12]) if return_x else [y11, y12]

    def d2F2(self, x, y, z):
        """In: (x21,x22),(y21,y22),(z21,z22) Out: (y11,y12)"""

        x21, x22 = x
        y21, y22 = y
        z21, z22 = z
        
        if y22 is z22:
            y12 = x22 * (-(y22**2))
        else:
            y12 = x22 * (-y22 * z22)

        y11 = cp.zeros_like(y21)
        
        return [y11, y12]
    
    def d2F_dF2(self, x, y, z, w):
        """In: (x21,x22),(y21,y22),(z21,z22),(w21,w22) Out: (y11,y12)"""

        x21, x22 = x
        y21, y22 = y
        z21, z22 = z
        w21, w22 = w
        
        
        if y22 is z22:
            y12 = x22 * (-(y22**2))
        else:
            y12 = x22 * (-y22 * z22)

        if w22 is not None:
            y12 = y12 + cp.exp(1j*x22) * 1j * w22

        y11 = w21
                
        return [y11, y12]
    
    def gF2(self,x,y):
        """In: x(x01, x02, x03) ,(y11,y12) Out: (y21,y22)"""

        y11, y12 = y

        # calc fwd starting from 3        
        for id in range(3, 4)[::-1]: 
            x = self.F[id](x)
        x21,x22 = x

        y22 = (-1j) * y12 * cp.conj(cp.exp(1j*x22))
        y22 = y22.real if self.obj_dtype == 'float32' else y22

        y21 = y11
        
        return [y21, y22]
    
    ####### (x21,x22) = F3(x31,x32,x33) = (x31,S_{x_33}(x32))
    def F3(self, x):
        """In: (x31, x32, x33)  Out: (x21,x22)"""

        x31, x32, x33 = x

        x22 = cp.empty([len(x33), self.ndist, self.nz, self.n], dtype=self.obj_dtype)
        c = self.cl_shift.coeff(x32)
        for k in range(self.ndist):
            x22[:, k] = self.cl_shift.curlySc(c, x33[:, k], k)

        x21 = x31
        return [x21, x22]

    def dF3(self, x, y, return_x=True):
        """In: (x31, x32, x33),(y31, y32, y33)  Out: (y31, y22)"""
        
        x31, x32, x33 = x
        y31, y32, y33 = y

        y22 = cp.zeros([len(y32), self.ndist, self.nz, self.n], dtype=self.obj_dtype)
        
        c = self.cl_shift.coeff(x32)
        c1 = self.cl_shift.coeff(y32)
        if return_x:
            x22 = cp.zeros([len(x32), self.ndist, self.nz, self.n], dtype=self.obj_dtype)
            for k in range(self.ndist):
                r = x33[:, k]
                x22[:, k] = self.cl_shift.curlySc(c, r, k)

        for k in range(self.ndist):
            r = x33[:, k]
            Deltar = y33[:, k]
            y22[:, k] = self.cl_shift.dcurlySc(c, r, k, c1, Deltar)

        x21 = x31
        y21 = y31
        return ([x21, x22], [y21, y22]) if return_x else [y21, y22]

    def d2F3(self, x, y, z):
        """In: (x31, x32, x33),(y31, y32, y33),(z31, z32, z33)  Out: (y21, y22)"""
        
        x31, x32, x33 = x
        y31, y32, y33 = y
        z31, z32, z33 = z
        y22 = cp.zeros([len(y32), self.ndist, self.nz, self.n], dtype=self.obj_dtype)
        
        c = self.cl_shift.coeff(x32)
        c1 = self.cl_shift.coeff(y32)
        c2 = self.cl_shift.coeff(z32)
        for k in range(self.ndist):
            r = x33[:, k]
            Deltar_y = y33[:, k]
            Deltar_z = z33[:, k]
            y22[:, k] = self.cl_shift.d2curlySc(c, r, k, c1, Deltar_y, c2, Deltar_z)

        y21 = cp.zeros_like(y31)
        
        return [y21, y22]
    
    def d2F_dF3(self, x, y, z, w):
        """In: (x31, x32, x33),(y31, y32, y33),(z31, z32, z33),(w31, w32, w33)  Out: (y21, y22)"""
        
        x31, x32, x33 = x
        y31, y32, y33 = y
        z31, z32, z33 = z
        w31, w32, w33 = w
        
        y22 = cp.zeros([len(y32), self.ndist, self.nz, self.n], dtype=self.obj_dtype)            
        
        c = self.cl_shift.coeff(x32)        
        cy = self.cl_shift.coeff(y32)
        cz = self.cl_shift.coeff(z32)
        for k in range(self.ndist):
            y22[:, k] = self.cl_shift.d2curlySc(c, x33[:, k], k, cy, y33[:, k], cz, z33[:, k])

        if w32 is not None:
            cy = self.cl_shift.coeff(w32)                
            for k in range(self.ndist):
                y22[:, k] += self.cl_shift.dcurlySc(c, x33[:, k], k, cy, w33[:, k])      

        y21 = w31
        
        return [y21, y22]
    
    def gF3(self, x, y):     
        """In: x(x01, x02, x03) ,(y21,y22) Out: (y31,y32)"""   

        y21, y22 = y

        for id in range(4, 4)[::-1]: 
            x = self.F[id](x)
        x31,x32,x33 = x

        y32 = cp.zeros([y22.shape[0], self.nzobj, self.nobj], dtype=self.obj_dtype)
        y33 = cp.empty([y22.shape[0], self.ndist, 2], dtype="float32")
        
        c = self.cl_shift.coeff(x32)
        for k in range(self.ndist):
            Deltapsi, Deltar = self.cl_shift.dcurlySadjc(c, x33[:, k], k, y22[:, k])
            y32[:] += Deltapsi
            y33[:, k] = Deltar
        
        y32[:] = self.cl_shift.coeff(y32)        
        
        y31 = y21
        return [y31, y32, y33]       
     
    def error_debug(self, vars, i):
        """Visualization and data saving"""
        if not (i % self.err_step == 0 and self.err_step != -1):
            return
            
        err = self.min([vars["prb"], vars["proj"], vars["pos"]])        
        print(f"iter={i}: {err=:1.5e} ")                                
    
    def min(self, x):
        ## batched computation
        y = x  # forming output
        # compute functional by applying operators in reverse order
        for id in range(1, len(self.F))[::-1]:  
            y = self.F[id](y)                
        out = self.F0(y, self.data)
        
        if self.lam_prbfit>0:
            Dprb = self.cl_prop.D(x[0],0)
            out += self.lam_prbfit / self.prb_size * cp.linalg.norm(cp.abs(Dprb) - self.ref) ** 2

        if self.lam_reg>0:        
            out += self.lam_reg / self.proj_size * cp.linalg.norm(self.laplacian(x[1])) ** 2
        return out.get()

    def vis_debug(self, vars, i):
        """Visualization and data saving"""
        if not (i % self.vis_step == 0 and self.vis_step != -1):
            return

        (prb, proj, pos) = (vars["prb"], vars["proj"], vars["pos"])
        
        mshow_complex(proj[0],self.voxelsize)
        mshow_complex(proj[0,self.nzobj//2+64-192:self.nzobj//2+64+192,self.nzobj//2-324:self.nzobj//2-324+384],self.voxelsize)
        mshow_polar(prb)
        # mshow_pos(pos-self.pos_init)
    
    def check_approximation(self, vars, etas, top, bottom, alpha, i):
        """Check the minimization functional behaviour"""
        if not (i % self.vis_step == 0 and self.vis_step != -1):
            return

        (prb, proj, pos) = (vars["prb"], vars["proj"], vars["pos"])
        (dprb, dproj, dpos) = (etas["prb"], etas["proj"], etas["pos"])

        npp = 5
        t = np.linspace(0, 2 * alpha, npp).astype("float32")
        err_real = np.zeros(npp)
        err_approx = np.zeros(npp)
        projt = np.empty_like(proj)
        prbt = cp.empty_like(prb)
        post = np.empty_like(pos)
        
        for k in range(0, npp):            
            projt[:] =  proj+t[k]*dproj
            prbt[:] =  prb+t[k]*dprb
            post[:] =  pos+t[k]*dpos
            err_real[k] = self.min([prbt, projt, post])
        
        err_approx = self.min([prb, proj, pos]) - top * t + 0.5 * bottom * t**2
        mshow_approx(t,err_real,err_approx)

    def laplacian(self, proj):
        """2D Laplacian"""
                
        stencil = cp.array([1, -2, 1]).astype("float32")
        out = ndimage.convolve1d(proj,stencil,axis=1)
        out[:] += ndimage.convolve1d(proj, stencil, axis=2)

        return out
    
    def laplacian_sq(self, proj):
        """twice 2D Laplacian"""
                
        stencil = cp.array([1, -4, 6, -4, 1]).astype("float32")
        out = ndimage.convolve1d(proj,stencil,axis=1)
        out[:] += ndimage.convolve1d(proj, stencil, axis=2)

        return out