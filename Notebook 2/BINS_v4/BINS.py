# Berkeley Incompressible Navier-Stokes solver
# aka Basic Incompressible Navier-Stokes = BINS
# an educational tool
# 
# originaly written in Matlab, the prototyping language of engineers
#
# Goodfriend, 2013; updated 11/18/2017 Tina K.C.
# updated to python, 2021, Ajay Harish

import numpy as np 
from scipy.io import savemat
import sys
import matplotlib.pyplot as plt

from IC import IC
from fillBC import fillBC
from make_matrix import make_matrix
from sca_trans import sca_trans
from temperature import temperature
from advection import advection
from diffusion import diffusion
from predict import predict
from pressure import pressure
from correct import correct
from check_stability import check_stability
from check_masscons import check_masscons
from plotScaTrans import plotScaTrans
from plotSoln import plotSoln
from benchmark import benchmark

def BINS(Nx:int,Ny:int,Lx:float,Ly:float,dt:float,T:float,BC:float,IC_choice:int,nu:float,ng:int, PGx:float, PGy:float, IBM:int):
    """
    BINS = Basic Incompressible Navier-Stokes
    BINS is an educational tool originally written in Matlab
    
    Parameters
    ------------
    N: number of cells in x 
    N: number of cells in y 
    L: physical dimension in x
    L: physical dimension in y
    dt: timestep
    T: final time
    BC: boundary conditions, size [4]
    IC_choice: choice of initial conditions defined in IC.py
    nu: molecular viscosity
    ng: number of ghost cells
    rho: density
    PGx, PGy: constant external pressure gradient in x,y
    IBM: flag to turn on IBM option for immersed obstacle
    NOTE: null density == unity, so it does not appear explicitly
    see sample scripts like cavity.py, couette.m, etc. to run BINS
    
    Returns
    --------
    None
    
    """

    # Some parameters    
    dx = Lx/(Nx-1)
    llx = ng+1
    ulx = ng+Nx
    
    dy = Ly/(Ny-1)
    lly = ng+1
    uly = ng+Ny
    
    
    # Initial conditions
    # to change the IC for the scalar field, see IC.py
    u,v,p,c,temp,temp_null,aa,inc = IC(Lx,Ly,Nx,Ny,ng,IC_choice,T,dt)
    
    # Enforce boundary conditions
    u,v,p,c,temp = fillBC(u,v,p,c,temp,ng,Nx,Ny,BC,IBM)
    
    # Define the Poisson operator for pressure solution
    A = make_matrix(Nx,Ny,dx,dy,BC)
    # A = make_matrix2(Nx,Ny,dx,dy,BC)
 
    # Others
    t = 0
    ii = 1
    check = 0 # For stability checking
    c_plot = np.array([0])
    mm = 0
    c_xy = [] # initialize breakthrough curve
    t_xy = []
    
    # Check till end of time
    while t < T:
        
        # (2) Step forward in time
    
        # (3) Solve scalar transport (using velocities at old timestep)
        cnew = sca_trans(Nx,Ny,Lx,Ly,dt,T,BC,IC_choice,nu,ng,u,v,ii,c)
        c = np.copy(cnew)
        
        # Enforce boundary conditions
        u,v,p,c,temp = fillBC(u,v,p,c,temp,ng,Nx,Ny,BC,IBM)
                
        # (4) Temperature effects for IC_choice = 6
        if aa == 1:
            # Construct the new temperature array
            tempnew = temperature(Nx,Ny,Lx,Ly,dt,T,BC,IC_choice,nu,ng,u,v,temp)
            temp = np.copy(tempnew)
            u,v,p,c,temp = fillBC(u,v,p,c,temp,ng,Nx,Ny,BC,IBM)
            
            # Construct the new gravity term
            alpha = 5e-4
            g = 9.81
            rho = 1-alpha*(temp-temp_null)
            v_gravity = np.zeros((Nx+2*ng,Ny+2*ng+1))
            endj = np.shape(v_gravity)[1]
            v_gravity[:,0:endj-1] = rho*g
            # Enforce boundary conditions
            u,v_gravity,p,c,temp = fillBC(u,v_gravity,p,c,temp,ng,Nx,Ny,BC,IBM)
        else:
            v_gravity = np.zeros((Nx+2*ng,Ny+2*ng+1))
        
        # (5) predict next step velocity with advection 
        # and diffusion terms (equations 1 and 2)
        # Advection term
        u_advection,v_advection = advection(u,v,dx,dy)
        # Diffusion term
        u_diffusion, v_diffusion = diffusion(u,v,dx,dy,nu)
        
        # estimate next time step velocity
        uStar,vStar = predict(u,v,u_advection,v_advection,u_diffusion,v_diffusion,v_gravity,dt) 

		# enforce boundary conditions
        uStar,vStar,p,c,temp = fillBC(uStar,vStar,p,c,temp,ng,Nx,Ny,BC,IBM)
        
        # (6) project the velocity field onto a divergence-free field 
        # to get the pressure (equation 3)
        newP = pressure(uStar,vStar,p,dx,dy,dt,ng,BC,Nx,Ny,A)
        
        #  enforce boundary conditions
        uStar,vStar,newP,c,temp = fillBC(uStar,vStar,newP,c,temp,ng,Nx,Ny,BC,IBM)
               
        # (7) correct the velocity to be divergence-free using the updated pressure
        #NOTE: the code for a pressure gradient forcing needs to be added in correct.py, and in the IC.py file
        newU,newV = correct(uStar,vStar,newP,dx,dy,ng,Nx,Ny,dt) # add PGx and PGy as inputs here
        u = np.copy(newU)
        v = np.copy(newV)
        p = np.copy(newP)
        # enforce boundary conditions
        u,v,p,c,temp = fillBC(u,v,p,c,temp,ng,Nx,Ny,BC,IBM)
        
        # (8) check cell Peclet number, excluding ghost points
        # (9) check r, Cr_x, Cr_y, excluding ghost points
        check = check_stability(u,v,c,llx,lly,ulx,uly,dx,dy,nu,dt,t,check)
        
        # (10) scalar mass conservation check
        c_plot = check_masscons(ng,Nx,Ny,c,ii,dt,t,T,c_plot)
        
        # Update the time
        t = t+dt
        ii = ii+1
       
        if ii%inc == 0:
            print('time = '+str(t))
            plotSoln(u,v,ng,Lx,Ly,Nx,Ny,dx,dy,nu,t,T,IC_choice)
            c_xy_i = plotScaTrans(Nx,Ny,Lx,Ly,t,ng,c) # consider toggling the colorbar axis limitation
            c_xy.append(c_xy_i)
            t_xy.append(t)

    # Plot final fields including benchmark for cavity flow case
    plotSoln(u,v,ng,Lx,Ly,Nx,Ny,dx,dy,nu,t,T,IC_choice)
    
    # uncomment to plot breakthrough curve
    fig30, main_ax30 = plt.subplots()
    fig30.set_size_inches(8, 8)
    main_ax30.scatter(t_xy,c_xy)
    main_ax30.set_title("Breakthrough curve at x~0.5, y~0.1")
    
    # Save end time data for later plotting
    T = t
    mdic = {"u":u,"v":v,"p":p,"c":c,"ng":ng,"nu":nu,"Lx":Lx,"Ly":Ly,"Nx":Nx,"Ny":Ny,"T":T}
    savemat("BINS_output.mat",mdic)
       
