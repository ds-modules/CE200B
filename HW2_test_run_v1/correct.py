import numpy as np
import matplotlib.pyplot as plt

def correct_dpdx(p:float,dx:float):
    """
    calculates dpdx on cell faces from p on cell centers
    for details, see Section 4.1 (page 10)

    Parameters
    ----------
    p : float
        pressure, size [N+2*ng   N+2*ng]
    dx : float
        spatial step

    Returns
    -------
    dpdx : float
        derivative of pressure in the x direction, size [Nx+2*ng+1   Ny+2*ng]

    """
    
    # initialize to correct size
    size_pi = np.shape(p)[0]
    size_pj = np.shape(p)[1]    
    dpdx = np.zeros((size_pi+1,size_pj))

    # take derivative
    endi = np.shape(dpdx)[0]
    dpdx[1:endi-1,:] = (p[1:size_pi,:] - p[0:size_pi-1,:])/dx
    
    return dpdx

def correct_dpdy(p:float,dy:float):
    """
    calculates dpdx on cell faces from p on cell centers
    for details, see Section 4.1 (page 10)

    Parameters
    ----------
    p : float
        pressure, size [Nx+2*ng   Ny+2*ng]
    dy : float
        spatial step

    Returns
    -------
    dpdy : float
        derivative of pressure in the y direction, size [Nx+2*ng   Ny+2*ng+1]

    """
    
    # initialize to correct size
    size_pi = np.shape(p)[0]
    size_pj = np.shape(p)[1]    
    dpdy = np.zeros((size_pi,size_pj+1))

    # take derivative
    endj = np.shape(dpdy)[1]
    dpdy[:,1:endj-1] = (p[:,1:size_pj] - p[:,0:size_pj-1])/dy
    
    return dpdy

def checkDiv(u:float,v:float,dx:float,dy:float,ng:int,Nx:int,Ny:int):
    """
    calculates divergence of updated velocity field
    maxDiv should be zero. If maxDiv is not zero, displays plot of 
    divergence to help debug and stops program
    
    for an overview of first derivatives, see Section 3.3.2 on page 7
    for details, see Section 4.1 on page 10

    Parameters
    ----------
    u : float
        x-direction velocity, size [Nx+2*ng+1   Ny+2*ng]
    v : float
        y-direction velocity, size [Nx+2*ng   Ny+2*ng+1]
    dx,dy : float
        spatial step
    ng : int
        number of ghostcells 
    Nx,Ny : int
        number of cells in x and y 

    Returns
    -------
    None.
  
    """
    
    # calculate divergence
    # endi = np.shape(u)[0]
    # dudx = (u[1:endi,:]-u[0:endi-1,:])/dx
    # endj = np.shape(v)[1]
    # dvdy = (v[:,1:endj]-v[:,0:endj-1])/dy
    dudx = (u[1:,:]-u[:-1,:])/dx;
    dvdy = (v[:,1:]-v[:,:-1])/dy;
    div = dudx + dvdy

    # check for divergence = 0
    maxDiv = np.amax(np.abs(div[ng:ng+Nx,ng:ng+Ny]))
    if maxDiv > 1e-8:
        # print the error message
        print('Maximum divergence is: ' + str(maxDiv) + '\n')
        
        # plot divergence
        fig, main_ax = plt.subplots()
        fig.set_size_inches(8, 8)
        main_ax.set_xlabel('x')
        main_ax.set_ylabel('y')
        main_ax.set_title('divergence')
        data = div[ng:ng+Nx,ng:ng+Ny]
        main_ax.imshow(data,vmin=-1e-8, vmax=1e-8)
        main_ax.set_xlim(1, Nx)
        main_ax.set_ylim(1, Ny)
        
        # Raise error
        raise Exception('Divergence too large')

def correct(uStar:float,vStar:float,newP:float,dx:float,dy:float,ng:float,Nx:int,Ny:int,dt:float):
    """
    correct the velocity with the new pressure field
    this forces the velocity field to be divergence free
    the correction step is described in Section 3.2, starting on page 5
    this function solves equations 4 and 5 (page 6)
    
    functions called by correct: correct_dpdx, correct_dpdy, checkDiv
        
    like the advection terms, the pressure derivatives are very similar
    but differ due to the staggered grid
    for an overview of the first derivatives, see Section 3.3.2 on page 7

    Parameters
    ----------
    uStar : float
        predicted x-direction velocity at next time step, size [Nx+2*ng+1   Ny+2*ng]
    vStar : float
        predicted y-direction velocity at next time step, size [Nx+2*ng   Ny+2*ng+1]
    newP : float
        pressure at next time step
    dx,dy : float
        dx and dy
    ng : float
        molecular viscosity
    Nx,Ny : int
        number of cells in x and y
    dt : float
        timestep

    Returns
    -------
    uNext : float
        x-direction velocity at next time step, size [Nx+2*ng+1   Ny+2*ng]
    vNext : float
        y-direction velocity at next time step, size [Nx+2*ng   Ny+2*ng+1]

    """
    
    # calculate new pressure derivatives
    dpdx = correct_dpdx(newP,dx)
    dpdy = correct_dpdy(newP,dy)
    
    # correct velocity to be divergence-free with new pressure derivatives
    uNext = uStar - dpdx * dt
    vNext = vStar - dpdy * dt
    
    # check that velocity is divergence-free
    checkDiv(uNext,vNext,dx,dy,ng,Nx,Ny)
    
    return uNext,vNext
    
