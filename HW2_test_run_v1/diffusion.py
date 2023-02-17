import numpy as np

def secondDeriv(u:float,dir:int,h:float): 
    """
    calculate second derivative
    This is much less complicated than the adv_dXXdX functions
    because all output derivatives are located in the same part
    of the cell as the input variable.
    for details, see Section 3.3.2 (page 7)

    Parameters
    ----------
    u : float
        variable to take derivative of (variable dimension)
    dir : int
        direction of derivative, 1 for x direction, 2 for y direction
    h : float
        spatial step

    Returns
    -------
    derivOut : float
        second derivative of input variable u (same size as u)

    """
    
    derivOut = np.copy(u)
    if dir == 1:
        endi = np.shape(u)[0]
        derivOut[1:endi-1,:] = (u[0:endi-2,:] + u[2:endi,:] - 2*u[1:endi-1,:])/(h*h)
    else:
        endj = np.shape(u)[1]
        derivOut[:,1:endj-1] = (u[:,0:endj-2] + u[:,2:endj] - 2*u[:,1:endj-1])/(h*h)
    
    return derivOut
       
def diffusion(u:float,v:float,hx:float,hy:float,nu:float):
    """
    calculates the diffusion term of the Navier-Stokes equation
    the diffusion term is discussed in Section 3.2, starting on page 5
    this function calculates the second terms of the right 
    hand size of equations 1 and 2 (page 5)
    

    Parameters
    ----------
    u : float
        x-direction velocity, size [Nx+2*ng+1   Ny+2*ng]
    v : float
        y-direction velocity, size [Nx+2*ng   Ny+2*ng+1]
    hx,hy : float
        dx and dy
    nu : float
        kinematic viscosity

    Returns
    -------
    u_diffusion : float
        x-direction diffusion term of the Navier-Stokes equation, size [Nx+2*ng+1   Ny+2*ng]
    v_diffusion : float
        y-direction diffusion term of the Navier-Stokes equation, size [Nx+2*ng   Ny+2*ng+1]

    """
    
    # calculate second derivatives of u
    du2dx2 = secondDeriv(u,1,hx)
    du2dy2 = secondDeriv(u,2,hy)
    
    # form u diffusion term
    u_diffusion = nu*(du2dx2 + du2dy2)
    
    # calculate second derivatives of v
    dv2dx2 = secondDeriv(v,1,hx)
    dv2dy2 = secondDeriv(v,2,hy)
    
    # form v diffusion term
    v_diffusion = nu*(dv2dx2 + dv2dy2)
    
    return u_diffusion,v_diffusion
    
