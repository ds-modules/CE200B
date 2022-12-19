import numpy as np
from poissonSolve import poissonSolve

def divergence(u:float,v:float,dx:float,dy:float):
    """
    calculates the divergence
    for an overview of first derivatives, see Section 3.3.2 on page 7
    for details, see Section 4.1 on page 10

    Parameters
    ----------
    u : float
        x-direction velocity, size [Nx+2*ng+1   Ny+2*ng]
    v : float
        y-direction velocity, size [Nx+2*ng   Ny+2*ng+1]
    dx : float
    dy : float
        spatial step

    Returns
    -------
    divOut : float
        divergence of u and v, size [Nx+2*ng   Ny+2*ng]

    """

    endi = np.shape(u)[0]
    endj = np.shape(v)[1]
    # dudx = (u[1:endi,:]-u[0:endi-1,:])/dx;
    # dvdy = (v[:,1:endj]-v[:,0:endj-1])/dy;
    dudx = (u[1:,:]-u[:-1,:])/dx;
    dvdy = (v[:,1:]-v[:,:-1])/dy;
    # Calculate divergence
    divOut = dudx + dvdy

    return divOut
    
def pressure(uStar:float,vStar:float,p:float,dx:float,dy:float,dt:float,ng:int,BC:complex,Nx:int,Ny:int,A:float): 
    """
    solves for the pressure using the requirement that the velocity
    field must be divergence free:
    Laplacian(p) =  divergence(uStar) / dt
    the Poisson pressure solve is discussed in Section 3.2, starting on page 5
    this function solves equation 3 for p^n+1 (page 5)

    Parameters
    ----------
    uStar : float
        predicted x-direction velocity at next time step, size [N+2*ng+1   N+2*ng]
    vStar : float
        predicted y-direction velocity at next time step, size [N+2*ng   N+2*ng+1]
    p : float
        pressure, size [N+2*ng  N+2*ng]
    dx,dy : float
        dx and dy
    dt : float
        timestep
    ng : int
        number of guardcells
    BC : complex
        boundary conditions, size [4]
    Nx, Ny : int
        number of cells in each direction
    A : float
        matrix defining Poisson operator

    Returns
    -------
    newP : float
        pressure at next time step

    """
    
    # Initialize
    newP = np.copy(p)

    # calculate right hand side of Poisson equation
    div = divergence(uStar,vStar,dx,dy)
    rhs = div/dt

    # solve Poisson equation for pressure
    newP = poissonSolve(rhs,dx,dy,BC,ng,Nx,Ny,A)
    
    # Return new pressure
    return newP
