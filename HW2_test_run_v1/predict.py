import numpy as np

def predict(u:float,v:float,u_advection:float,v_advection:float,
            u_diffusion:float,v_diffusion:float,v_gravity:float,dt:float): 
    """
    estimates the velocity at the next time step
    the prediction step is discussed in Section 3.2, starting on page 5
    the time stepping method used here (forward Euler) is 
    discussed in Section 3.3.1 on page 7
    this function solves equations 1 and 2 for u* (page 5)

    Parameters
    ----------
    u : float
        x-direction velocity, size [Nx+2*ng+1   Ny+2*ng]
    v : float
        y-direction velocity, size [Nx+2*ng   Ny+2*ng+1]
    u_advection : float
        x-direction advection term, size [Nx+2*ng+1   Ny+2*ng]
    v_advection : float
        y-direction advection term, size [Nx+2*ng   Ny+2*ng+1]
    u_diffusion : float
        x-direction diffusion term, size [Nx+2*ng+1   Ny+2*ng]
    v_diffusion : float
        y-direction diffusion term, size [Nx+2*ng   Ny+2*ng+1]
    v_gravity : float
        DESCRIPTION.
    dt : float
        timestep

    Returns
    -------
    uStar : float
        x-direction predicted velocity at next time step, size [Nx+2*ng+1   Ny+2*ng]
    vStar : float
        y-direction predicted velocity at next time step, size [Nx+2*ng   Ny+2*ng+1]

    """
    
    uStar = u + (-u_advection + u_diffusion)*dt
    vStar = v + (-v_advection - v_gravity + v_diffusion)*dt
    
    return uStar,vStar