import numpy as np

def adv_duudx(u:float,hx:float): 
    """
    Calculate the quantity d(uu)/dx for use in the x-direction advection term
    for details, see Section 4.1 (page 10)

    Parameters
    ----------
    u : float
        x-direction velocity, size [Nx+2*ng+1   Ny+2*ng]
    hx : float
        spatial step in x direction

    Returns
    -------
    duudx : float
        derivative of quantity uu with respect to x, size [Nx+2*ng+1   Ny+2*ng]

    """
    
    # average u to cell centers
    endi = np.shape(u)[0]
    u_right = 0.5*(u[2:endi,:] + u[1:endi-1,:])
    u_left = 0.5*(u[1:endi-1,:] + u[0:endi-2,:])

    # combine
    uu_right = np.multiply(u_right,u_right)
    uu_left = np.multiply(u_left,u_left)
    
    # take derivative
    duudx = np.zeros_like(u)
    duudx[1:endi-1,:] = (uu_right - uu_left)/hx
    
    return duudx

def adv_duvdy(u:float,v:float,hy:float): 
    """
    calculate the quantity d(uv)/dy for use in the x-direction advection term
    for details, see Section 4.1 (page 10)

    Parameters
    ----------
    u : float
        x-direction velocity, size [Nx+2*ng+1   Ny+2*ng]
    v : float
        y-direction velocity, size [Nx+2*ng   Ny+2*ng+1]
    hy : float
        spatial step in y direction

    Returns
    -------
    duvdy : float
        derivative of quantity uv with respect to y, size [Nx+2*ng+1   Ny+2*ng]
        
    """
    
    # average u to corners
    endi = np.shape(u)[0]
    endj = np.shape(u)[1]
    u_up = 0.5*(u[1:endi-1,2:endj]+u[1:endi-1,1:endj-1])
    u_down = 0.5*(u[1:endi-1,1:endj-1]+u[1:endi-1,0:endj-2])
    
    # average v to corners
    endi = np.shape(v)[0]
    endj = np.shape(v)[1]
    v_up = 0.5*(v[0:endi-1,2:endj-1]+v[1:endi,2:endj-1]);
    v_down = 0.5*(v[0:endi-1,1:endj-2]+v[1:endi,1:endj-2])
    
    # combine
    uv_up = np.multiply(u_up,v_up)
    uv_down = np.multiply(u_down,v_down)
    
    # take derivative
    duvdy = np.zeros_like(u)
    endi = np.shape(duvdy)[0]
    endj = np.shape(duvdy)[1]
    duvdy[1:endi-1,1:endj-1] = (uv_up-uv_down)/hy
    
    return duvdy

def adv_dvudx(u:float,v:float,hx:float): 
    """
    calculate the quantity d(vu)/dx for use in the y-direction advection term
    for details, see Section 4.1 (page 10)

    Parameters
    ----------
    u : float
        x-direction velocity, size [Nx+2*ng+1   Ny+2*ng]
    v : float
        y-direction velocity, size [Nx+2*ng   Ny+2*ng+1]
    h : float
        spatial step in y direction

    Returns
    -------
    dvudx : float
        derivative of quantity vu with respect to y, size  [Nx+2*ng   Ny+2*ng+1]
        
    """
    
    # average v to corners
    endi = np.shape(v)[0]
    endj = np.shape(v)[1]
    v_right = 0.5*(v[2:endi,1:endj-1]+v[1:endi-1,1:endj-1])
    v_left = 0.5*(v[1:endi-1,1:endj-1]+v[0:endi-2,1:endj-1])
    
    # average u to corners
    endi = np.shape(u)[0]
    endj = np.shape(u)[1]
    u_right = 0.5*(u[2:endi-1,1:endj]+u[2:endi-1,0:endj-1]);
    u_left = 0.5*(u[1:endi-2,1:endj]+u[1:endi-2,0:endj-1])
    
    # combine
    vu_right = np.multiply(v_right,u_right)
    vu_left = np.multiply(v_left,u_left)
    
    # take derivative
    dvudx = np.zeros_like(v)
    endi = np.shape(dvudx)[0]
    endj = np.shape(dvudx)[1]
    dvudx[1:endi-1,1:endj-1] = (vu_right-vu_left)/hx
    
    return dvudx

def adv_dvvdy(v:float,hy:float): 
    """
    Calculate the quantity d(vv)/dy for use in the y-direction advection term
    for details, see Section 4.1 (page 10)

    Parameters
    ----------
    v : float
        y-direction velocity, size [Nx+2*ng   Ny+2*ng+1]
    h : float
        spatial step in y direction

    Returns
    -------
    dvvdy : float
        derivative of quantity vv with respect to y, size [Nx+2*ng   Ny+2*ng+1]

    """
    
    # average u to cell centers
    endj = np.shape(v)[1]
   
    v_up = 0.5*(v[:,2:endj] + v[:,1:endj-1])
    v_down = 0.5*(v[:,1:endj-1] + v[:,0:endj-2])

    # combine
    vv_up = np.multiply(v_up,v_up)
    vv_down = np.multiply(v_down,v_down)
    
    # take derivative
    dvvdy = np.zeros_like(v)
    dvvdy[:,1:endj-1] = (vv_up - vv_down)/hy
    
    return dvvdy
    
def advection(u:float,v:float,hx:float,hy:float): 
    """
    Calculates the advection term of the Navier-Stokes equation
    the advection term is discussed in Section 3.2, starting on page 5
    this function calculates the first two terms of the
    right hand side of equations 1 and 2 (page 5)
    
    Functions called by advection: adv_duudx, adv_duvdy, adv_dvudx, adv_dvvdy
        
    These functions are very similar, but differ slightly due to the staggered grid
    For an overview of the advection term derivatives, see Section 3.3.2 on page 7

    Parameters
    ----------
    u : float
        x-direction velocity, size [Nx+2*ng+1   Ny+2*ng]
    v : float
        y-direction velocity, size [Nx+2*ng   Ny+2*ng+1]
    hx,hy : float
        dx and dy

    Returns
    -------
    u_advection : float
        x-direction advection term of the Navier-Stokes equation, size [Nx+2*ng+1   Ny+2*ng]
    v_advection : float
        y-direction advection term of the Navier-Stokes equation, size [Nx+2*ng   Ny+2*ng+1]

    """
    
    # calculate derivatives
    duudx = adv_duudx(u,hx)
    duvdy = adv_duvdy(u,v,hy)
    dvudx = adv_dvudx(u,v,hx)
    dvvdy = adv_dvvdy(v,hy)
    
    # form advection term
    u_advection = duudx + duvdy
    v_advection = dvudx + dvvdy
    
    return u_advection,v_advection
