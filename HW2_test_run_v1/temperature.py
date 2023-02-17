import numpy as np

def temp_adv_x(u:float,temp:float,dx:float,Nx:int,Ny:int,ng:int):
    """
    Function called in below temperature
    """
    
    # average temp to cell face
    endi = np.shape(temp)[0]
    t_left = 0.5*(temp[0:-1-1,:] + temp[1:-1,:])
    t_right = 0.5*(temp[1:-1,:] + temp[2:,:])
      
    # combine
    endi = np.shape(u)[0]
    ut_left = np.multiply(u[1:endi-2,:],t_left)
    ut_right = np.multiply(u[2:endi-1,:],t_right)

    # take derivative
    udtdx = np.zeros((Nx+2*ng, Ny+2*ng))
    udtdx[1:-1,:] = (ut_right-ut_left)/dx
    
    return udtdx

def temp_adv_y(v:float,temp:float,dy:float,Nx:int,Ny:int,ng:int):
    """
    Function called in below temperature
    """
    
    # average temp to cell face
    endj = np.shape(temp)[1]
    t_down = 0.5*(temp[:,0:-1-1] + temp[:,1:-1])
    t_up = 0.5*(temp[:,1:-1] + temp[:,2:])
    
    # combine
    endj = np.shape(v)[1]
    vt_down = np.multiply(v[:,1:endj-2],t_down)
    vt_up = np.multiply(v[:,2:endj-1],t_up)
    
    # take derivative
    vdtdy = np.zeros((Nx+2*ng,Ny+2*ng))
    vdtdy[:,1:-1] = (vt_up-vt_down)/dy
    
    return vdtdy

def temp_diff_x(temp:float,dx:float,Nx:int,Ny:int,ng:int):  #TINA changed to dx,Nx

    # take scalar derivative
    dtemp2dx2 = np.zeros((Nx+2*ng,Ny+2*ng))
    endi = np.shape(temp)[0]
    dtemp2dx2[1:endi-1,:] = (temp[0:endi-2,:] + temp[2:endi,:]
                                       - 2*temp[1:endi-1,:])/(dx*dx)

    return dtemp2dx2

def temp_diff_y(temp:float,dy:float,Nx:int,Ny:int,ng:int):  #TINA
    
    # take derivative
    dtemp2dy2 = np.zeros((Nx+2*ng,Ny+2*ng))
    endj = np.shape(temp)[1]
    dtemp2dy2[:,1:endj-1] = (temp[:,0:endj-2] + temp[:,2:endj]
                                       - 2*temp[:,1:endj-1])/(dy*dy)

    return dtemp2dy2
        
def temperature(Nx:int,Ny:int,Lx:float,Ly:float,dt:float,T:float,BC:complex,IC_choice:int,nu:int,ng:int,u:float,v:float,temp:float):  #TINA edited
    """
    Density effects incorporated into the BINS code
    dtemp/dt + u*dtemp/dx + v*dtemp/dy = kappa*(d^2temp/dx^2 + d^2temp/dy^2)
    Discretization scheme: forward euler, 1st order/2nd order central in space
    
    Parameters
    -----------
        u: x-direction velocity, size [Nx+2*ng+1   Ny+2*ng] #TINA - correct?
        v: y-direction velocity, size [Nx+2*ng   Ny+2*ng+1]
        temp: temperature at previous time step, size [Nx+2*ng   Ny+2*ng]
        dt: time step
    
    Returns
    --------
        tempnew: temperature at next time step    
    
    """
    
    # Grid spacing
    dx = Lx/(Nx - 1) 
    dy = Ly/(Ny - 1) 
    
    # Form derivatives
    udtempdx = temp_adv_x(u,temp,dx,Nx,Ny,ng)
    vdtempdy = temp_adv_y(v,temp,dy,Nx,Ny,ng)
    dtemp2dx2 = temp_diff_x(temp,dx,Nx,Ny,ng)
    dtemp2dy2 = temp_diff_y(temp,dy,Nx,Ny,ng)
    
    # Gather terms
    kappa = 1.43e-07
    tempnew = temp + dt * (- udtempdx - vdtempdy + kappa * (dtemp2dx2 + dtemp2dy2))
    
    return tempnew
