import numpy as np

def c_adv_x(u:float,c:float,dx:float,Nx:int,Ny:int,ng:int):
    """
    Function called by below sca_trans   

    """
       
    # Average c to cell face
    c_left = 0.5*(c[0:-1-1,:] + c[1:-1,:]);
    c_right = 0.5*(c[1:-1,:] + c[2:,:]);

    # Combine
    endi = np.shape(u)[0]
    #endj = np.shape(u)[1]
    uc_left = np.multiply(u[1:endi-2,:],c_left);
    uc_right = np.multiply(u[2:endi-1,:],c_right);
 
    # Take derivative
    udcdx = np.zeros([Nx+2*ng,Ny+2*ng]);
    udcdx[1:-1,:] = (uc_right-uc_left)/dx;
    
    return udcdx

def c_adv_y(v:float,c:float,dy:float,Nx:int,Ny:int,ng:int):
    """
    Function called by below sca_trans   

    """
    
    # Average c to cell face
    c_down = 0.5*(c[:,0:-1-1] + c[:,1:-1]);
    c_up = 0.5*(c[:,1:-1] + c[:,2:]);
    
    # Combine
    #endi = np.shape(v)[0]
    endj = np.shape(v)[1]
    vc_down = np.multiply(v[:,1:endj-2],c_down);
    vc_up = np.multiply(v[:,2:endj-1],c_up);

    # Take derivative
    vdcdy = np.zeros([Nx+2*ng,Ny+2*ng]);
    vdcdy[:,1:-1] = (vc_up-vc_down)/dy;

    return vdcdy

def c_diff_x(c:float,dx:float,Nx:int,Ny:int,ng:int):
    """
    Function called by below sca_trans   

    """    
        
    # Initialize
    dc2dx2 = np.zeros([Nx+2*ng,Ny+2*ng])
    end = np.shape(dc2dx2)[0]
    
    # Take scalar derivative
    dc2dx2[1:end-1,:] = (c[0:end-2,:] + c[2:end,:]
                         - 2*c[1:end-1,:])/(dx*dx);

    
    
    return dc2dx2

def c_diff_y(c:float,dy:float,Nx:int,Ny:int,ng:int):
    """
    Function called by below sca_trans   

    """    
    
    # Initialize
    dc2dy2 = np.zeros([Nx+2*ng,Ny+2*ng])
    end = np.shape(dc2dy2)[1]
    
    # Take scalar derivative
    dc2dy2[:,1:end-1] = (c[:,0:end-2] + c[:,2:end]
                         - 2*c[:,1:end-1])/(dy*dy);

    return dc2dy2

def sca_trans(Nx:int,Ny:int,Lx:float,Ly:float,dt:float,T:float,BC:complex,IC_choice:int,nu:float,ng:int,u:float,v:float,ii:float,c:float):
    """
    Scalar transport incorporated into the BINS code
    dc/dt + u*dc/dx + v*dc/dy = D*(d^2c/dx^2 + d^2c/dy^2)
    
    Discretization scheme: forward euler, 2nd order central in space

    Returns
    -------
    cnew : TYPE
        Updated value of c

    """
    
    # Grid spacing
    dx = Lx/(Nx-1);
    dy = Lx/(Ny-1);

    # Form derivatives
    udcdx = c_adv_x(u,c,dx,Nx,Ny,ng);
    vdcdy = c_adv_y(v,c,dy,Nx,Ny,ng);
    dc2dx2 = c_diff_x(c,dx,Nx,Ny,ng);
    dc2dy2 = c_diff_y(c,dy,Nx,Ny,ng);

    # gather terms
    cnew = c[:,:] + dt*( - udcdx - vdcdy + nu*(dc2dx2 + dc2dy2));

    return cnew
