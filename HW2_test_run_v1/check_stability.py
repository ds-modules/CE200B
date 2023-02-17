import numpy as np
    
def check_stability(u:float,v:float,c:float,llx:int,lly:int,ulx:int,uly:int,hx:float,hy:float,nu:float,dt:float,t:float,check:int): 
    """
    

    Parameters
    ----------
    hx,hy : float
        spatial step
    nu : float
        molecular viscosity
    dt : float
        timestep
    t : float
        currrent time
    check : int
        flag related to the check is initially 0

    Returns
    -------
    check : TYPE
        flag related to the check

    """
    
    if check == 0:
        
        Pe_x = np.abs(u[llx-1:ulx,lly-1:uly]*(hx/nu))
        max_x = np.amax(Pe_x)
        Pe_y = np.abs(v[llx-1:ulx,lly-1:uly]*(hy/nu))
        max_y = np.amax(Pe_y)
        
        if (Pe_x>2).sum() > 0:
            print('t = ' + str(t))
            print('Pe(x)-max = '+ str(max_x))
            print('--> WARNING: cell Pe limit reached; nu may be too small <--')
            check = 1
        elif (Pe_y>2).sum() > 0:
            print('t = ' + str(t))
            print('Pe(y)-max = '+ str(max_y))
            print('--> WARNING: cell Pe limit reached; nu may be too small <--')
            check = 1
            
    if check == 0:
        
        rx = nu*dt/(hx*hx)
        ry = nu*dt/(hy*hy)
        Cr_x = np.abs(u[llx-1:ulx,lly-1:uly]*(dt/hx))
        max_x = np.amax(Cr_x)
        Cr_y = np.abs(v[llx-1:ulx,lly-1:uly]*(dt/hy))
        max_y = np.amax(Cr_y)
        
        if (Cr_x>0.5).sum() > 0:
            print('t = ' + str(t))
            print('Cr(x)-max = '+ str(max_x))
            print('--> WARNING: Cr(x) limit reached <--')
            check = 1
        elif (Cr_y>0.5).sum() > 0:
            print('t = ' + str(t))
            print('Cr(y)-max = '+ str(max_x))
            print('--> WARNING: Cr(y) limit reached <--')
            check = 1
        elif rx>0.25:
            print('t = ' + str(t))
            print('r = '+ str(rx))
            print('--> WARNING: r limit reached <--')
            check = 1
        elif ry>0.25:
            print('t = ' + str(t))
            print('r = '+ str(ry))
            print('--> WARNING: r limit reached <--')
            check = 1
    
    # Return flag
    return check