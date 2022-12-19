import numpy as np
import matplotlib.pyplot as plt
    
def check_masscons(ng:int,Nx:int,Ny:int,c:float,ii:int,dt:float,t:float,T:float,c_plot:float): 
    #check mass conservation
    
    llx = ng + 1
    ulx = ng + Nx
    
    lly = ng + 1
    uly = ng + Ny
    
    temp_cplot = np.array([np.sum(c[llx-1:ulx,lly-1:uly])])
    if ii == 1:
        c_plot = np.copy(temp_cplot)
    else:
        c_plot = np.vstack([c_plot, temp_cplot])
    
    if np.round(temp_cplot) != np.round(c_plot[0]):
        print('Mass is not conserved!')
    
    if (t >= (T - dt)):
        print('Plotting coming')          
        fig, main_ax = plt.subplots()
        fig.set_size_inches(8, 8)
        main_ax.set_xlabel('time step')
        main_ax.set_ylabel('total scalar mass within domain')
        main_ax.set_title('conservation of mass')
        main_ax.plot(range(len(c_plot)), c_plot)
      
    return c_plot