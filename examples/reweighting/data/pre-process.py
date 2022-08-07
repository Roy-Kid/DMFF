
import numpy as np
import mdtraj as md

def extract_log_data(temperature):
    
    filename = f'{temperature}K_long.log'
    
    print(f'extract {filename}')
    
    with open(filename) as f:
        
        lines = f.readlines()
        
        energy = []
        density = []
        
        for line in lines[6:]:
            data = line.split()
            energy.append(float(data[2]))
            density.append(float(data[-1]))
            
    np.save(f'{temperature}K_long_energy', np.array(energy))
    np.save(f'{temperature}K_long_density', np.array(density))
        
def extract_traj_data(temperature):
    
    filename = f'{temperature}K_long.nc'
    
    t = md.load_netcdf(filename, top='waterbox.pdb')
    traj = np.array(t.xyz)  # (n_frames, n_atoms, 3)
    box = np.array(t.unitcell_vectors) # (n_frames, 3)
    
    np.save(f'{temperature}K_long_traj', traj)
    np.save(f'{temperature}K_long_box', box)
    
if __name__ == '__main__':
    
    for T in [300, 305, 310]:
        extract_log_data(T)
        extract_traj_data(T)
   
    