import numpy as np

def gen_traj(env, T=5):
    ''' Generate a path with associated observations.


        Paramaters
        ----------

        T : int
            how long is the path

        Returns
        -------

        o : (T,d)-shape array
            sequence of observations
        s : T-length array of states
            sequence of tiles
    '''
    # TODO 
        # 1. init the return arrays
    s = np.zeros(T, dtype=int)              # state sequence
    o = np.zeros((T, env.d_obs), dtype=int) #observe sequence

    # 2. init the environment usig None as the initial state
    env._s= None
    s[0], o[0] = env.step(None)
    # 3. loop over the T timesteps
    for t in range(1, T):
        # 4. at each timestep, sample an action and observe the next state and observation
        s[t] ,o[t]=  env.step(s[t-1]) 
    
    return o, s

