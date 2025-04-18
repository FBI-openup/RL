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

class Agent: 

    def __init__(self, env): 
        '''
            env : Environment 
                of the type provided to you
        '''
        self.env = env

    # TODO (optional): Add any auxilliary functions you might use here 


    def P_traj(self, ooo, M=-1):
        '''
        Provides full conditional distribution P(SSS | ooo) where SSS and ooo are sequences of length T.
        $$
            P( Y_1,\\ldots,Y_T | o_1,\\ldots,o_T )
        $$


        Parameters
        ----------

        ooo : array_like(t, d)
            t observations (of d dimensions each)

        M : int
            -1 indicates to use a brute force solution (exact recovery of the distribution) 
            M > 0 indicates to use M Monte Carlo simulations (this parameter is used in Week 2)


        Returns
        -------

        p : dict(str:float)
            such that p[sss] = P(sss | ooo)
            and if sss not in p, it implies P(sss | ooo) = 0

            important: let sss be a string representation of the state sequence, separated by spaces, e.g., 
            the string representation of np.array([1,2,3,4],dtype=int) should be '1 2 3 4'. 
        '''        
        p = {}
        # TODO 
        T = len(ooo)  # 观测序列长度
        # partial[t] 将存储所有长度为 t 的可能状态序列的 “联合概率” p(s_1,...,s_t, o_1,...,o_t)
        partial = [dict() for _ in range(T+1)]
        # 初始空路径概率 = 1
        partial[0][()] = 1.0

        # 逐步扩展到长度 T
        for t in range(T):
            # 当前要纳入第 t+1 个观测 ooo[t]
            obs_t = ooo[t]  # 例如 [0,1] 或 [1,0] 等
            for path_so_far, p_val in partial[t].items():
                if t == 0:
                    # 这是要采样 s_1
                    for s1 in range(self.env.n_states):
                        # p(s1) = env.P_1[s1]
                        p_s1 = self.env.P_1[s1]
                        # p(o_1 | s1)
                        p_o1 = 1.0
                        for j in range(self.env.d_obs):
                            p_o1 *= self.env.P_O[s1, j, int(obs_t[j])]
                        p_new = p_val * p_s1 * p_o1
                        if p_new > 0:
                            partial[t+1][(s1,)] = partial[t+1].get((s1,), 0.0) + p_new
                else:
                    # 已有 path_so_far = (s_1, ..., s_t)
                    s_prev = path_so_far[-1]
                    for s_new in range(self.env.n_states):
                        p_trans = self.env.P_S[s_prev, s_new]  # p(s_{t+1} | s_t)
                        # p(o_{t+1} | s_{t+1})
                        p_obs = 1.0
                        for j in range(self.env.d_obs):
                            p_obs *= self.env.P_O[s_new, j, int(obs_t[j])]
                        p_new = p_val * p_trans * p_obs
                        if p_new > 0:
                            new_path = path_so_far + (s_new,)
                            partial[t+1][new_path] = partial[t+1].get(new_path, 0.0) + p_new

        # 此时 partial[T] 里存放了所有长为T的路径 (s_1, ..., s_T) 对应的 p( s_1,...,s_T, o_1,...,o_T ).
        # 我们要得到条件概率 p(...|oo) = 上式除以对所有路径的总和。
        denom = sum(partial[T].values())
        p = {}
        for path, val in partial[T].items():
            path_str = ' '.join(str(x) for x in path)
            p[path_str] = val / denom
        return p


        
    def P_S(self, o, t=-1, M=-1): 
        '''
        Provide P(s_t | o) given observations o from 1,...,T.  

        $$
            P(S_t | o_1,...,o_T ).
        $$
        
        The probability (distribution) of the t-th state, given the observed evidence 'o'.

        Parameters
        ----------

        o : array_like(t,d)
            up to t observations (of d dimensions each)

        t : int
            the state being queried, e.g., 3, or -1 for final state (corresponding to o[-1])

        Returns
        -------

        P : array_like(float,ndim=1) 
            such that P[s] = P(S_t = s | o_1,...,o_t)
        '''
        # TODO 
        T = len(o)
        # 如果 t = -1，就默认看最终时刻 t = T
        if t == -1:
            t = T

        # 先计算完整轨迹的后验分布
        p_traj = self.P_traj(o)  # dict: path_str -> prob

        # 求和所有满足 s_t = s 的路径概率
        P = np.zeros(self.env.n_states)
        for path_str, prob_val in p_traj.items():
            path = [int(x) for x in path_str.split()]  # 转成真正的序列
            # path[t-1] 就是 s_t （因为Python下标从0开始）
            s_t = path[t-1]
            P[s_t] += prob_val
        
        return P

    def Q(self, o): 
        '''
            Provide Q(o,a) for all a i.e., the value for any given a under observation o. 

            Parameters
            ----------

            o : array_like(int,ndim=2)
                t observations (of 2 bits each)

            Returns
            -------

            Q : array_like(float,ndim=n_actions)
                such that Q[a] is the value (expected reward) of action a.

        '''
        Q = np.zeros(self.env.n_states)
        # TODO 
        Q= self.P_S(o, -1)
        return Q

    def act(self, obs): 
        '''
        Decide on the best action to take, under the provided observation. 

        Parameters
        ----------

        obs : array_like(int,ndim=2)
            t observations (of 2 bits each)

        Returns
        -------

        a : int
            the chosen action a
        '''

        a = -1
        # TODO 
        Q_values = self.Q(obs)
        a = np.argmax(Q_values)
        return a

