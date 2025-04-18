import numpy as np
import scipy

from bandit_machines import BernoulliMAB, GaussianMAB

def randmax(a):
    """ return a random maximum """
    a = np.array(a)  
    max_indices = np.flatnonzero(a == a.max())
    return np.random.choice(max_indices)

def evaluate_run(env, pi, T=1000):
    '''
    Run a bandit agent on a bandit instance (environment) for T steps.
    '''
    r_log = []
    a_log = []
    pi.clear()
    for t in range(T):
        a = pi.act()
        r = env.rwd(a)
        pi.update(a,r)
        r_log.append(r)
        a_log.append(a)
    return a_log, r_log

def evaluate_runs(env, pi, T=1000, N=10):
    '''
        Parameters
        ----------

        env : 
            bandit machine
        pi : 
            bandit algorithm
        N : int
            number of experiments
        T : int
            number of trails per experiment
    '''
    R_log = np.zeros((N,T))
    A_log = np.zeros((N,T),dtype=int)
    for n in range(N):
        np.random.seed()
        A_log[n], R_log[n] = evaluate_run(env, pi, T)

    return A_log, R_log

class UCB():

    """UCB1 with parameter alpha"""

    def __init__(self,n_arms,alpha=1/2):
        self.n_arms = n_arms
        self.alpha = alpha
        self.clear()
        
    # TODO 
    def clear(self):
        # 重置参数 
        self.counts = np.zeros(self.n_arms)  # 每个臂的拉取次数
        self.values = np.zeros(self.n_arms)  # 每个臂的平均奖励
        self.total_pulls = 0  # 记录总共的拉取次数

    def act(self):
        # 选择要拉的臂 (UCB1) 
        self.total_pulls += 1  # 总拉取次数 +1

        # 先尝试所有臂至少一次（确保分母不会是 0）
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                return arm

        # 计算 UCB 公式
        confidence_bounds = self.values + self.alpha * np.sqrt(np.log(self.total_pulls) / self.counts)
        return np.argmax(confidence_bounds)  # 选择 UCB 最高的臂

    def update(self, chosen_arm, reward):
        # 根据奖励更新估计值 
        self.counts[chosen_arm] += 1  # 该臂被选择的次数 +1
        n = self.counts[chosen_arm]
        self.values[chosen_arm] = (self.values[chosen_arm] * (n - 1) + reward) / n  # 增量更新

    def name(self):
        return "UCB(%3.2f)" % self.alpha

