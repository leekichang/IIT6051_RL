import numpy as np

class RandomWalk:
    def __init__(self, N=5, n_step=1, GAMMA=1.0, is_Q1=True):
        self.N = N
        self.is_Q1 = is_Q1
        self.init_values = np.zeros(N + 2)
        if is_Q1:
            # use this True value for Q1
            self.init_values[1:N+1]    = 0.5
            self.TRUE_VALUE        = np.zeros(N + 2)
            self.TRUE_VALUE[1:N+1] = np.arange(1, N+1)/(N+1)
            self.TRUE_VALUE[N+1]   = 1
        else:
            # use this True value for Q2
            self.TRUE_VALUE         = np.arange(-1*(N+1), N+3, 2) / (N+1)
            self.TRUE_VALUE[0] = self.TRUE_VALUE[-1] = 0

        self.actions           = {"LEFT":0, "RIGHT":1}
        self.term_state        = [0, N+1]
        self.GAMMA             = GAMMA
        self.n_step            = n_step

    def random_walk(self):
        return -1 if np.random.binomial(1, 0.5) == self.actions["LEFT"] else 1

    def TD(self, value, n=1, alpha=0.0, init_state=None):
        state = init_state if init_state!=None else (self.N+2)//2 
        states, rewards = [state], [0]
        
        time = 0                      
        T = np.inf                
        while True:
            time += 1
            if time < T:          
                next_state = state + self.random_walk() 
    
                if self.is_Q1:
                    
                    # use this reward for Q1
                    reward = 1 if next_state == self.N+1 else 0
                else:                
                    
                    # Use this reward for Q2
                    if next_state == self.N+1:
                        reward = 1
                    elif next_state == 0:
                        reward = -1
                    else:
                        reward = 0
                
                states.append(next_state)               
                rewards.append(reward)                  

                if next_state in self.term_state:       
                    T = time
            update_time = time - n                      
            if time >= n:
                returns = 0.0                           
                for t in range(update_time + 1, \
                               min(T, update_time + n) + 1):
                    returns += pow(self.GAMMA, \
                                   t - update_time - 1) * rewards[t]
                if time <= T:                                                
                    returns += pow(self.GAMMA, n)\
                          * value[states[(time)]]
                update_state = states[update_time]
                
                if not update_state in self.term_state:
                    value[update_state] += alpha * \
                        (returns - value[update_state])
            if update_time == T - 1:
                break
            state = next_state

    def MonteCarlo(self, values, init_state=3, alpha=0.1):
        state = init_state if init_state != None else (self.N+2)//2
        trace = [init_state]
        while True:
            state += self.random_walk()
            trace.append(state)

            returns = 1.0 if state == self.N+1 else 0.0
            if state in self.term_state:
                break

        for state_ in trace[:-1]:
            values[state_] += alpha * (returns - values[state_])