# The problem
## The basic setting
 - Reinforcement learning setting, one time step
    1. Agent __observes__ environement state: _S0_
    2. Agent __acts__ on the environment: _A0_
    3. Environement __rewarda__ agent: _R1_ & _S1_

 Sequence of _states, actions and rewards_ : S0, A0, R1, S1, A1, R2, S2...
 
 __Goals__: maximize expected cumulative reward 


 ### Episodic vs continuing task:
 - Episodic: the interactio ends at some time step T
 - Continuing: the go on forever, i.e. trading algorithm...


 
    A task is an instance of the reinforcement learning (RL) problem.
    Continuing tasks are tasks that continue forever, without end.
    Episodic tasks are tasks with a well-defined starting and ending point.
        In this case, we refer to a complete sequence of interaction, from start to finish, as an episode.
        Episodic tasks come to an end whenever the agent reaches a terminal state.
