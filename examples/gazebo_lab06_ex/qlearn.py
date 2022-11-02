import random
import pickle
import numpy as np


class QLearn:
    def __init__(self, actions, epsilon, alpha, gamma):
        self.q = {}
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha      # discount constant
        self.gamma = gamma      # discount factor
        self.actions = actions

    def loadQ(self, filename):
        '''
        Load the Q state-action values from a pickle file.
        '''
        with open(filename) as f:
            self.q = pickle.load(f)
            print("Loaded file: {}".format(filename+".pickle"))

    def saveQ(self, filename):
        '''
        Save the Q state-action values in a pickle file.
        '''
        with open(filename, "wb") as f:
            pickle.dump(self.q, f)
            print("Wrote to file: {}".format(filename+".pickle"))

    def getQ(self, state, action):
        '''
        @brief returns the state, action Q value or 0.0 if the value is 
            missing
        '''
        return self.q.get((state, action), 0.0)

    def chooseAction(self, state, return_q=False):
        '''
        @brief returns a random action epsilon % of the time or the action 
            associated with the largest Q value in (1-epsilon)% of the time
        '''
        # TODO: Implement exploration vs exploitation
        #    if we need to take a random action:
        #       * return a random action
        #    else:
        #       * determine which action has the highest Q value for the state 
        #          we are in.
        #       * address edge cases - what if 2 actions have the same max Q 
        #          value?
        #       * return the action with highest Q value
        #
        # NOTE: if return_q is set to True return (action, q) instead of
        #       just action

        if random.random() < self.epsilon:
            action = random.randint(0,2)
        else:
            qs = [(self.getQ(state, a),a) for a in self.actions]
            if qs.count(max(qs[0])) > 1:
                options = [q[1] for q in qs if q[0] == max(qs[0])]
                action = random.choice(options)
            else:
                action = qs[0].index(max(qs[0]))
        # THE NEXT LINE NEEDS TO BE MODIFIED TO MATCH THE REQUIREMENTS ABOVE
        if return_q == True:
            return self.actions[action], self.getQ(state,action)
        return self.actions[action]

    def learn(self, state1, action1, reward, state2):
        '''
        @brief updates the Q(state,value) dictionary using the bellman update
            equation
        '''
        # TODO: Implement the Bellman update function:
        #     Q(s1, a1) += alpha * [reward(s1,a1) + gamma* max(Q(s2)) - Q(s1,a1)]
        # 
        # NOTE: address edge cases: i.e. 
        # 
        # Find Q for current (state1, action1)
        # Address edge cases what do we want to do if the [state, action]
        #       is not in our dictionary?
        # Find max(Q) for state2
        # Update Q for (state1, action1) (use discount factor gamma for future 
        #   rewards)
        maxQ = max([self.getQ(state2, a) for a in self.actions])
        q = self.getQ(state1, action1)
        if q == 0.0:
            self.q[(state1,action1)] = reward
        else:
            self.q[(state1,action1)] = q + self.alpha * ((reward + self.gamma * maxQ) - q)
