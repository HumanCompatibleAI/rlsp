import numpy as np
from copy import deepcopy
from itertools import product

from envs.env import DeterministicEnv, Direction


class RoomState(object):
    '''
    state of the environment; describes positions of all objects in the env.
    '''
    def __init__(self, agent_pos, vase_states):
        """
        agent_pos: (x, y) tuple for the agent's location
        vase_states: Dictionary mapping (x, y) tuples to booleans, where True
            means that the vase is intact
        """
        self.agent_pos = agent_pos
        self.vase_states = vase_states

    def __eq__(self, other):
        return isinstance(other, RoomState) and \
            self.agent_pos == other.agent_pos and \
            self.vase_states == other.vase_states

    def __hash__(self):
        def get_vals(dictionary):
            return tuple([dictionary[loc] for loc in sorted(dictionary.keys())])
        return hash(self.agent_pos + get_vals(self.vase_states))


class RoomEnv(DeterministicEnv):
    def __init__(self, spec, compute_transitions=True):
        """
        height: Integer, height of the grid. Y coordinates are in [0, height).
        width: Integer, width of the grid. X coordinates are in [0, width).
        init_state: RoomState, initial state of the environment
        vase_locations: List of (x, y) tuples, locations of vases
        num_vases: Integer, number of vases
        carpet_locations: Set of (x, y) tuples, locations of carpets
        feature_locations: List of (x, y) tuples, locations of features
        s: RoomState, Current state
        nA: Integer, number of actions
        """
        self.height = spec.height
        self.width = spec.width
        self.init_state = deepcopy(spec.init_state)
        self.vase_locations = list(self.init_state.vase_states.keys())
        self.num_vases = len(self.vase_locations)
        self.carpet_locations = set(spec.carpet_locations)
        self.feature_locations = list(spec.feature_locations)

        self.default_action = Direction.get_number_from_direction(Direction.STAY)
        self.nA = 5
        self.num_features = len(self.s_to_f(self.init_state))

        self.reset()

        if compute_transitions:
            states = self.enumerate_states()
            self.make_transition_matrices(
                states, range(self.nA), self.nS, self.nA)
            self.make_f_matrix(self.nS, self.num_features)


    def enumerate_states(self):
        state_num = {}

        # Possible vase states
        for vase_intact_bools in product([True, False], repeat=self.num_vases):
            vase_states = dict(zip(self.vase_locations, vase_intact_bools))
            # Possible agent positions
            for y in range(self.height):
                for x in range(self.width):
                    pos = (x, y)
                    if pos in vase_states and vase_states[pos]:
                        # Can't have the agent on an intact vase
                        continue
                    state = RoomState(pos, vase_states)
                    if state not in state_num:
                        state_num[state] = len(state_num)

        self.state_num = state_num
        self.num_state = {v: k for k, v in self.state_num.items()}
        self.nS = len(state_num)

        return state_num.keys()

    def get_num_from_state(self, state):
        return self.state_num[state]

    def get_state_from_num(self, num):
        return self.num_state[num]


    def s_to_f(self, s):
        '''
        Returns features of the state:
        - Number of broken vases
        - Whether the agent is on a carpet
        - For each feature location, whether the agent is on that location
        '''
        num_broken_vases = list(s.vase_states.values()).count(False)
        carpet_feature = int(s.agent_pos in self.carpet_locations)
        features = [int(s.agent_pos == fpos) for fpos in self.feature_locations]
        features = [num_broken_vases, carpet_feature] + features
        return np.array(features)


    def get_next_state(self, state, action):
        '''returns the next state given a state and an action'''
        action = int(action)
        new_x, new_y = Direction.move_in_direction_number(state.agent_pos, action)
        # New position is still in bounds:
        if not (0 <= new_x < self.width and 0 <= new_y < self.height):
            new_x, new_y = state.agent_pos
        new_agent_pos = new_x, new_y
        new_vase_states = deepcopy(state.vase_states)
        if new_agent_pos in new_vase_states:
            new_vase_states[new_agent_pos] = False  # Break the vase
        return RoomState(new_agent_pos, new_vase_states)


    def print_state(self, state):
        '''Renders the state.'''
        h, w = self.height, self.width
        canvas = np.zeros(tuple([2*h-1, 3*w+1]), dtype='int8')

        # cell borders
        for y in range(1, canvas.shape[0], 2):
            canvas[y, :] = 1
        for x in range(0, canvas.shape[1], 3):
            canvas[:, x] = 2

        # vases
        for x, y in self.vase_locations:
            if state.vase_states[(x, y)]:
                canvas[2*y, 3*x+1] = 4
            else:
                canvas[2*y, 3*x+1] = 6

        # agent
        x, y = state.agent_pos
        canvas[2*y, 3*x + 2] = 3

        black_color = '\x1b[0m'
        purple_background_color = '\x1b[0;35;85m'

        for line in canvas:
            for char_num in line:
                if char_num==0:
                    print('\u2003', end='')
                elif char_num==1:
                    print('─', end='')
                elif char_num==2:
                    print('│', end='')
                elif char_num==3:
                    print('\x1b[0;33;85m█'+black_color, end='')
                elif char_num==4:
                    print('\x1b[0;32;85m█'+black_color , end='')
                elif char_num==5:
                    print(purple_background_color+'█'+black_color, end='')
                elif char_num==6:
                    print('\033[91m█'+black_color, end='')
            print('')
