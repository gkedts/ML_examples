class Distribution(dict):
    """
    The Distribution class extend the Python dictionary such that
    each key's value should correspond to the probability of the key.

    For example, here's how you can create a random variable X that takes on
    value 'spam' with probability .7 and 'eggs' with probability .3:

    X = Distribution()
    X['spam'] = .7
    X['eggs'] = .3

    Methods
    -------
    renormalize():
      scales all the probabilities so that they sum to 1
    get_mode():
      returns an item with the highest probability, breaking ties arbitrarily
    sample():
      draws a sample from the Distribution
    """
    def __missing__(self, key):
        # if the key is missing, return probability 0
        return 0

    def renormalize(self):
        normalization_constant = float(sum(self.itervalues()))
        for key in self.iterkeys():
            self[key] /= normalization_constant

    def get_mode(self):
        maximum = -1
        arg_max = None

        for key in self.iterkeys():
            if self[key] > maximum:
                arg_max = key
                maximum = self[key]

        return arg_max

def forward_backward(all_possible_hidden_states,
                     all_possible_observations,
                     prior_distribution,
                     transition_model,
                     observation_model,
                     observations):
    """
    Inputs
    ------
    all_possible_hidden_states: a list of possible hidden states

    all_possible_observations: a list of possible observed states

    prior_distribution: a distribution over states

    transition_model: a function that takes a hidden state and returns a
        Distribution for the next state

    observation_model: a function that takes a hidden state and returns a
        Distribution for the observation from that hidden state

    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    A list of marginal distributions at each time step; each distribution
    should be encoded as a Distribution (see the Distribution class in
    robot.py and see how it is used in both robot.py and the function
    generate_data() above, and the i-th Distribution should correspond to time
    step i
    """

    num_possible_hidden_states = len(prior_distribution)
    num_time_steps = len(observations)

    # phi[i][state] is the probability for each state of seeing the observation at step i
    phi = [None] * num_time_steps
    for i in range(num_time_steps):
        if observations[i] is not None:
            phi[i] = Distribution()
            for state in all_possible_hidden_states:
                phi[i][state] = observation_model(state)[observations[i]]
        else:
            phi[i] = Distribution()
            for state in all_possible_hidden_states:
                phi[i][state] = 1

    # alpha[i][state] is the probability for each state at step i, given past observations
    alpha = [None] * num_time_steps
    for i in range(num_time_steps):
        if i == 0:
            alpha[i] = prior_distribution
        else:
            alpha[i] = Distribution()
            for state in all_possible_hidden_states:
                trans = transition_model(state)
                for new_state in all_possible_hidden_states:
                    alpha[i][new_state] += alpha[i-1][state] * phi[i-1][state] * trans[new_state]
        alpha[i].renormalize()

    # beta[i][state] is the probability for each state at step i, given future observations
    beta = [None] * num_time_steps
    for i in reversed(range(num_time_steps)):
        if i == num_time_steps - 1:
            beta[i] = Distribution()
            for state in all_possible_hidden_states:
                beta[i][state] = 1
        else:
            beta[i] = Distribution()
            for prev_state in all_possible_hidden_states:
                trans = transition_model(prev_state)
                for state in all_possible_hidden_states:
                    beta[i][prev_state] += trans[state] * phi[i+1][state] * beta[i+1][state]
        beta[i].renormalize()

    # marginals[i][state] is the probability for each state at step i, given all observations
    marginals = [None] * num_time_steps
    for i in range(num_time_steps):
        marginals[i] = Distribution()
        for state in all_possible_hidden_states:
            marginals[i][state] = alpha[i][state] * beta[i][state] * phi[i][state]
        marginals[i].renormalize()

    return marginals

def viterbi(all_possible_hidden_states,
            all_possible_observations,
            prior_distribution,
            transition_model,
            observation_model,
            observations):
    """
    Inputs
    ------
    See the list inputs for the function forward_backward() above.

    Output
    ------
    A list of esimated hidden states
    """

    num_possible_hidden_states = len(prior_distribution)
    num_time_steps = len(observations)

    # phi[i][state] is the probability for each state of seeing the observation at step i
    phi = [None] * num_time_steps
    for i in range(num_time_steps):
        if observations[i] is not None:
            phi[i] = Distribution()
            for state in all_possible_hidden_states:
                phi[i][state] = observation_model(state)[observations[i]]
        else:
            phi[i] = Distribution()
            for state in all_possible_hidden_states:
                phi[i][state] = 1

    # message[i][state] is the probability of the most likely state sequence
    # traceback[i][state] is the previous state in the sequence
    traceback = [None] * num_time_steps
    message = [None] * num_time_steps
    for i in range(0, num_time_steps):
        if i == 0:
            message[i] = prior_distribution
        else:
            message[i] = Distribution()
            traceback[i] = {}
            for state in all_possible_hidden_states:
                trans = transition_model(state)
                for new_state in all_possible_hidden_states:
                    p = message[i-1][state] * phi[i-1][state] * trans[new_state]
                    if p > message[i][new_state]:
                        message[i][new_state] = p
                        traceback[i][new_state] = state
        message[i].renormalize()

    estimated_hidden_states = [None] * num_time_steps
    for i in reversed(range(num_time_steps)):
        if i == num_time_steps - 1:
            final_message = Distribution()
            for state in all_possible_hidden_states:
                final_message[state] = message[i][state] * phi[i][state]
            estimated_hidden_states[i] = final_message.get_mode()
        else:
            estimated_hidden_states[i] = traceback[i+1][estimated_hidden_states[i+1]]

    return estimated_hidden_states

## example

all_possible_hidden_states = ['healthy', 'fever']
all_possible_observations = ['normal', 'cold', 'dizzy']

initial_distribution = Distribution({'healthy': 0.6, 'fever': 0.4})

def transition_model(state):
    if state == 'healthy':
        return Distribution({'healthy': 0.7, 'fever': 0.3})
    elif state == 'fever':
        return Distribution({'healthy': 0.4, 'fever': 0.6})

def observation_model(state):
    if state == 'healthy':
        return Distribution({'normal': 0.5, 'cold': 0.4, 'dizzy': 0.1})
    elif state == 'fever':
        return Distribution({'normal': 0.1, 'cold': 0.3, 'dizzy': 0.6})

observations = ['normal', 'cold', 'dizzy', 'dizzy', 'cold']

print forward_backward(all_possible_hidden_states, all_possible_observations,
                       initial_distribution, transition_model, observation_model,
                       observations)

print viterbi(all_possible_hidden_states, all_possible_observations,
              initial_distribution, transition_model, observation_model,
              observations)

