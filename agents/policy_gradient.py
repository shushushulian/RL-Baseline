import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Input, Dense, Activation, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras import utils as np_utils
 
tf.compat.v1.disable_eager_execution()

class PolicyGradient():

    def __init__(self, env):
        
        # hyperparameters
        self.env = env
        self.gamma = 0.99
        self.nn_learning_rate = 0.001
        self.n_s = env.observation_space.shape[0]
        self.n_a = env.action_space.n
        self.description = 'Vanilla Policy Gradient'
        self.verbose = False

        self.reset()


    def reset(self):
            
        model = Sequential()
        model.add(Dense(64, input_dim=self.n_s, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.n_a, activation = 'softmax'))

        self.model = model
        self._build_train_fn()
        self.reset_experience()

    def reset_experience(self):

        self.experience_cache = {
            's': [],
            'a': [],
            'r': [],
        }

    def pick_action(self, state):

        tmp = self.model.predict(np.array([state,]))
        action = np.random.choice(np.arange(self.n_a), p=tmp[0])
        
        if self.verbose: 
            print(f'prediction confidence {tmp}')

        return action


    def _build_train_fn(self):

        """
        Adapted from https://gist.github.com/kkweon/c8d1caabaf7b43317bc8825c226045d2
        """

        action_prob_placeholder = self.model.output
        action_onehot_placeholder = K.placeholder(shape=(None, self.n_a), name="action_onehot")
        discount_reward_placeholder = K.placeholder(shape=(None,), name="disc_reward")

        action_prob = K.sum(action_prob_placeholder * action_onehot_placeholder, axis=1)
        log_action_prob = K.log(action_prob)

        loss = - log_action_prob * discount_reward_placeholder
        loss = K.mean(loss)

        adam = Adam(lr=self.nn_learning_rate)

        updates = adam.get_updates(params=self.model.trainable_weights,
                                   loss=loss)

        self.train_fn = K.function(inputs=[self.model.input,
                                           action_onehot_placeholder,
                                           discount_reward_placeholder],
                                   outputs=[loss],
                                   updates=updates)

    def update(self, s, a, r, s_prime, done):

        self.experience_cache['s'].append(s)
        self.experience_cache['a'].append(a)
        self.experience_cache['r'].append(r)
        
        if done:

            n = len(self.experience_cache['s'])
            states = np.array(self.experience_cache['s'])
            actions = np.array(self.experience_cache['a'])
            rewards = np.array(self.experience_cache['r'])


            # Calculated discounted rewards
            disc_reward = rewards.copy()
            for i in range(n-2, -1, -1):
                disc_reward[i] = disc_reward[i+1]*self.gamma + rewards[i]

            # print('unnormalized disc reward', disc_reward)

            # Normalize discounted rewards
            disc_reward = (disc_reward - disc_reward.mean()) / (disc_reward.std() + 1e-9)

            # One-hot encode the actions
            action_onehot_encoded =  np_utils.to_categorical(actions, num_classes=self.n_a)

            # Train
            loss = self.train_fn([
                states,
                action_onehot_encoded,
                disc_reward
            ])

            self.reset_experience()

            