import numpy as np
import random
import tensorflow as tf

from tensorflow.keras.layers import Input, Dense, Activation, Dropout, Concatenate, add, multiply
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras import utils as np_utils
 
tf.compat.v1.disable_eager_execution()



class DDPG():

    def __init__(self, env):
        
        # hyperparameters
        self.env = env
        self.epochs = 1
        self.gamma = 0.99
        self.actor_learning_rate = 0.0007
        self.critic_learning_rate = 0.0002
        self.n_s = env.observation_space.shape[0]
        self.n_a = env.action_space.shape[0]
        self.description = 'Deep Deterministic Policy Gradient'
        self.verbose = False
        self.tau = 0.001
        self.nn_batch_size = None
        self.minibatch_sz = 64
        self.action_low = env.action_space.low
        self.action_high = env.action_space.high
        self.sigma = self.action_high - self.action_low
        self.sigma_min = (self.action_high - self.action_low) / 1000
        self.sigma_decay = .9999

        print('action high', self.action_high)
        print('action low', self.action_low)

        # memory
        self.memory_max = 50000

        self.reset()

    def create_actor_model(self):
        
        inputs = Input(shape=self.n_s)
        x = Dense(32, activation='relu')(inputs)
        x = Dense(32, activation='relu')(x)
        x = Dense(1, activation='sigmoid')(x)
        x = multiply([x, self.action_high-self.action_low])
        x = add([x, self.action_low])

        model = Model(inputs=inputs, outputs=x)

        return model
    
    def create_critic_model(self):

        critic = Sequential()
        critic.add(Dense(32, input_dim=self.n_s+self.n_a, activation='relu'))
        critic.add(Dense(32, activation='relu'))
        critic.add(Dense(1, activation='linear'))
        critic.compile(loss='mse', optimizer=Adam(lr=self.critic_learning_rate))
        return critic

    def reset(self):
        
        self.sigma = self.action_high - self.action_low
        self.step = 0
        self.memory = [] 
         
        # Initialize actor model
        self.model = self.create_actor_model()

        # Initialize actor target model
        self.model_target = self.create_actor_model()
        self.model_target.set_weights(self.model.get_weights())

        # Initialize critic model
        self.critic = self.create_critic_model()

        # Initialize critic target model
        self.critic_target = self.create_critic_model()
        self.critic_target.set_weights(self.critic.get_weights())

        # Create train function for actor
        self._build_actor_train_fn()


    def _update_target_models_(self):

        # Update actor target
        new_weights = []
        for m_weights, t_weights in zip(self.model.get_weights(), self.model_target.get_weights()):
            new_weights.append(self.tau * m_weights + (1-self.tau) * t_weights)
        self.model_target.set_weights(new_weights)

        # Update critic target
        new_weights = []
        for m_weights, t_weights in zip(self.critic.get_weights(), self.critic_target.get_weights()):
            new_weights.append(self.tau * m_weights + (1-self.tau) * t_weights)
        self.critic_target.set_weights(new_weights)

    def pick_action(self, state):

        action = self.model.predict(np.array([state,]))[0]
        
        if self.verbose:
            print('model prediction', action)
        
        action = np.clip(action + np.random.normal(0, self.sigma), self.env.action_space.low, self.env.action_space.high)
        self.sigma = max(self.sigma * self.sigma_decay, self.sigma_min)

        if self.verbose:
            print(f'Action taken in state {state}: {action}')

        return action


    def _build_actor_train_fn(self):

        actions = self.model.output
        state_actions = Concatenate(axis=1)([self.model.input, actions])
        q_values = self.critic(state_actions)

        loss = -K.mean(q_values)

        adam = Adam(lr=self.actor_learning_rate)

        updates = adam.get_updates(params=self.model.trainable_weights,
                                   loss=loss)

        self.actor_train_fn = K.function(inputs=[self.model.input],
                                   outputs=[loss],
                                   updates=updates)


    def _batch_train_(self, batch):

        # get predictions
        s_vec = np.array([x[0] for x in batch])
        a_vec = np.array([x[1] for x in batch])
        s_a_vec = np.array([x[2] for x in batch])
        sp_vec = np.array([x[4] for x in batch])

        ap_vec = self.model_target.predict(sp_vec)
        q_sp_ap = self.critic_target.predict(np.concatenate([sp_vec, ap_vec], axis=1))

        q_s_a = self.critic.predict(s_a_vec)

        # train critic
        for i in range(len(batch)): 
            s, a, sa, r, s_prime, done = batch[i]
            target = r
            if not done:
                target += self.gamma * q_sp_ap[i]
            q_s_a[i] = target

        self.critic.fit(s_a_vec, q_s_a, epochs = self.epochs, 
                                        batch_size = self.nn_batch_size,
                                        verbose = False)

        # train actor
        loss = self.actor_train_fn([s_vec])

    def update(self, s, a, r, s_prime, done):

        # Store in memory
        if len(self.memory) > self.memory_max: self.memory.pop(0)

        sa = np.concatenate([s, a])
        self.memory.append([s, a, sa, r, s_prime, done])


        if len(self.memory) > self.minibatch_sz:

            # create training batch
            batch = random.sample(self.memory, self.minibatch_sz)
            
            # update critic
            self._batch_train_(batch)
        
        self._update_target_models_()