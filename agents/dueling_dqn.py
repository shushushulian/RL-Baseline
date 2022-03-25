import numpy as np
import keras
import random
import tensorflow as tf

from keras.layers import Input, Dense, Activation, Dropout, Add, Lambda
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import backend as K

class DuelingDQN():
    def __init__(self, env):
        
        # hyperparameters
        self.env = env
        self.gamma = 0.99
        self.nn_learning_rate = 0.0001
        self.nn_batch_size = None
        self.epochs = 1
        self.minibatch_sz = 64
        self.epsilon = 1.
        self.epsilon_decay = 0.992
        self.epsilon_floor = 0.05
        self.n_s = env.observation_space.shape[0]
        self.n_a = env.action_space.n
        self.description = 'Dueling DQN Learner'
        self.update_frequency = 100
        self.verbose = False

        # memory
        self.memory_max = 50000
        self.reset()

    def reset(self):
        self.epsilon = 1.
        self.step = 0
        self.memory = [] 

        # create nn's
        self.model = self._make_model_()
        self.target_model = self._make_model_()

    def _make_model_(self):

        inputs = Input(shape=(self.n_s,))

        # Common layers
        common = Dense(64)(inputs)
        common = Activation("relu")(common)

        # Value layers
        value = Dense(32)(common)
        value = Activation("relu")(value)
        value = Dense(1)(value)
        value = Activation("relu")(value)

        # Advantage layers
        adv = Dense(32)(common)
        adv = Activation("relu")(adv)
        adv = Dense(self.n_a)(adv)
        adv = Activation("relu")(adv)
        
        neg_mean_adv = Lambda(lambda x: -K.mean(x, axis=1))(adv)

        # Combining them
        comb = Add()([value, adv])
        comb = Add()([comb, neg_mean_adv])

        model = Model(inputs=inputs, outputs=comb)

        model.compile(loss='mse', optimizer=Adam(lr=self.nn_learning_rate))

        return model

    def _model_update_(self):
        self.target_model.set_weights(self.model.get_weights())

    def pick_action(self, state):

        if np.random.random() < self.epsilon:
            return np.random.choice(self.n_a)
        else:
            tmp = self.model.predict(np.array([state,]))
            return np.argmax(tmp[0])


    def update(self, s, a, r, s_prime, done):

        if len(self.memory) > self.memory_max: self.memory.pop(0)
        self.memory.append([s, a, r, s_prime, done])

        self._batch_train_()

        if self.step % self.update_frequency == 0:
            self._model_update_()

        if done and self.epsilon > self.epsilon_floor:
            self.epsilon = self.epsilon * self.epsilon_decay
        self.step += 1

    def _batch_train_(self):
        if len(self.memory) > self.minibatch_sz:

            # create training batch
            batch = random.sample(self.memory, self.minibatch_sz)

            # get predictions
            s_vec = np.array([x[0] for x in batch])
            sp_vec = np.array([x[3] for x in batch])

            m_pred = self.model.predict(s_vec)
            tm_pred = self.target_model.predict(sp_vec)

            # use update rule from Minh 2013
            for i in range(len(batch)): 
                s, a, r, s_prime, done = batch[i]
                target = r
                if not done:

                    target += self.gamma * np.max(tm_pred[i])
                m_pred[i][a] = target

            self.model.fit(s_vec, m_pred, epochs = self.epochs, 
                                          batch_size = self.nn_batch_size,
                                          verbose = False)