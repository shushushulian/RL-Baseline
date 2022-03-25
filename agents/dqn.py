import numpy as np
import keras
import random
from keras.layers import Input, Dense, Activation, Dropout
from keras.models import Sequential
from keras.optimizers import Adam

class DQN():
    def __init__(self, env):
        
        # hyperparameters
        self.env = env
        self.gamma = 0.99
        self.nn_learning_rate = 0.0002
        self.nn_batch_size = None
        self.epochs = 1
        self.minibatch_sz = 64
        self.epsilon = 1.
        self.epsilon_decay = 0.992
        self.epsilon_floor = 0.05
        self.n_s = env.observation_space.shape[0]
        self.n_a = env.action_space.n

        self.description = 'DQN Learner'
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
        model = Sequential()
        model.add(Dense(64, input_dim=self.n_s, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.n_a, activation='linear'))
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