import numpy as np
import keras
import random
from keras.layers import Input, Dense, Activation, Dropout
from keras.models import Sequential
from keras.optimizers import Adam

class PrioritizedDDQN():
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
        self.description = 'Prioritized Replay Double DQN'
        self.update_frequency = 30
        self.verbose = False

        # memory
        self.memory_max = 50000
        self.reset()

    def reset(self):
        self.epsilon = 1.
        self.step = 0
        self.memory = [] 

        # Prioritized replay
        self.pr_alpha = 0.6
        self.pr_weights = np.array([])
        self.pr_epsilon = 0.01
        self.pr_max = 10
        self.pr_beta = 0.4
        self.pr_beta_incr = 1.005

        # create nn's
        self.model = self._make_model_()
        self.target_model = self._make_model_()


    def _make_model_(self):

        model = Sequential()
        model.add(Dense(64, input_dim=self.n_s, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.n_a, activation = 'linear'))
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

        if len(self.memory) == self.memory_max:
            i = np.argmin(self.pr_weights[:len(self.memory)])  # 给出水平方向最小值的下标
            self.memory.pop(i)  # 删除memory中的第i个元素并返回值
            self.pr_weights = np.delete(self.pr_weights, i)

        self.memory.append([s, a, r, s_prime, done])
        self.pr_weights = np.append(self.pr_weights, self.pr_max)

        self._batch_train_()

        if self.step % self.update_frequency == 0:
            self._model_update_()
        if done and self.epsilon > self.epsilon_floor:
            self.epsilon = self.epsilon * self.epsilon_decay
            self.pr_beta = min(1, self.pr_beta*self.pr_beta_incr)
            if self.verbose: print(self.pr_beta, max(self.pr_weights), np.mean(self.pr_weights))
        self.step += 1


    def _batch_train_(self):
        if len(self.memory) > self.minibatch_sz:

            # calculate weights and sample
            # np.float_power第一个数组元素在元素方面从第二个数组提升为幂
            weight_exp = np.float_power(self.pr_weights[:len(self.memory)], self.pr_alpha)
            p = weight_exp / np.sum(weight_exp)
            batch_ids = np.random.choice(
                range(len(self.memory)),
                self.minibatch_sz,
                replace=True, p=p
            )
            # batch = np.random.choice(self.memory, self.minibatch_sz, replace=True, p=p)
            batch = [self.memory[i] for i in batch_ids]
            print(len(batch))

            # get predictions and td-error
            s_vec = np.array([x[0] for x in batch])
            sp_vec = np.array([x[3] for x in batch])

            m_pred = self.model.predict(s_vec)
            m_pred_sp = self.model.predict(sp_vec)
            tm_pred = self.target_model.predict(sp_vec)

            #用于返回一个numpy数组中最大值的索引值。当一组中同时出现几个最大值时，返回第一个最大值的索引值
            a_maxes = np.argmax(m_pred_sp, axis=1)
            td_errors = np.zeros(len(batch))

            for i in range(len(batch)): 
                s, a, r, s_prime, done = batch[i]
                target = r
                if not done:
                    Q_target_max = tm_pred[i][a_maxes[i]]
                    target += self.gamma * Q_target_max
                td_errors[i] = np.abs(target - m_pred[i][a])
                m_pred[i][a] = target
                
            # update priority in memory and calculate importance sample weights
            is_weights = np.zeros(len(batch))

            for sample_i, memory_i in enumerate(batch_ids):
                self.pr_weights[memory_i] = td_errors[sample_i] + self.pr_epsilon
                is_weights[sample_i] = np.float_power(len(self.memory)*p[memory_i], -self.pr_beta)

            is_weights = is_weights / max(is_weights)


            self.model.fit(s_vec, m_pred, epochs=self.epochs,
                                          sample_weight=is_weights,
                                          batch_size=self.nn_batch_size,
                                          verbose=False)