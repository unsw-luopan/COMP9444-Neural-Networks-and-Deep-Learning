import gym
import tensorflow as tf
import numpy as np
import random

# General Parameters
# -- DO NOT MODIFY --
ENV_NAME = 'CartPole-v0'
EPISODE = 200000  # Episode limitation
STEP = 200  # Step limitation in an episode
TEST = 10  # The number of tests to run every TEST_FREQUENCY episodes
TEST_FREQUENCY = 100  # Num episodes to run before visualizing test accuracy

# TODO: HyperParameters
GAMMA = 0.99  # discount factor
INITIAL_EPSILON = 0.5  # starting value of epsilon
FINAL_EPSILON = 0.01  # final value of epsilon
EPSILON_DECAY_STEPS = 100  # decay period
BATCH_SIZE = 128
REPLAY_SIZE = 30000
HIDDEN_NODES = 30
# Create environment
# -- DO NOT MODIFY --
env = gym.make(ENV_NAME)
epsilon = INITIAL_EPSILON
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.n

# Placeholders
# -- DO NOT MODIFY --
state_in = tf.placeholder("float", [None, STATE_DIM])
action_in = tf.placeholder("float", [None, ACTION_DIM])
target_in = tf.placeholder("float", [None])


# TODO: Define Network Graph
def weight(shape):
    initial = tf.truncated_normal(shape)
    return tf.Variable(initial)


def bias(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)


W1 = weight([STATE_DIM, HIDDEN_NODES])
b1 = bias([HIDDEN_NODES])
W2 = weight([HIDDEN_NODES, HIDDEN_NODES])
b2 = bias([1, HIDDEN_NODES])
W3 = weight([HIDDEN_NODES, ACTION_DIM])
b3 = bias([1, ACTION_DIM])
# hidden layers
h1_layer = tf.matmul(state_in, W1) + b1
h1_out = tf.nn.relu(h1_layer)
h2_layer = tf.matmul(h1_out, W2) + b2
h2_out = tf.nn.relu(h2_layer)
h3_layer = tf.matmul(h2_out, W3) + b3
h3_out = h3_layer

# TODO: Network outputs

q_values = h3_out
q_action = tf.reduce_sum(tf.multiply(q_values, action_in), reduction_indices=1)
# TODO: Loss/Optimizer Definition
loss = tf.reduce_mean(tf.square(target_in - q_action))
optimizer = tf.train.AdamOptimizer(0.003).minimize(loss)

# Start session - Tensorflow housekeeping
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())

replay_buffer = []


def buffer_train(replay_buffer, state_in, action_in, target_in,
                 q_values):
    replay_buffer.append((state, action, reward, next_state, done))
    if len(replay_buffer) > REPLAY_SIZE:
        replay_buffer.pop(0)
    if len(replay_buffer) > BATCH_SIZE:
        train(replay_buffer, state_in, action_in, target_in, q_values)



def divide_batch(q_values, state_in, minibatch):
    state_batch = [data[0] for data in minibatch]
    action_batch = [data[1] for data in minibatch]
    reward_batch = [data[2] for data in minibatch]
    next_state_batch = [data[3] for data in minibatch]
    target_batch = []
    Q_value_batch = q_values.eval(feed_dict={state_in: next_state_batch})
    for i in range(0, BATCH_SIZE):
        done = minibatch[i][4]
        if done:
            target_batch.append(reward_batch[i])
        else:
            target_batch.append(reward_batch[i] + GAMMA * np.max(Q_value_batch[i]))
    return target_batch, state_batch, action_batch
    # Do one training step


def train(replay_buffer, state_in, action_in, target_in, q_values):
    minibatch = random.sample(replay_buffer, BATCH_SIZE)
    target_batch, state_batch, action_batch = divide_batch(q_values, state_in, minibatch)
    session.run([optimizer], feed_dict={
        target_in: target_batch,
        action_in: action_batch,
        state_in: state_batch
    })



# -- DO NOT MODIFY ---
def explore(state, epsilon):
    """
    Exploration function: given a state and an epsilon value,
    and assuming the network has already been defined, decide which action to
    take using e-greedy exploration based on the current q-value estimates.
    """
    Q_estimates = q_values.eval(feed_dict={
        state_in: [state]
    })
    if random.random() <= epsilon:
        action = random.randint(0, ACTION_DIM - 1)
    else:
        action = np.argmax(Q_estimates)
    one_hot_action = np.zeros(ACTION_DIM)
    one_hot_action[action] = 1
    return one_hot_action


# Main learning loop
for episode in range(EPISODE):
    # initialize task
    state = env.reset()

    # Update epsilon once per episode
    if epsilon > FINAL_EPSILON:
        epsilon -= (epsilon - FINAL_EPSILON) / EPSILON_DECAY_STEPS

    # Move through env according to e-greedy policy
    for step in range(STEP):
        action = explore(state, epsilon)
        next_state, reward, done, _ = env.step(np.argmax(action))
        nextstate_q_values = q_values.eval(feed_dict={
            state_in: [next_state]
        })

        # TODO: Calculate the target q-value.
        # hint1: Bellman
        # hint2: consider if the episode has terminated
        if done:
            target = reward
        else:
            target = reward + GAMMA * np.max(nextstate_q_values)

        # Update
        if episode < 100:
            buffer_train(replay_buffer, state_in, action_in, target_in,
                         q_values)
        
        state = next_state
        if done:
            break

    # Test and view sample runs - can disable render to save time
    # -- DO NOT MODIFY --
    if (episode % TEST_FREQUENCY == 0 and episode != 0):
        total_reward = 0
        for i in range(TEST):
            state = env.reset()
            for j in range(STEP):
                env.render()
                action = np.argmax(q_values.eval(feed_dict={
                    state_in: [state]
                }))
                state, reward, done, _ = env.step(action)
                total_reward += reward
                if done:
                    break
        ave_reward = total_reward / TEST
        print('episode:', episode, 'epsilon:', epsilon, 'Evaluation '
                                                        'Average Reward:', ave_reward)

env.close()
