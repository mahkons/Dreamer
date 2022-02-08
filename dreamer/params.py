

RANDOM_SEED = 239
FROM_PIXELS = True
SEQ_LEN = 20
BATCH_SIZE = 2
TRAIN_ITERS_PER_EPISODE = 2

MODEL_LR = 6e-4
ACTOR_LR = 8e-5
CRITIC_LR = 8e-5
MAX_GRAD_NORM = 100.
MAX_KL = 100.
MIN_STD = 0.1

GAMMA = 0.99
LAMBDA = 0.95
HORIZON = 15

STOCH_DIM = 32
DETER_DIM = 256
EMBED_DIM = 256
RSSM_HIDDEN_DIM = 200

INIT_STEPS = 10**5
MEMORY_SIZE = 10**5
TOTAL_STEPS = 10**5
ACTION_REPEAT = 2
PREDICT_DONE = True



