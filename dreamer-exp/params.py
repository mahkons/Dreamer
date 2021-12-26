
RANDOM_SEED = 239
FROM_PIXELS = True
SEQ_LEN = 20 # 50
BATCH_SIZE = 2 # 50
TRAIN_ITERS_PER_EPISODE = 2 # 100

ED_MODEL_LR = 5e-4
MODEL_LR = 1e-3
ACTOR_LR = 8e-5
CRITIC_LR = 8e-5
REC_L2_REG = 1e-5
FLOW_LOSS_IN_GRU_MULTIPLIER = 0.1
FLOW_LOSS_COEFF = 0.1
ANOTHER_REC_COEFF = 1.

MAX_GRAD_NORM = 100.
MAX_KL = 100.
MIN_STD = 0.1
MODEL_WEIGHT_DECAY = 1e-6

GAMMA = 0.99
LAMBDA = 0.95
HORIZON = 15

STOCH_DIM = 32
DETER_DIM = 256
EMBED_DIM = 256
RSSM_HIDDEN_DIM = 200

FLOW_GRU_DIM = 256
FLOW_HIDDEN_DIM = 512
FLOW_NUM_BLOCKS = 5
FLOW_WEIGHT_DECAY = 1e-6

INIT_STEPS = 10**2 # 10**4
MEMORY_SIZE = 10**6
TOTAL_STEPS = 5*10**6
ACTION_REPEAT = 2
PREDICT_DONE = True
WITH_PRIOR_MODEL = False

