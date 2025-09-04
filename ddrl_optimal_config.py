# DDRL Training Configuration - Auto-generated
# Generated for: 8 CPU cores, 8.0GB RAM, 0 GPUs

# Training Parameters
DDRL_EPISODES = 1600
DDRL_BATCH_SIZE = 128
DDRL_MEMORY_SIZE = 677
DDRL_LEARNING_RATE = 0.0005
DDRL_GAMMA = 0.7
DDRL_TARGET_UPDATE = 10

# System Parameters
PARALLEL_AGENTS = 4
TRAINING_MODE = "CPU"
USE_GPU = False

# Environment Parameters
NUM_UE = 20  # Adjust based on your scenario
NUM_EDGE = 2  # Adjust based on your scenario
NUM_TIME = 100  # Adjust based on your scenario

# Logging
LOG_LEVEL = "INFO"
LOG_DIR = "logs"
SAVE_MODEL_EVERY = 100  # Save model every N episodes
