#!/usr/bin/env python3
"""
System Utilities for DDRL Training
Handles GPU detection, resource optimization, and system monitoring
"""

import logging
import os
import time
from datetime import datetime

import numpy as np
import psutil

try:
    import tensorflow as tf

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import GPUtil

    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


class SystemOptimizer:
    """Optimizes system resources for DDRL training"""

    def __init__(self):
        self.gpu_count = 0
        self.gpu_memory = []
        self.cpu_count = 0
        self.total_memory = 0
        self.optimized_config = {}

    def detect_gpus(self):
        """Detect available GPUs and their specifications"""
        gpu_info = []

        if TF_AVAILABLE:
            try:
                gpus = tf.config.list_physical_devices("GPU")
                self.gpu_count = len(gpus)

                for i, gpu in enumerate(gpus):
                    try:
                        # Get GPU memory info
                        gpu_details = tf.config.experimental.get_device_details(gpu)
                        memory_info = gpu_details.get("device_memory_size", 0)
                        gpu_info.append(
                            {
                                "id": i,
                                "name": gpu_details.get("device_name", f"GPU-{i}"),
                                "memory_gb": (
                                    memory_info / (1024**3) if memory_info > 0 else 0
                                ),
                                "memory_mb": (
                                    memory_info / (1024**2) if memory_info > 0 else 0
                                ),
                            }
                        )
                    except:
                        gpu_info.append(
                            {
                                "id": i,
                                "name": f"GPU-{i}",
                                "memory_gb": 0,
                                "memory_mb": 0,
                            }
                        )

            except Exception as e:
                print(f"Warning: Could not detect GPUs via TensorFlow: {e}")

        elif GPU_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                self.gpu_count = len(gpus)

                for gpu in gpus:
                    gpu_info.append(
                        {
                            "id": gpu.id,
                            "name": gpu.name,
                            "memory_gb": gpu.memoryTotal / 1024,
                            "memory_mb": gpu.memoryTotal,
                        }
                    )
            except Exception as e:
                print(f"Warning: Could not detect GPUs via GPUtil: {e}")

        self.gpu_memory = gpu_info
        return gpu_info

    def get_system_info(self):
        """Get comprehensive system information"""
        # CPU info
        self.cpu_count = psutil.cpu_count(logical=True)
        cpu_freq = psutil.cpu_freq()

        # Memory info
        memory = psutil.virtual_memory()
        self.total_memory = memory.total / (1024**3)  # GB

        # Disk info
        disk = psutil.disk_usage("/")

        system_info = {
            "cpu_count": self.cpu_count,
            "cpu_freq_mhz": cpu_freq.max if cpu_freq else 0,
            "total_memory_gb": self.total_memory,
            "available_memory_gb": memory.available / (1024**3),
            "disk_total_gb": disk.total / (1024**3),
            "disk_free_gb": disk.free / (1024**3),
            "gpu_count": self.gpu_count,
            "gpus": self.gpu_memory,
        }

        return system_info

    def optimize_tensorflow(self):
        """Optimize TensorFlow for maximum performance"""
        if not TF_AVAILABLE:
            print("Warning: TensorFlow not available, skipping GPU optimization")
            return {}

        config = {}

        try:
            # Enable memory growth for all GPUs
            gpus = tf.config.list_physical_devices("GPU")
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"✓ Enabled memory growth for {len(gpus)} GPU(s)")
                config["memory_growth"] = True

            # Set mixed precision for better performance
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            print("✓ Enabled mixed precision training")
            config["mixed_precision"] = True

            # Optimize for performance
            tf.config.optimizer.set_jit(True)
            print("✓ Enabled XLA JIT compilation")
            config["jit_compilation"] = True

            # Set thread count for CPU operations
            tf.config.threading.set_inter_op_parallelism_threads(self.cpu_count)
            tf.config.threading.set_intra_op_parallelism_threads(self.cpu_count)
            print(f"✓ Set TensorFlow to use {self.cpu_count} CPU threads")
            config["cpu_threads"] = self.cpu_count

        except Exception as e:
            print(f"Warning: Could not optimize TensorFlow: {e}")

        return config

    def get_optimal_batch_size(self, base_batch_size=32, memory_factor=0.8):
        """Calculate optimal batch size based on available GPU memory"""
        if not self.gpu_memory:
            return base_batch_size

        # Use the GPU with most memory
        max_memory_gb = max([gpu["memory_gb"] for gpu in self.gpu_memory])

        # Rough estimation: 1GB can handle ~1000 samples for typical DQN
        # Adjust based on your model complexity
        estimated_samples_per_gb = 1000
        optimal_batch_size = int(
            max_memory_gb * estimated_samples_per_gb * memory_factor
        )

        # Ensure it's at least the base batch size
        optimal_batch_size = max(optimal_batch_size, base_batch_size)

        print(
            f"✓ Optimal batch size: {optimal_batch_size} (based on {max_memory_gb:.1f}GB GPU memory)"
        )
        return optimal_batch_size

    def get_optimal_memory_size(self, base_memory_size=10000):
        """Calculate optimal memory size based on available RAM"""
        # Use 70% of available RAM for experience replay
        available_gb = psutil.virtual_memory().available / (1024**3)
        optimal_memory = int(available_gb * 1000 * 0.7)  # Rough estimate

        optimal_memory = max(optimal_memory, base_memory_size)

        print(
            f"✓ Optimal memory size: {optimal_memory} (based on {available_gb:.1f}GB available RAM)"
        )
        return optimal_memory

    def optimize_system(self):
        """Run complete system optimization"""
        print("🔧 OPTIMIZING SYSTEM FOR DDRL TRAINING")
        print("=" * 50)

        # Detect system resources
        gpu_info = self.detect_gpus()
        system_info = self.get_system_info()

        # Print system information
        print(f"🖥️  System Resources Detected:")
        print(f"   CPU Cores: {system_info['cpu_count']}")
        print(f"   Total RAM: {system_info['total_memory_gb']:.1f} GB")
        print(f"   Available RAM: {system_info['available_memory_gb']:.1f} GB")
        print(f"   GPUs: {system_info['gpu_count']}")

        for gpu in gpu_info:
            print(f"     - {gpu['name']}: {gpu['memory_gb']:.1f} GB")

        # Optimize TensorFlow
        tf_config = self.optimize_tensorflow()

        # Calculate optimal parameters
        optimal_batch_size = self.get_optimal_batch_size()
        optimal_memory_size = self.get_optimal_memory_size()

        # Store optimization results
        self.optimized_config = {
            "gpu_count": self.gpu_count,
            "gpu_memory": self.gpu_memory,
            "cpu_count": self.cpu_count,
            "total_memory_gb": self.total_memory,
            "optimal_batch_size": optimal_batch_size,
            "optimal_memory_size": optimal_memory_size,
            "tf_config": tf_config,
        }

        print("✅ System optimization completed!")
        return self.optimized_config


class TrainingLogger:
    """Comprehensive logging system for DDRL training"""

    def __init__(self, log_dir="logs", log_level=logging.INFO):
        self.log_dir = log_dir
        self.start_time = time.time()

        # Create log directory
        os.makedirs(log_dir, exist_ok=True)

        # Setup logging
        self.setup_logging(log_level)

        # Training metrics
        self.episode_rewards = []
        self.episode_energy_efficiency = []
        self.episode_conflicts = []
        self.episode_drifts = []
        self.training_losses = []

    def setup_logging(self, log_level):
        """Setup comprehensive logging system"""
        # Create formatters
        detailed_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        simple_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"
        )

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(simple_formatter)

        # File handlers
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Main training log
        training_handler = logging.FileHandler(
            os.path.join(self.log_dir, f"training_{timestamp}.log")
        )
        training_handler.setLevel(logging.INFO)
        training_handler.setFormatter(detailed_formatter)

        # Detailed debug log
        debug_handler = logging.FileHandler(
            os.path.join(self.log_dir, f"debug_{timestamp}.log")
        )
        debug_handler.setLevel(logging.DEBUG)
        debug_handler.setFormatter(detailed_formatter)

        # Setup root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        root_logger.addHandler(console_handler)
        root_logger.addHandler(training_handler)
        root_logger.addHandler(debug_handler)

        # Create specific loggers
        self.training_logger = logging.getLogger("training")
        self.system_logger = logging.getLogger("system")
        self.performance_logger = logging.getLogger("performance")

        self.training_logger.info("🚀 DDRL Training Logger Initialized")
        self.system_logger.info("🔧 System monitoring enabled")
        self.performance_logger.info("📊 Performance tracking enabled")

    def log_episode_start(self, episode, total_episodes):
        """Log episode start with progress"""
        progress = (episode / total_episodes) * 100
        elapsed = time.time() - self.start_time

        self.training_logger.info(
            f"🎯 EPISODE {episode:4d}/{total_episodes} | "
            f"Progress: {progress:5.1f}% | "
            f"Elapsed: {elapsed/3600:.1f}h"
        )

    def log_episode_end(self, episode, reward, ee, conflicts, drift, loss=None):
        """Log episode completion with metrics"""
        # Store metrics
        self.episode_rewards.append(reward)
        self.episode_energy_efficiency.append(ee)
        self.episode_conflicts.append(conflicts)
        self.episode_drifts.append(drift)
        if loss is not None:
            self.training_losses.append(loss)

        # Calculate running averages
        avg_reward = (
            np.mean(self.episode_rewards[-10:])
            if len(self.episode_rewards) >= 10
            else np.mean(self.episode_rewards)
        )
        avg_ee = (
            np.mean(self.episode_energy_efficiency[-10:])
            if len(self.episode_energy_efficiency) >= 10
            else np.mean(self.episode_energy_efficiency)
        )
        avg_conflicts = (
            np.mean(self.episode_conflicts[-10:])
            if len(self.episode_conflicts) >= 10
            else np.mean(self.episode_conflicts)
        )

        self.training_logger.info(
            f"✅ Episode {episode:4d} Complete | "
            f"Reward: {reward:8.3f} (avg: {avg_reward:8.3f}) | "
            f"EE: {ee:12.0f} (avg: {avg_ee:12.0f}) | "
            f"Conflicts: {conflicts:2d} (avg: {avg_conflicts:5.1f}) | "
            f"Drift: {drift:12.0f}"
        )

        if loss is not None:
            self.performance_logger.info(
                f"Episode {episode} | Training Loss: {loss:.6f}"
            )

    def log_step(self, step, reward, ee, conflicts, raw_reward=None):
        """Log individual step details (for debugging)"""
        if step % 100 == 0:  # Log every 100 steps
            self.performance_logger.debug(
                f"Step {step:4d} | Reward: {reward:8.3f} | "
                f"EE: {ee:12.0f} | Conflicts: {conflicts:2d}"
            )
            if raw_reward is not None:
                self.performance_logger.debug(f"Raw Reward: {raw_reward:12.0f}")

    def log_system_status(self):
        """Log current system resource usage"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()

        self.system_logger.info(
            f"System Status | CPU: {cpu_percent:5.1f}% | "
            f"RAM: {memory.percent:5.1f}% ({memory.used/(1024**3):.1f}GB/"
            f"{memory.total/(1024**3):.1f}GB)"
        )

    def log_training_progress(self, episode, total_episodes, metrics):
        """Log comprehensive training progress"""
        progress = (episode / total_episodes) * 100
        elapsed = time.time() - self.start_time
        eta = (elapsed / episode) * (total_episodes - episode) if episode > 0 else 0

        self.training_logger.info(
            f"📊 Training Progress | Episode {episode}/{total_episodes} | "
            f"Progress: {progress:5.1f}% | Elapsed: {elapsed/3600:.1f}h | "
            f"ETA: {eta/3600:.1f}h"
        )

        # Log detailed metrics
        for key, value in metrics.items():
            self.performance_logger.info(f"  {key}: {value}")

    def save_metrics(self, filename="training_metrics.npz"):
        """Save training metrics to file"""
        metrics_file = os.path.join(self.log_dir, filename)

        np.savez(
            metrics_file,
            episode_rewards=self.episode_rewards,
            episode_energy_efficiency=self.episode_energy_efficiency,
            episode_conflicts=self.episode_conflicts,
            episode_drifts=self.episode_drifts,
            training_losses=self.training_losses,
        )

        self.training_logger.info(f"💾 Metrics saved to {metrics_file}")

    def get_summary(self):
        """Get training summary statistics"""
        if not self.episode_rewards:
            return "No training data available"

        summary = {
            "total_episodes": len(self.episode_rewards),
            "avg_reward": np.mean(self.episode_rewards),
            "max_reward": np.max(self.episode_rewards),
            "min_reward": np.min(self.episode_rewards),
            "avg_energy_efficiency": np.mean(self.episode_energy_efficiency),
            "avg_conflicts": np.mean(self.episode_conflicts),
            "total_training_time": time.time() - self.start_time,
        }

        return summary


def setup_environment():
    """Setup optimal environment variables for training"""
    # Set TensorFlow environment variables
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Reduce TF logging
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"

    # Set CUDA environment variables
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # Use all available GPUs

    print("✓ Environment variables configured for optimal performance")


def analyze_system_capabilities():
    """Extract raw system specs"""
    import subprocess

    import psutil

    print("SYSTEM SPECS:")

    # CPU
    cpu_cores = psutil.cpu_count()
    cpu_freq = psutil.cpu_freq()
    print(f"CPU Cores: {cpu_cores}")
    print(f"CPU Frequency: {cpu_freq.max if cpu_freq else 0:.0f} MHz")

    # RAM
    ram_total = psutil.virtual_memory().total / (1024**3)
    ram_available = psutil.virtual_memory().available / (1024**3)
    print(f"Total RAM: {ram_total:.1f} GB")
    print(f"Available RAM: {ram_available:.1f} GB")

    # GPU
    gpu_count = 0
    gpu_vram = 0
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            gpu_count = len(lines)
            for i, line in enumerate(lines):
                name, memory = line.split(", ")
                vram_gb = int(memory) / 1024
                gpu_vram = max(gpu_vram, vram_gb)
                print(f"GPU {i}: {name} - {vram_gb:.1f} GB VRAM")
        else:
            print("No GPUs detected")
    except:
        print("No GPUs detected")

    print(f"\nSUMMARY:")
    print(f"CPU Cores: {cpu_cores}")
    print(f"RAM: {ram_total:.1f} GB")
    print(f"GPU Count: {gpu_count}")
    print(f"Max GPU VRAM: {gpu_vram:.1f} GB")


if __name__ == "__main__":
    # Run system analysis
    config = analyze_system_capabilities()
