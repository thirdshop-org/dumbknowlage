import os
import logging
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

for mod in ["pynvml", "nvidia", "chromadb", "onnxruntime"]:
    logging.getLogger(mod).setLevel(logging.ERROR)

warnings.filterwarnings("ignore", message=".*NVML.*")
warnings.filterwarnings("ignore", message=".*nvml.*")
warnings.filterwarnings("ignore", message=".*onnxruntime.*cuda.*")
warnings.filterwarnings("ignore", message=".*GPU.*")
