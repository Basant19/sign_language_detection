from signLanguage.pipeline.training_pipeline import TrainPipeline
from signLanguage.logger import logging 
import torch
logging.info(f"[Model Trainer] Using device: {'CUDA (GPU)' if torch.cuda.is_available() else 'CPU'}")

obj = TrainPipeline()
obj.run_pipeline()