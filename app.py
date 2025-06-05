from signLanguage.pipeline.training_pipeline import TrainPipeline
from signLanguage.logger import logging 
import torch

device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only"
logging.basicConfig(level=logging.INFO)
logging.info(f"[App] Device in use: {device_name}")




obj = TrainPipeline()
obj.run_pipeline()