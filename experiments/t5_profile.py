import torch
import transformers
import os
from transformers import T5EncoderModel
from custom_profiler import profile_macs, CustomMACProfiler

def load_or_download_model(local_path, model_name, subfolder=None, variant=None):
    if os.path.exists(local_path):
        print(f"Loading model from {local_path}...")
        model = T5EncoderModel.from_pretrained(local_path)
    else:
        print(f"Model not found at {local_path}. Downloading from Hugging Face Hub...")
        model = T5EncoderModel.from_pretrained(
            model_name,
            subfolder=subfolder,
            variant=variant
        )
        # Save the model locally for future use
        os.makedirs(local_path, exist_ok=True)
        model.save_pretrained(local_path)
        print(f"Model saved to {local_path}.")
    
    return model

if __name__ == "__main__":
    local_model_path = "./saved_models/t5_encoder_3"
    model_name = "stabilityai/stable-diffusion-3-medium-diffusers"
    subfolder = "text_encoder_3"
    variant = "fp16"

    t5 = load_or_download_model(local_model_path, model_name, subfolder=subfolder, variant=variant)
    print("model loaded")

    macs, breakdown = CustomMACProfiler().profile_macs(t5, reduction=sum, return_dict_format='both')

    print("Macs: "+ str(macs))
    print(breakdown)
