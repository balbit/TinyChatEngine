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

def profile_t5(model, seq_len=77, device="cpu", print_stats=True):
    VOCAB_SIZE = model.config.vocab_size

    model = model.to(device)
    input_ids = torch.randint(0, 32100, (1, seq_len)).to(device)

    macs, results = CustomMACProfiler().profile_macs(t5, input_ids, reduction=sum, return_dict_format='both')
    if print_stats:
        print(f"GOPs: {macs / (10**9)}")
        print(results)
    return macs, results

if __name__ == "__main__":
    local_model_path = "./saved_models/t5_encoder_3"
    model_name = "stabilityai/stable-diffusion-3-medium-diffusers"
    subfolder = "text_encoder_3"
    variant = "fp16"

    t5 = load_or_download_model(local_model_path, model_name, subfolder=subfolder, variant=variant)
    print("model loaded")

    
    profile_t5(t5)
