import transformers
import os
import warnings
from transformers import T5EncoderModel
from custom_profiler import profile_macs, CustomMACProfiler
import torch
from torch.profiler import profile, record_function, ProfilerActivity
import time
import argparse

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

def profile_t5_macs(model, seq_len=77, device="cpu", print_stats=True):
    VOCAB_SIZE = model.config.vocab_size

    model = model.to(device)
    input_ids = torch.randint(0, 32100, (1, seq_len)).to(device)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        macs, results = CustomMACProfiler().profile_macs(model, input_ids, reduction=sum, return_dict_format='both')
    if print_stats:
        print(f"GOPs: {macs / (10**9)}")
        print(results)
    return macs, results

def profile_t5_speed(model, seq_len=77, device="cpu"):
    VOCAB_SIZE = model.config.vocab_size

    model = model.to(device)
    input_ids = torch.randint(0, 32100, (1, seq_len)).to(device)
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            model(input_ids)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=15))

def measure_wall_clock_time(model, seq_len=77, device="cpu"):
    model = model.to(device)
    input_ids = torch.randint(0, 32100, (1, seq_len)).to(device)
    
    start_time = time.time()
    model(input_ids)
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    print(f"Wall-clock time for model inference: {elapsed_time:.6f} seconds")
    return elapsed_time

if __name__ == "__main__":
    local_model_path = "./saved_models/t5_encoder_3"
    model_name = "stabilityai/stable-diffusion-3-medium-diffusers"
    subfolder = "text_encoder_3"
    variant = "fp16"

    t5 = load_or_download_model(local_model_path, model_name, subfolder=subfolder, variant=variant)
    print("model loaded")

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Profile T5 model with specified sequence length.")
    parser.add_argument("--seq_len", type=int, default=80, help="Sequence length for the input tensor.")
    args = parser.parse_args()

    seq_len = args.seq_len
    profile_t5_macs(seq_len=seq_len, model=t5)

    num_threads = 6
    torch.set_num_threads(num_threads)
    print(f"Number of torch threads: {torch.get_num_threads()}")
    profile_t5_speed(seq_len=seq_len, model=t5)

    measure_wall_clock_time(seq_len=seq_len, model=t5)
