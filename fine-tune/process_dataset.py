
from datasets import load_from_disk
import matplotlib.pyplot as plt
import numpy as np
import os
from transformers import AutoImageProcessor, AutoModelForVideoClassification, AutoConfig, TrainingArguments, Trainer
from pathlib import Path
from peft import LoraConfig, get_peft_model
from multiprocessing import Pool, cpu_count



#load unprocessed dataset
dataset = load_from_disk("C:\\Users\\brand\\desktop\\workspace\\f1-telemetry-app\\extract-frames\\datasets\\datasets_2025-09-25_13-20-40")


model_name = "MCG-NJU/videomae-base-finetuned-kinetics"



# condfig to overide number of classes
config = AutoConfig.from_pretrained(model_name)
config.num_labels = len(dataset["label"])
# take raw video frames, resize to right size, convert to pytorch tensors, normalize pixel values
processor = AutoImageProcessor.from_pretrained(model_name)

model = AutoModelForVideoClassification.from_pretrained(
    model_name,
    config = config,
    ignore_mismatched_sizes = True
)


# wrap VideoMae model with LoRA


# LoRA config
lora_config = LoraConfig(
    r = 16, # try 4-32
    lora_alpha = 32, # scaling factor,
    target_modules = ["query", "value"], # attention layers to apply LoRA to
    lora_dropout = 0.1,
    bias = "none",
    task_type = "FEATURE_EXTRACTION" # find a better way for this
)

model = get_peft_model(model, lora_config) # wrapping model

model.print_trainable_parameters() # see trainable parameters





# dataset = dataset.select(range(1)) # testing on 1 example
# change dataset structure
def preprocess_shard(example) :
    all_pixel_values = []
    all_labels = []

    for video, label in zip(example["video"], example["label"]) :
        frames = np.array(video)
        frames_list = []

        for frame in frames :
            frames_list.append(frame)     
        inputs = processor(frames_list, return_tensors = "pt")
        all_pixel_values.append(inputs["pixel_values"].squeeze(0))
        all_labels.append(label)

    return {
        "pixel_values" : all_pixel_values,
        "labels" : all_labels
    }



if __name__ == "__main__":
    n_cores = cpu_count()
    print(f"Using {n_cores} CPU cores...")

    processed_dataset = dataset.map(preprocess_shard, batched = True, batch_size = 1, remove_columns = "video", num_proc = n_cores)

    output_dir = Path("C:\\Users\\brand\\desktop\\workspace\\f1-telemetry-app\\fine-tune\\preprocessed_dataset")
    output_dir.mkdir(exist_ok = True)

    processed_dataset.save_to_disk(output_dir)

    os.system("shutdown /s /t 600")
