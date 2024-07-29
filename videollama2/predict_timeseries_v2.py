import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import sys

# Get the absolute path of the project root directory
project_root = os.path.dirname(os.path.abspath(__file__)) 

# Add the project root to sys.path
rrd = f"{project_root}/../"
print(rrd)
sys.path.insert(0, rrd) 

from videollama2 import TimeSeriesProjector, TimeSeriesMistral

# 1. Load the tokenizer and model
model_path = "/home/barbosa/rihome/projects/TSLLaMA2/models/mistral_ts_projector"  # Your model directory
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = TimeSeriesMistral.from_pretrained(model_path, torch_dtype=torch.bfloat16) # Assuming you trained with BF16
model = model.to('cuda') # Move to GPU

# 2. Prepare time series data for a single example
# ... (Your code to load and preprocess a time series into a tensor) ...
# time_series = ... 
time_series_projector = TimeSeriesProjector(input_dim=2000, hidden_dim=mistral_model.config.hidden_size, output_dim=mistral_model.config.hidden_size)

# 3. Create dummy input_ids and attention mask for text generation
input_ids = tokenizer("<pad> <pad>", return_tensors="pt").input_ids  # Or use a special "start" token
attention_mask = torch.zeros_like(input_ids)

# 4. Generate text
with torch.no_grad():
    outputs = model(
        input_ids=input_ids.to('cuda'), 
        attention_mask=attention_mask.to('cuda'), 
        time_series=time_series.to('cuda')
    )

# 5. Decode the output tokens
generated_text = tokenizer.decode(outputs.logits[0].argmax(dim=-1))
print(generated_text)