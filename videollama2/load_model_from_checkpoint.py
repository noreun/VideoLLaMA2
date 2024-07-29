import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from model.timeseries_model import TimeSeriesProjector, TimeSeriesMistral

def load_model_from_checkpoint(checkpoint_dir):
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model = AutoModelForCausalLM.from_pretrained(checkpoint_dir, torch_dtype=torch.bfloat16)
    return tokenizer, model

def main():

    # Load the latest checkpoint
    output_dir = './results'
    checkpoints = [os.path.join(output_dir, d) for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print(f"Loading from checkpoint: {latest_checkpoint}")
    else:
        print("No checkpoints found.")
        return

    tokenizer, model = load_model_from_checkpoint(latest_checkpoint)

    # Create a sample input for testing
    sample_text = "Sample text to test the loaded model."
    inputs = tokenizer(sample_text, return_tensors="pt", padding="max_length", max_length=512, truncation=True)

    # Load the custom model for inference
    time_series_projector = TimeSeriesProjector(input_dim=2000, hidden_dim=model.config.hidden_size, output_dim=model.config.hidden_size)
    custom_model = TimeSeriesMistral(model, time_series_projector, attn_implementation="eager")

    # Move the model to the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    custom_model.to(device)

    # Perform inference
    custom_model.eval()
    with torch.no_grad():
        outputs = custom_model(input_ids=inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"].to(device))
        print("Inference outputs:")
        print(outputs)

if __name__ == "__main__":
    main()
