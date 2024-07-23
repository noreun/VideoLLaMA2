import os
import pickle
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, PreTrainedModel

class TimeSeriesDataset(Dataset):
    """Dataset for time series data and associated text from a CSV."""

    def __init__(self, ts_path, text_path, tokenizer, prediction_target='description'):
        super().__init__()
        self.tokenizer = tokenizer
        self.ts_path = ts_path
        self.text_path = text_path
        self.prediction_target = prediction_target

        self.data_df = pd.read_csv(text_path)

        # Filter time series files based on CSV entries:
        self.ts_files = [
            f for f in os.listdir(ts_path) if f.endswith('_ts.pkl') 
            and f"{f.split('_')[0]}.mp4" in self.data_df['video_names'].values 
        ] 

        print(f"Found {len(self.ts_files)} matching time series files.")

    # def __init__(self, ts_path, text_path, tokenizer, prediction_target='description'):
    #     super().__init__()
    #     self.tokenizer = tokenizer
    #     self.ts_path = ts_path
    #     self.text_path = text_path
    #     self.prediction_target = prediction_target 

    #     self.data_df = pd.read_csv(text_path)
    #     self.ts_files = [f for f in os.listdir(ts_path) if f.endswith('_ts.pkl')]

    def __len__(self):
        return len(self.ts_files)

    def __getitem__(self, idx):
        ts_file = self.ts_files[idx]
        ts_file_path = os.path.join(self.ts_path, ts_file)
        with open(ts_file_path, 'rb') as f:
            time_series_data, _ = pickle.load(f) 

        time_series_data = torch.tensor(time_series_data.T, dtype=torch.float)
        time_series_data = time_series_data[:2000, :] # Truncate

        record_id = ts_file.split('_')[0]
        # row = self.data_df[self.data_df['video_names'] == f"{record_id}.mp4"].iloc[0]

        matching_rows = self.data_df[self.data_df['video_names'] == f"{record_id}.mp4"]

        if len(matching_rows) > 0:
            row = matching_rows.iloc[0]

            if self.prediction_target == 'description':
                target_text = row['descriptions']
            elif self.prediction_target == 'valence':
                target_text = str(row['overall'])
            elif self.prediction_target == 'emotion':
                target_text = row['one_word']
            else:
                raise ValueError(f"Invalid prediction_target: {self.prediction_target}")

            # Tokenize the target text
            target_text_encoded = self.tokenizer(
                target_text,
                return_tensors="pt",
                padding="max_length",
                max_length=512,  # Adjust this as needed
                truncation=True,
            )

            return {
                'input_ids': target_text_encoded.input_ids.squeeze(0), # Remove batch dimension
                'attention_mask': target_text_encoded.attention_mask.squeeze(0),
                'time_series': time_series_data 
            }
        else:
            # Handle the case where no matching row is found
            print(f"Warning: No matching row found for record_id: {record_id}")
            # You can either return a default value or raise an error here
            return None  # Or raise an exception



class TimeSeriesProjector(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.layers(x)


class TimeSeriesMistral(nn.Module):
# class TimeSeriesMistral(PreTrainedModel):
    def __init__(self, mistral_model, time_series_projector, time_series_embedding_dim=4096):
    # def __init__(self, config, mistral_model, time_series_projector, time_series_embedding_dim=4096):
        super().__init__()
        # super().__init__(config) # Initialize PreTrainedModel
        self.mistral_model = mistral_model
        self.time_series_projector = time_series_projector
        self.time_series_embedding = nn.Embedding(2000, time_series_embedding_dim)  

    def forward(self, input_ids, attention_mask, time_series, labels=None):

        # Project the time series
        projected_time_series = self.time_series_projector(time_series).to(torch.bfloat16) 

        # Embed the time series positions
        time_series_positions = torch.arange(time_series.shape[1], dtype=torch.long, device=time_series.device).unsqueeze(0) 
        time_series_embedding = self.time_series_embedding(time_series_positions).to(torch.bfloat16)  

        # Add projected time series features to the time series embedding 
        time_series_embedding = time_series_embedding + projected_time_series

        # Get the text embeddings
        text_embeddings = self.mistral_model.get_input_embeddings()(input_ids).to(torch.bfloat16)
        print("Text embeddings:")
        print(text_embeddings.shape)

        # Concatenate along the sequence dimension (dim=1)
        combined_embeddings = torch.cat([text_embeddings, time_series_embedding], dim=1)
        print("Combined embeddings:")
        print(combined_embeddings.shape)

        # Adjust the attention mask
        attention_mask = attention_mask.to(torch.bfloat16)
        attention_mask = torch.cat([attention_mask, 
                                    torch.ones(attention_mask.shape[0], time_series.shape[1], dtype=torch.long, device=attention_mask.device)], dim=1)

        # Pass through the Mistral model
        outputs = self.mistral_model(
            inputs_embeds=combined_embeddings,
            attention_mask=attention_mask,
            labels=labels
        )
        
        print("Outputs:")
        print(outputs.__class__)
        print(outputs.logits.shape)
        print(outputs.logits.dtype)

        # Calculate the loss (if labels are provided)
        if labels is not None:
            # Shift labels for next token prediction
            shifted_labels = labels[..., 1:].contiguous()
            shifted_logits = outputs.logits[..., :-1, :].contiguous()
            loss = F.cross_entropy(shifted_logits.view(-1, shifted_logits.size(-1)), shifted_labels.view(-1)) 

            # Return the loss and the original outputs 
            return loss, outputs
        else:
            # Return only the outputs during inference 
            return outputs

        return outputs
    
    # def forward(self, input_ids, attention_mask, time_series, labels=None):
    #     time_series_embedding = self.time_series_projector(time_series)
    #     # Prepend the time series embedding to the input_ids
    #     # We assume a batch size of 1 for simplicity
    #     input_embeds = self.mistral_model.get_input_embeddings()(input_ids)
    #     combined_input_embeds = torch.cat([time_series_embedding.unsqueeze(0), input_embeds], dim=1)
        
    #     # Adjust attention mask
    #     attention_mask = torch.cat([torch.ones(1, time_series_embedding.shape[1], dtype=torch.long, device=attention_mask.device), 
    #                                 attention_mask], dim=1)

    #     outputs = self.mistral_model(
    #         inputs_embeds=combined_input_embeds,
    #         attention_mask=attention_mask,
    #         labels=labels,
    #     )

    #     return outputs

def main():
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    # model_name = "mistralai/Mistral-7B-Instruct-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load model with BF16 support:
    mistral_model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        # torch_dtype=torch.float16, # Enable Float16
        torch_dtype=torch.bfloat16, # Enable BF16
    )

    # mistral_model = AutoModelForCausalLM.from_pretrained(model_name)

    # Enable gradient checkpointing on Mistral:
    mistral_model.gradient_checkpointing_enable()

    time_series_projector = TimeSeriesProjector(input_dim=2000, hidden_dim=mistral_model.config.hidden_size, output_dim=mistral_model.config.hidden_size)

    model = TimeSeriesMistral(mistral_model, time_series_projector)

    ts_path = "/mnt/nfs/proj/hnl_downloaded_public_data/PFCTS/"
    text_path = "/mnt/nfs/proj/hnl_downloaded_public_data/clip_description.csv"
    prediction_target = 'description' # 'description', 'valence', 'emotion'

    train_dataset = TimeSeriesDataset(ts_path, text_path, tokenizer, prediction_target=prediction_target)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=1,
        # gradient_checkpointing=True,
        gradient_accumulation_steps=8,
        save_steps=1000,
        save_total_limit=2,
        fp16=True,  # Enable mixed precision training
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset)
    trainer.train()

    # Save the trained model
    trainer.save_model("/home/barbosa/rihome/projects/TSLLaMA2/models/mistral_ts_projector")

if __name__ == "__main__":
    main()