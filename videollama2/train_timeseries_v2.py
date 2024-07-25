import os
import pickle
import json
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, PreTrainedModel
from transformers import DataCollatorForLanguageModeling
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from typing import List, Dict, Any

class DataCollatorWithPaddingAndTimeSeries(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, mlm: bool = False):
        super().__init__(tokenizer=tokenizer, mlm=mlm)
        self.tokenizer = tokenizer
        self.mlm = mlm
        if self.mlm and not self.tokenizer.mask_token:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. "
                "You should pass `mlm=False` to train on causal language modeling instead."
            )

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        def collate_fn(batch):
            # Filter out None values
            batch = [item for item in batch if item is not None]
            
            # Ensure the batch is not empty
            if len(batch) == 0:
                return {}

            return {
                'input_ids': torch.stack([item['input_ids'] for item in batch]),
                'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
                'time_series': torch.stack([item['time_series'] for item in batch]),
                'labels': torch.stack([item['labels'] for item in batch])
            }

        return collate_fn(features)

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

            input_ids = target_text_encoded.input_ids.squeeze(0)
            labels = input_ids.clone()
            
            # Shift the labels
            labels = torch.roll(labels, shifts=-1, dims=0)
            labels[-1] = self.tokenizer.pad_token_id  # Ensure the last token is the pad token

            return {
                'input_ids': input_ids,  # Remove batch dimension
                'attention_mask': target_text_encoded.attention_mask.squeeze(0),
                'time_series': time_series_data,
                'labels': labels  # Include shifted labels in the dictionary
            }
        
        else:
            # Handle the case where no matching row is found
            print(f"Warning: No matching row found for record_id: {record_id}")
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


# class TimeSeriesMistral(nn.Module):
class TimeSeriesMistral(PreTrainedModel):
    def __init__(self, mistral_model, time_series_projector, time_series_embedding_dim=4096, attn_implementation="eager"):
    # def __init__(self, config, mistral_model, time_series_projector, time_series_embedding_dim=4096):
        # super().__init__()
        # super().__init__(config) # Initialize PreTrainedModel

        # Change in __init__ method of TimeSeriesMistral class
        super().__init__(mistral_model.config, attn_implementation=attn_implementation) # Initialize PreTrainedModel
        self.mistral_model = mistral_model
        self.time_series_projector = time_series_projector
        self.time_series_embedding = nn.Embedding(2000, time_series_embedding_dim)  

    def forward(self, input_ids=None, attention_mask=None, time_series=None, labels=None):

        # Project the time series
        projected_time_series = self.time_series_projector(time_series).to(torch.bfloat16)
        
        # Embed the time series positions
        time_series_positions = torch.arange(time_series.shape[1], dtype=torch.long, device=time_series.device).unsqueeze(0)
        time_series_embedding = self.time_series_embedding(time_series_positions).to(torch.bfloat16)

        # Add projected time series features to the time series embedding
        time_series_embedding = time_series_embedding + projected_time_series

        # Get the text embeddings
        text_embeddings = self.mistral_model.get_input_embeddings()(input_ids).to(torch.bfloat16)

        # Concatenate along the sequence dimension (dim=1)
        combined_embeddings = torch.cat([text_embeddings, time_series_embedding], dim=1)

        # Adjust the attention mask
        attention_mask = attention_mask.to(torch.bfloat16)
        attention_mask = torch.cat([attention_mask,
                                    torch.ones(attention_mask.shape[0], time_series.shape[1], dtype=torch.long, device=attention_mask.device)], dim=1)

        # Pass through the Mistral model
        outputs = self.mistral_model(
            inputs_embeds=combined_embeddings,
            attention_mask=attention_mask,
            # labels=labels
        )

        if labels is not None:

            # Extract the logits corresponding to the text tokens
            text_logits = outputs.logits[:, :input_ids.shape[1], :]  

            # print(labels)
            # print(input_ids)

            # Manually shift the labels
            shifted_labels = torch.roll(labels, shifts=-1, dims=1)
            shifted_labels[:, -1] = -100  # Set the last token to the ignore index

            # print(shifted_labels)
            # print(torch.unique(shifted_labels))

            # print(self.mistral_model.config.vocab_size)

            # Convert shifted_labels to one-hot encoding
            # one_hot_labels = F.one_hot(shifted_labels, num_classes=self.mistral_model.config.vocab_size).float()

            # print("embeddings:")
            # print(text_embeddings.shape)
            # print(text_embeddings.dtype)
            # print(time_series_embedding.shape)
            # print(time_series_embedding.dtype)
            # print(combined_embeddings.shape)
            # print(combined_embeddings.dtype)

            # print("Labels:")
            # print(labels.__class__)
            # print(labels.shape)

            # print("Shifted labels:")
            # print(shifted_labels.__class__)
            # print(shifted_labels.shape)

            # # print("One-hot labels:")
            # # print(one_hot_labels.shape)
            # # print(one_hot_labels.dtype)

            # print("Text logits:")
            # print(text_logits.shape)
            # print(text_logits.dtype)

            # print("Outputs:")
            # print(outputs.__class__)
            # print(outputs.logits.shape)
            # print(outputs.logits.dtype)

            # print("text_logits after view:")
            # print(text_logits.view(-1, text_logits.size(-1)).shape)
                        
            if torch.isnan(text_logits).any() or torch.isinf(text_logits).any():
                print("Warning: NaN or Inf values in text_logits!")
            # if torch.isnan(one_hot_labels).any() or torch.isinf(one_hot_labels).any():
                # print("Warning: NaN or Inf values in one_hot_labels!")

            # Calculate the loss using only the text logits and one-hot labels

            loss = F.cross_entropy(text_logits.view(-1, text_logits.size(-1)), shifted_labels.view(-1), ignore_index=-100)

            # loss = F.cross_entropy(text_logits.view(-1, text_logits.size(-1)), shifted_labels)
            # loss = F.cross_entropy(text_logits.view(-1, text_logits.size(-1)), one_hot_labels.view(-1, one_hot_labels.size(-1)))

            # loss = F.cross_entropy(outputs.logits, one_hot_labels)  # Use one-hot labels 
            # loss = outputs.loss # Use the loss computed by Mistral
            # loss = F.cross_entropy(outputs.logits.view(-1, outputs.logits.size(-1)), padded_labels.view(-1))

            # loss = outputs.loss
            return loss, outputs
        else:
            return outputs


    def compute_loss(model, inputs, return_outputs=False): 
        """Computes the loss for the time series Mistral model."""
        labels = inputs.pop("labels")
        outputs = model(**inputs) 

        # Calculate the loss
        shifted_labels = labels[..., 1:].contiguous()
        shifted_logits = outputs.logits[..., :-1, :].contiguous()
        loss = F.cross_entropy(shifted_logits.view(-1, shifted_logits.size(-1)), shifted_labels.view(-1))

        # Return the loss and outputs (if requested)
        return (loss, outputs) if return_outputs else loss


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
        attn_implementation="eager",
    )

    # mistral_model = AutoModelForCausalLM.from_pretrained(model_name)

    # Enable gradient checkpointing on Mistral:
    mistral_model.gradient_checkpointing_enable()

    time_series_projector = TimeSeriesProjector(input_dim=2000, hidden_dim=mistral_model.config.hidden_size, output_dim=mistral_model.config.hidden_size)

    # model = TimeSeriesMistral(mistral_model, time_series_projector)
    model = TimeSeriesMistral(mistral_model, time_series_projector, attn_implementation="eager")

    # Load configuration
    with open('config.json', 'r') as config_file:
        local_config = json.load(config_file)

    ts_path = local_config.get('ts_path')
    text_path = local_config.get('text_path')
    #ts_path = "/mnt/nfs/proj/hnl_downloaded_public_data/PFCTS/"
    #text_path = "/mnt/nfs/proj/hnl_downloaded_public_data/clip_description.csv"
    prediction_target = 'description' # 'description', 'valence', 'emotion'

    train_dataset = TimeSeriesDataset(ts_path, text_path, tokenizer, prediction_target=prediction_target)
    # train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    # train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=1,
        # gradient_checkpointing=True,
        gradient_accumulation_steps=8,
        save_steps=1000,
        save_total_limit=2,
        # fp16=True,  # Enable mixed precision training
        report_to="none",  # Disable wandb
    )

    # Define the Data Collator
    data_collator = DataCollatorWithPaddingAndTimeSeries(tokenizer=tokenizer, mlm=False)

    # Check for the latest checkpoint
    checkpoints = [os.path.join(training_args.output_dir, d) for d in os.listdir(training_args.output_dir) if d.startswith("checkpoint-")]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print(f"Resuming from checkpoint: {latest_checkpoint}")
    else:
        latest_checkpoint = None

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    # trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset)

    # if latest_checkpoint:
    #     trainer.train(resume_from_checkpoint=latest_checkpoint)
    # else:
    #     trainer.train()

    trainer.train()

    # Save the trained model
    trainer.save_model("/home/barbosa/rihome/projects/TSLLaMA2/models/mistral_ts_projector")

if __name__ == "__main__":
    main()
