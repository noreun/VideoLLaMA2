import os
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, random_split

# Import from typing module
from typing import List, Optional, Tuple, Union

# You will need to install these packages
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

# Assuming these are from your video-llama2 project
from videollama2 import conversation as conversation_lib
from videollama2.constants import IGNORE_INDEX
from videollama2.train import (
    TrainingArguments,
    DataCollatorForSupervisedDataset,
    LazySupervisedDataset,
    preprocess,
    _mask_targets,
    _add_speaker_and_signal,
)
from videollama2.videollama2_trainer import VideoLLaMA2Trainer


class TimeSeriesDataset(Dataset):
    """Dataset for time series data and associated text from a CSV."""

    def __init__(self, ts_path, text_path, tokenizer, data_args, prediction_target='description'):
        super().__init__()
        self.tokenizer = tokenizer
        self.ts_path = ts_path
        self.text_path = text_path
        self.data_args = data_args
        self.prediction_target = prediction_target  # 'description', 'valence', or 'emotion'

        # Load CSV file
        self.data_df = pd.read_csv(text_path)

        # Get list of time series files
        self.ts_files = [f for f in os.listdir(ts_path) if f.endswith('_ts.pkl')]

    def __len__(self):
        return len(self.ts_files)

    def __getitem__(self, idx):
        # Load time series data
        ts_file = self.ts_files[idx]
        ts_file_path = os.path.join(self.ts_path, ts_file)
        with open(ts_file_path, 'rb') as f:
            time_series_data, _ = pickle.load(f)  # Ignore 'pixels'

        # Transpose time series to (time, features) format
        time_series_data = torch.tensor(time_series_data.T, dtype=torch.float)
        
        # For now, instead of segmenting the time series, we'll just truncate it
        time_series_data = time_series_data[:2000, :]

        # # Time Series Segmentation
        # time_steps = time_series_data.shape[0] 
        # patch_size = 150  # Example starting patch size
        # num_patches = time_steps // patch_size 

        # # Adjust patch size if needed to get approximately 16 patches
        # while num_patches < 16 and patch_size > 1:  
        #     patch_size -= 1
        #     num_patches = time_steps // patch_size

        # # Create patches and embed them
        # patches = [time_series_data[i * patch_size:(i + 1) * patch_size, :] 
        #            for i in range(num_patches)] 

        # # Pad the last patch if necessary
        # if time_steps % patch_size != 0:
        #     padding = torch.zeros((patch_size - (time_steps % patch_size), time_series_data.shape[1]), 
        #                             dtype=time_series_data.dtype)
        #     patches[-1] = torch.cat((patches[-1], padding), dim=0) 

        # # Simple patch embedding layer (replace with more complex embedding if needed)
        # patch_embedding = nn.Linear(patch_size * time_series_data.shape[1], self.data_args.time_series_embedding_dim) 
        # embedded_patches = torch.stack([patch_embedding(patch.flatten()) for patch in patches]) 

        # Get record ID from file name
        record_id = ts_file.split('_')[0]

        # Find corresponding row in the CSV
        row = self.data_df[self.data_df['video_names'] == f"{record_id}.mp4"].iloc[0]

        # Select the target for prediction
        if self.prediction_target == 'description':
            target_text = row['descriptions']
        elif self.prediction_target == 'valence':
            target_text = str(row['overall'])  # Convert valence to string
        elif self.prediction_target == 'emotion':
            target_text = row['one_word']
        else:
            raise ValueError(f"Invalid prediction_target: {self.prediction_target}")

        # Add the special token for time series (only for 'description' target)
        if self.prediction_target == 'description':
            target_text = '<timeseries>\n' + target_text

        # Process the text data
        source = [{"from": "human", "value": target_text}]
        data_dict = preprocess([source], self.tokenizer)

        # Add the time series data to the dictionary
        data_dict['time_series'] = time_series_data

        return data_dict

class TimeSeriesProjector(nn.Module):
    """Projector for time series data."""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.layers(x)


class TimeSeriesLLaMAForCausalLM(AutoModelForCausalLM):
    """LLaMA model adapted for time series data."""

    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.time_series_projector = TimeSeriesProjector(
            input_dim=config.time_series_dim, 
            hidden_dim=config.hidden_size, 
            output_dim=config.hidden_size
        )

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        time_series: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_timeseries(
                input_ids,
                attention_mask,
                past_key_values,
                labels,
                time_series
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        time_series: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if time_series is not None:
            (
                input_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_timeseries(
                input_ids=inputs,
                attention_mask=attention_mask,
                past_key_values=None,
                labels=None,
                time_series=time_series
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        time_series = kwargs.pop("time_series", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if time_series is not None:
            _inputs['time_series'] = time_series
        return _inputs

    def prepare_inputs_labels_for_timeseries(
        self, input_ids, attention_mask, past_key_values, labels, time_series
    ):
        if time_series is None:
            return input_ids, attention_mask, past_key_values, None, labels

        # Project time series data to the same dimension as the LLM embeddings
        time_series_features = self.time_series_projector(time_series)

        # Insert time series features into the input embeddings
        new_input_embeds = []
        for batch_idx, cur_input_ids in enumerate(input_ids):
            # Assumes the time series token is the first token
            time_series_start = 1
            cur_new_input_embeds = []
            cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:time_series_start]))
            cur_new_input_embeds.append(time_series_features[batch_idx])
            cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[time_series_start:]))
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            new_input_embeds.append(cur_new_input_embeds)

        # Pad the input embeddings
        max_len = max(x.shape[0] for x in new_input_embeds)
        new_input_embeds_aligned = []
        for cur_new_embed in new_input_embeds:
            padding = torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]), 
                                    dtype=cur_new_embed.dtype, device=cur_new_embed.device)
            cur_new_embed = torch.cat((cur_new_embed, padding), dim=0)
            new_input_embeds_aligned.append(cur_new_embed)
        new_input_embeds = torch.stack(new_input_embeds_aligned, dim=0)

        # Adjust the attention mask and labels accordingly
        if attention_mask is not None:
            new_attn_mask_pad_left = torch.full((attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), 
                                                True, dtype=attention_mask.dtype, device=attention_mask.device)
            attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)

        if labels is not None:
            new_labels = torch.cat((torch.full((labels.shape[0], time_series_features.shape[1]), IGNORE_INDEX, 
                                                device=labels.device, dtype=labels.dtype), labels), dim=1)

        return None, attention_mask, past_key_values, new_input_embeds, new_labels
