import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PreTrainedModel

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
