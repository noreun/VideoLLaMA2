
# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.

import sys
sys.path.append('./')

from videollama2.train_timeseries import train

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")