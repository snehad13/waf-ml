from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.processors import RobertaProcessing
from transformers import PreTrainedTokenizerFast
import os

os.makedirs("models/tokenizer_hf", exist_ok=True)

# Load existing vocab + merges
tokenizer = Tokenizer(BPE.from_file("models/tokenizer/tokenizer-vocab.json",
                                    "models/tokenizer/tokenizer-merges.txt"))
tokenizer.pre_tokenizer = ByteLevel()
tokenizer.decoder = ByteLevelDecoder()
tokenizer.post_processor = RobertaProcessing(
    sep=("</s>", tokenizer.token_to_id("</s>") or 2),
    cls=("<s>", tokenizer.token_to_id("<s>") or 0)
)

# Save fast tokenizer as tokenizer.json
tokenizer.save("models/tokenizer_hf/tokenizer.json")

# Wrap in HuggingFace PreTrainedTokenizerFast
hf_tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="models/tokenizer_hf/tokenizer.json",
    unk_token="<unk>",
    pad_token="<pad>",
    cls_token="<s>",
    sep_token="</s>"
)

hf_tokenizer.save_pretrained("models/tokenizer_hf")
print("Tokenizer converted and saved to models/tokenizer_hf/")
