# %%
from transformers import XLMRobertaTokenizer, XLMRobertaForMaskedLM
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import BertForMaskedLM, BertTokenizer
from transformers import pipeline
from normalizer import normalize
import torch
import string


# বাংলাবার্ট (বুয়েট)
bb_path = "csebuetnlp/banglabert_generator"

bb_model = AutoModelForMaskedLM.from_pretrained(bb_path)
bb_tokenizer = AutoTokenizer.from_pretrained(bb_path)

bb_fill_mask = pipeline(
    "fill-mask",
    model=bb_model,
    tokenizer=bb_tokenizer
)

# বাংলাবার্ট (সাগর)
bb2_model = BertForMaskedLM.from_pretrained("sagorsarker/bangla-bert-base")
bb2_tokenizer = BertTokenizer.from_pretrained("sagorsarker/bangla-bert-base")
bb2_fill_mask = pipeline(
    "fill-mask",
    model=bb2_model,
    tokenizer=bb2_tokenizer
)

# এক্সএলএম রোবের্টা বেইস
xlmroberta_tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
xlmroberta_model = XLMRobertaForMaskedLM.from_pretrained(
    'xlm-roberta-base').eval()

# আমাদের মডেল
wiki_model = AutoModelForMaskedLM.from_pretrained(
    'f00d/Multilingual-MiniLM-L12-H384-MLM-finetuned-wikipedia_bn')
# wiki_model = AutoModelForMaskedLM.from_pretrained(
#     '../Models/Multilingual-MiniLM-L12-H384-MLM-finetuned-wikipedia_bn')
wiki_fill_mask = pipeline(
    "fill-mask",
    model=wiki_model,
    tokenizer=xlmroberta_tokenizer
)

# মিনিএলএম
minilm_model = AutoModelForMaskedLM.from_pretrained(
    "microsoft/Multilingual-MiniLM-L12-H384")
minilm_fill_mask = pipeline(
    "fill-mask",
    model=minilm_model,
    tokenizer=xlmroberta_tokenizer
)

# বার্ট (বহুভাষিক)
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
bert_model = BertForMaskedLM.from_pretrained("bert-base-multilingual-cased")
bert_fill_mask = pipeline(
    "fill-mask",
    model=bert_model,
    tokenizer=bert_tokenizer
)


top_k = 5


def decode(tokenizer, pred_idx, top_clean):
    ignore_tokens = string.punctuation + '[PAD]'
    tokens = []
    for w in pred_idx:
        token = ''.join(tokenizer.decode(w).split())
        if token not in ignore_tokens:
            tokens.append(token.replace('##', ''))
    return '\n'.join(tokens[:top_clean])


def encode(tokenizer, text_sentence, add_special_tokens=True):
    text_sentence = text_sentence.replace('<mask>', tokenizer.mask_token)

    # শেষে ফুলস্টপ যোগ করা হয়েছে
    if tokenizer.mask_token == text_sentence.split()[-1]:
        text_sentence += ' .'

    input_ids = torch.tensor(
        [tokenizer.encode(text_sentence, add_special_tokens=add_special_tokens)])
    mask_idx = torch.where(input_ids == tokenizer.mask_token_id)[1].tolist()[0]
    return input_ids, mask_idx


def get_all_predictions(text_sentence, top_clean=5):

    # ডিবাগিং এর জন্য
    print(text_sentence)

    def get_sentence(model_name, sentence):
        sentence = text_sentence.split('<mask>')

        switcher = {
            'bb': f"{bb_fill_mask.tokenizer.mask_token}",
            'our': f"{wiki_fill_mask.tokenizer.mask_token}",
            'minilm': f"{minilm_fill_mask.tokenizer.mask_token}",
        }

        mask_token = switcher.get(model_name)

        if sentence[1] == '':
            # পরবর্তি শব্দের প্রস্তাবের জন্য
            sentence = f'{sentence[0]} {mask_token}'
        else:
            # মুখোশ পূরণের জন্য
            sentence = f'{sentence[0]} {mask_token} {sentence[1]}'

        return sentence

    # ========================= বাংলাবার্ট (সাগর) =================================
    input_ids, mask_idx = encode(bb2_tokenizer, text_sentence)
    with torch.no_grad():
        predict = bb2_model(input_ids)[0]
    bb2 = decode(bb2_tokenizer, predict[0, mask_idx, :].topk(
        top_k).indices.tolist(), top_clean)

    # ========================= বার্ট (বহুভাষিক) =================================
    input_ids, mask_idx = encode(bert_tokenizer, text_sentence)
    with torch.no_grad():
        predict = bert_model(input_ids)[0]
    bert = decode(bert_tokenizer, predict[0, mask_idx, :].topk(
        top_k).indices.tolist(), top_clean)

    # ========================= এক্সএলএম রোবের্টা বেইস =================================
    input_ids, mask_idx = encode(
        xlmroberta_tokenizer, text_sentence, add_special_tokens=True)
    with torch.no_grad():
        predict = xlmroberta_model(input_ids)[0]
    xlm = decode(xlmroberta_tokenizer, predict[0, mask_idx, :].topk(
        top_k).indices.tolist(), top_clean)

    # ========================= আমাদের মডেল =================================
    sentence = get_sentence('our', text_sentence)

    res = wiki_fill_mask(
        normalize(sentence)
    )

    our = []
    for dict in res:
        our.append(dict['token_str'])
    our = '\n'.join(our)

    # =========================  বাংলাবার্ট (বুয়েট) =================================
    sentence = get_sentence('bb', text_sentence)

    res = bb_fill_mask(
        normalize(sentence)
    )
    bb = []
    for dict in res:
        bb.append(dict['token_str'])
    bb = '\n'.join(bb)

    # ========================= মিনিএলএম =================================
    sentence = get_sentence('minilm', text_sentence)

    res = minilm_fill_mask(
        normalize(sentence)
    )

    minilm = []
    for dict in res:
        minilm.append(dict['token_str'])
    minilm = '\n'.join(minilm)

    return {'our': our,
            'bb': bb,
            'xlm': xlm,
            'minilm': minilm,
            'bb2': bb2,
            'bert': bert}
