from enum import Enum
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

class languages(Enum):
    English = "eng_Latn"
    Bengali = "ben_beng"
    Hindi = "hin_Deva"
    Tamil = "tam_Taml"
    Telugu = "tel_Telu"   

class NLLB():
    """Summary: this is a model class for model facebook/nllb. 
    `No Language Left Behind` is NMT Neural Machine Translation model for performing text to text translations.
    This class uses 600M parameter varient, However there's a 1.3B variant available too!
    """
    def __init__(self,):
        # model artefacts
        # Note: the model assume the input is in English by default.
        self.model_id = "facebook/nllb-200-distilled-600M"
        self.tokenizer = None
        self.model =  None
        self.tokenizer_lang = languages.English  #Default
        self.device = "cpu"
    
    def load_model(self, src_lang = languages.English):
        # load tokenizer checks for src_language conflict and reloads the  tokenizer if necessary.
        self.load_tokenizer(src_lang = src_lang)        
        if self.model is None:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id, token=True)
            self.model.to(self.device)
            
    def load_tokenizer(self, src_lang = languages.English):
        if self.tokenizer is None or self.tokenizer_lang!=src_lang:
            self.tokenizer =  AutoTokenizer.from_pretrained(self.model_id,src_lang = src_lang.value, token = True)            
    
    def unload_model(self):
        self.model.to('cpu')
        self.model = None    
    
    def translate(self, text:str, src_lang:str, tgt_lang:languages, unload_model_after_inference = False):
        # loads model only once.
        self.load_tokenizer(src_lang = src_lang)
        if self.model == None:
            self.load_model(src_lang = src_lang)
            
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        translated_tokens = self.model.generate(
            **inputs, forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(tgt_lang.value), max_length=30
        )
        outputs = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
        
        if unload_model_after_inference:
            self.unload_model()
                    
        return outputs
    

# if __name__ == "__main__":
#     text = "welcome to pavan-translate. This is a application to translate text into multiple languages."
#     print(f"Testing sample text: {text}\n")
    
#     nllb = NLLB()    
#     # language selection.
#     src_lang = languages.English
#     tgt_lang  = languages.Hindi
#     translations = nllb.translate(text= text, src_lang = src_lang, tgt_lang = tgt_lang)
    
    
#     # testing all the languages
#     for lang in languages:
#         translations = nllb.translate(text= text, src_lang = src_lang, tgt_lang = lang)
#         print(f"[{lang.name}]: {translations}")
    