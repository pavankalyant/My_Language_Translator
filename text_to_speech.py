from transformers import VitsModel, AutoTokenizer
import torch


class mms_TTS():
    def __init__(self):
        self.current_language = None
        self.tokenizer = None
        self.model = None
        self.model_id_templete = "facebook/mms-tts-{lang_code}"
        self.mms_language_mappings = {
                      "Bengali": "ben",
                      "English":"eng",
                      "Hindi": "hin",
                      "Telugu": "tel",
                      "Tamil": "tam"}
        
        
        self.device = "cpu" if torch.cuda.is_available() else "cpu"
        
    def load_model(self, language):
        if language != self.current_language:
            # delete existing model 
            if self.model is not None:
                self.model.to('cpu')
                del self.model
                del self.tokenizer
            
            # dereference lang_code and get the model_id
            lang_code =  self.mms_language_mappings.get(language, None)
            if lang_code is None:
                raise Exception("Bad Language name!")
            self.current_language = language
            model_id = self.model_id_templete.format(lang_code = lang_code)
            
            # load the model and send it to gpu
            tokenizer = AutoTokenizer.from_pretrained(model_id)        
            model = VitsModel.from_pretrained(model_id)
            model.to(self.device)
            self.model = model
            self.tokenizer = tokenizer      
        
        return self.model, self.tokenizer   

    def generate_speech(self, text, language):
        model, tokenizer = self.load_model(language = language)
        inputs = tokenizer(text, return_tensors="pt")
        inputs.to(self.device)        
        with torch.no_grad():
            output = model(**inputs).waveform.cpu().numpy().flatten()            
            
        return output
    
# if __name__ == "__main__":
#     tts = mms_TTS()
#     text = "welcome to pavan-translate. This is a application to translate text into multiple languages." 
#     lang = "  "
#     tts.generate_speech(text = text , language = lang)
                