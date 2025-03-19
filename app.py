from flask import Flask, render_template, request, jsonify, send_file
from googletrans import Translator
from text_to_text import NLLB, languages
from text_to_speech import mms_TTS
import io
import soundfile as sf
import numpy as np

app = Flask(__name__)
translator = Translator()

languages_type = [lang.name for lang in languages]

nllb = NLLB()  
tts = mms_TTS()


def resolve_languge(lang: str):
    for language in languages:
        if language.name == lang:
            return language
    return languages.English

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html",languages=languages_type)

@app.route("/translate", methods=["POST"])
def translate_text():
    source_lang = request.json.get("source")
    target_lang = request.json.get("target")
    text = request.json.get("text")
    source_lang = resolve_languge(source_lang)
    target_lang = resolve_languge(target_lang)  # language.English
    print(text,source_lang,target_lang) 

    translations = nllb.translate(text= text, src_lang = source_lang, tgt_lang = target_lang)
    print(translations,"............ananth...............")
    return jsonify({"translation": translations})

@app.route("/audio", methods=["POST"])
def audio():
    text = request.json.get("text")
    lang = request.json.get("language")
    print(text,lang,"....audio.....")
    # audio_data = tts.generate_speech(text=text, lang=lang)
    audio_data = tts.generate_speech(text=text, language=lang)
    print(type(audio_data))  # Should be <class 'bytes'> if correct
    print(len(audio_data))   # Should be greater than 0

    print(audio_data,".....audio_data................")
    if not isinstance(audio_data, np.ndarray):
        return jsonify({"error": "Invalid audio data"}), 500

    print(f"Audio shape: {audio_data.shape}, dtype: {audio_data.dtype}")

    # Convert NumPy array to WAV format
    audio_buffer = io.BytesIO()
    sf.write(audio_buffer, audio_data, samplerate=22050, format="WAV")  # Use an appropriate sample rate
    audio_buffer.seek(0)  # Reset buffer position

    return send_file(audio_buffer, mimetype="audio/wav")

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=2000,debug=True)


# app.route("/translate", methods=["POST"])
# def translate_text():
#     data = request.json
#     source_lang = data.get("source")
#     target_lang = data.get("target")
#     text = data.get("text")
#     source_lang = resolve_languge(source_lang)
#     target_lang = resolve_languge(target_lang)
#     print(text,source_lang,target_lang) 

#     if not text.strip():
#         return jsonify({"translation": ""})

#     try:
#         translated = translator.translate(text, src=source_lang, dest=target_lang)
#         print(translated.text)
#         return jsonify({"translation": translated.text})
#     except Exception as e:
#         return jsonify({"error": str(e), "translation": "Error in translation"})