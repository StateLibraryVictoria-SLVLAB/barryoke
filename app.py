import gradio as gr
import torch

import numpy as np

import random

from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline

processor = AutoProcessor.from_pretrained("openai/whisper-tiny")
model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-tiny")

MODEL_NAME = "openai/whisper-tiny"
BATCH_SIZE = 8

device = 0 if torch.cuda.is_available() else "cpu"


pipe = pipeline(
    task="automatic-speech-recognition",
    model=MODEL_NAME,
    chunk_length_s=30,
    device=device,
)

def chunks_to_srt(chunks):
    srt_format = ""
    for i, chunk in enumerate(chunks, 1):
        start_time, end_time = chunk['timestamp']
        start_time_hms = "{:02}:{:02}:{:02},{:03}".format(int(start_time // 3600), int((start_time % 3600) // 60), int(start_time % 60), int((start_time % 1) * 1000))
        end_time_hms = "{:02}:{:02}:{:02},{:03}".format(int(end_time // 3600), int((end_time % 3600) // 60), int(end_time % 60), int((end_time % 1) * 1000))
        srt_format += f"{i}\n{start_time_hms} --> {end_time_hms}\n{chunk['text']}\n\n"
    return srt_format

def transcribe(inputs, task, return_timestamps, language):

    print(f"Input type: {type(inputs)}")
    if inputs is None:
        raise gr.Error("No audio file submitted! Please upload or record an audio file before submitting your request.")
    
    # Map the language names to their corresponding codes
    language_codes = {"English": "en", "Korean": "ko", "Japanese": "ja"}
    language_code = language_codes.get(language, "en")  # Default to "en" if the language is not found
    result = pipe(inputs, batch_size=BATCH_SIZE, generate_kwargs={"task": task, "language": f"<|{language_code}|>"}, return_timestamps=return_timestamps)
    
    if return_timestamps:
        return chunks_to_srt(result['chunks'])
    else:
        return result['text']


# def convert_audio(audio_input):

#     sample_rate, sample_array = audio_input
#     if sample_array.ndim > 1:
#         sample_array = sample_array.mean(axis=1)
        
#     sample_array = sample_array.astype(np.float32)
#     sample_array /= np.max(np.abs(sample_array))

#     return sample_rate,  sample_array

# def transcribe(audio_input):
#     print(f"Type = {type(audio_input)}")
#     transcription = 'Default value'

#     try:

#         model.config.forced_decoder_ids = None

#         sample_rate, sample_array = convert_audio(audio_input)

#         input_features = processor(sample_array, sampling_rate=sample_rate, return_tensors="pt").input_features

#         predicted_ids = model.generate(input_features)
#     # transcription = processor.batch_decode(predicted_ids, skip_special_tokens=False)

#         transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

#     except Exception as e:
#         print(f"An error occurred: {e}")
#         gr.Error(f"An error occurred: {e}")

#     return transcription


def select_quote():

    barry_quotes = [
    'this is a quote',  
    'this is another one',  
    'one more',  
    'not a quote',  
    ]    

    rand_idx = random.randint(0,(len(barry_quotes) - 1))

    return barry_quotes[rand_idx]


with gr.Blocks() as demo:

    gr.Markdown("Barryoke")
    quote = gr.Textbox()

    rand_btn = gr.Button("Quote")
    rand_btn.click(fn=select_quote,inputs=[],outputs=quote)

    with gr.Row():
        inp = gr.Audio(sources='microphone',waveform_options='recording'),
        out_1 = gr.Textbox()
        out_2 = gr.Textbox()
    btn = gr.Button("Run")
    btn.click(fn=transcribe, inputs=inp, outputs=[out_1, out_2])

demo.launch()
