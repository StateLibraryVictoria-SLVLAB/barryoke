import gradio as gr
import torch

import numpy as np

import random

from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

processor = AutoProcessor.from_pretrained("openai/whisper-tiny")
model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-tiny")


device = 0 if torch.cuda.is_available() else "cpu"


def convert_audio(audio_input):

    sample_rate, sample_array = audio_input
    if sample_array.ndim > 1:
        sample_array = sample_array.mean(axis=1)
        
    sample_array = sample_array.astype(np.float32)
    sample_array /= np.max(np.abs(sample_array))

    return sample_rate,  sample_array

def transcribe(audio_input):

    try:

        model.config.forced_decoder_ids = None

        sample_rate, sample_array = convert_audio(audio_input)

        input_features = processor(sample_array, sampling_rate=sample_rate, return_tensors="pt").input_features

        predicted_ids = model.generate(input_features)
    # transcription = processor.batch_decode(predicted_ids, skip_special_tokens=False)

        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

    except Exception as e:
        gr.Error(f"An error occurred: {e}")

    return transcription


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
