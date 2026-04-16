import gradio as gr
import torch
from transformers import pipeline

import random

MODEL_NAME = "openai/whisper-large-v3-turbo"
BATCH_SIZE = 8

device = 0 if torch.cuda.is_available() else "cpu"

pipe = pipeline(
    task="automatic-speech-recognition",
    model=MODEL_NAME,
    chunk_length_s=30,
    device=device,
)


# @spaces.GPU
def transcribe(inputs):
    if inputs is None:
        raise gr.Error("No audio file submitted! Please upload or record an audio file before submitting your request.")

    text = pipe(inputs, batch_size=BATCH_SIZE, return_timestamps=True)["text"]
    return  [text, 'placeholder']

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
