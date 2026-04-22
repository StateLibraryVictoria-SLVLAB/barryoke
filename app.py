import gradio as gr
import torch

import random

from transformers import pipeline


BASE_MODEL_NAME = "openai/whisper-tiny"
LAB_MODEL_NAME = ''
BATCH_SIZE = 8
TASK = "automatic-speech-recognition"

device = 0 if torch.cuda.is_available() else "cpu"


pipe = pipeline(
    task=TASK,
    model=BASE_MODEL_NAME,
    chunk_length_s=30,
    device=device,
)


def transcribe(inputs, task=TASK):

    placeholder = 'Placeholder'
    try:
        if inputs is None:
            raise gr.Error("No audio file submitted! Please upload or record an audio file before submitting your request.")
        
        result = pipe(inputs, batch_size=BATCH_SIZE, generate_kwargs={"task": task, "language": "en"})
        
        return result['text'], placeholder
        
    except Exception as e:

        print(f"Error - {e}")

        return e, placeholder


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
        inp = gr.Audio(sources='microphone',type="filepath"),
        out_1 = gr.Textbox()
        out_2 = gr.Textbox()
    btn = gr.Button("Run")
    btn.click(fn=transcribe, inputs=inp, outputs=[out_1, out_2])

demo.launch()
