import gradio as gr
import torch

import random
from difflib import Differ

from transformers import pipeline


BASE_MODEL_NAME = "openai/whisper-tiny"
FT_MODEL_NAME = "sotirios-slv/whisper-tiny-au-en"
BATCH_SIZE = 8
TASK = "automatic-speech-recognition"

device = 0 if torch.cuda.is_available() else "cpu"

pipe = pipeline(
    task=TASK,
    model=BASE_MODEL_NAME,
    chunk_length_s=30,
    device=device,
)

ft_pipe = pipeline(
    task=TASK,
    model=FT_MODEL_NAME,
    chunk_length_s=30,
    device=device,
)

def diff_texts(text1, text2):
    d = Differ()
    return [
        (token[2:], token[0] if token[0] != " " else None)
        for token in d.compare(text1, text2)
    ]


def transcribe(inputs, selected_quote, task='transcribe'):

    print(f"selected_quote: {selected_quote}")
    
    try:
        if inputs is None:
            raise gr.Error("No audio file submitted! Please upload or record an audio file before submitting your request.")
        
        base_result = pipe(inputs, batch_size=BATCH_SIZE, generate_kwargs={"task": task, "language": "en"})
        
        ft_result = ft_pipe(inputs, batch_size=BATCH_SIZE, generate_kwargs={"task": task, "language": "en"})

        base_diff_txt = diff_texts('base line',base_result['text'])

        ft_diff_text = diff_texts('fine tune',ft_result['text'])

        return base_diff_txt, ft_diff_text
        
    except Exception as e:

        print(f"Error - {e}")

        return e, ''


def select_quote():

    

    barry_quotes = [
    'this is a quote',  
    'this is another one',  
    'one more',  
    'not a quote',  
    ]    

    rand_idx = random.randint(0,(len(barry_quotes) - 1))

    selected_quote = barry_quotes[rand_idx]

    return barry_quotes[rand_idx]


with gr.Blocks() as demo:

    selected_quote = gr.State([])

    gr.Markdown("Barryoke")
    quote = gr.Textbox()

    rand_btn = gr.Button("Quote")
    rand_btn.click(fn=select_quote,inputs=[],outputs=quote)

    with gr.Row():
        mf_input = gr.Audio(sources='microphone',type="filepath"),
        
        out_1 = gr.HighlightedText(
            label="WhisperDiff",
            combine_adjacent=True,
            show_legend=True,
            color_map={"+": "green", "-": "blue"}
            )
        
        out_2 = gr.HighlightedText(
            label="FinetuneDiff",
            combine_adjacent=True,
            show_legend=True,
            color_map={"+": "green", "-": "blue"}
            )
    btn = gr.Button("Run")
    btn.click(fn=transcribe, inputs=[mf_input,selected_quote], outputs=[out_1, out_2])

demo.launch()
