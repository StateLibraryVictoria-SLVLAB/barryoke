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

    text1 = text1.casefold()
    text2 = text2.casefold()

    return [
        (token[2:], token[0] if token[0] != " " else None)
        for token in d.compare(text1, text2)
    ]


def transcribe(inputs, task='transcribe'):

    print(f"selected_quote: {selected_quote}")
    
    try:
        if inputs is None:
            raise gr.Error("No audio file submitted! Please upload or record an audio file before submitting your request.")
        
        base_result = pipe(inputs, batch_size=BATCH_SIZE, generate_kwargs={"task": task, "language": "en"})
        
        ft_result = ft_pipe(inputs, batch_size=BATCH_SIZE, generate_kwargs={"task": task, "language": "en"})

        base_diff_txt = diff_texts(selected_quote,base_result['text'])

        ft_diff_text = diff_texts(selected_quote,ft_result['text'])

        return base_result, base_diff_txt, ft_result, ft_diff_text
        
    except Exception as e:

        print(f"Error - {e}")

        return e, ''


def select_quote():

    global selected_quote

    barry_quotes = [
    'The hours of labor reduced to eight, leave to artisans, tradesmen, and other dwellers in towns a vary large portion of the remainder of the twenty-four virtually unoccupied.',  
    'How is this leisure to be disposed of? In the public-house? the singing hall? the dancing-saloon? which hold out seductions somewhat more dangerous, methinks, to honest labor than those presented by a library;',  
    'We may well rejoice, then, when we see a room such as this filled with attentive and reflective readers.',  
    'The insinuation of the waste of time in the perusal of unprofitable, trashy books must be met also by the enquiry — What does the expression mean?',
    "Men's minds are not cast in one mould — what charms one may repel another — nor is one man's mind at all times in the same frame."  
    ]    

    rand_idx = random.randint(0,(len(barry_quotes) - 1))

    selected_quote = barry_quotes[rand_idx]

    return f'## *"{barry_quotes[rand_idx]}"*'

descriptive_markdown = """
# Barryoke

Quotes taken from an address given by Sir Redmond Barry on the opening of the free public library of Ballarat East.

See the full address here - [https://latrobejournal.slv.vic.gov.au/latrobejournal/issue/latrobe-26/t1-g-t3.html](https://latrobejournal.slv.vic.gov.au/latrobejournal/issue/latrobe-26/t1-g-t3.html)
"""

with gr.Blocks() as demo:

    selected_quote = gr.State([])

    gr.Markdown(descriptive_markdown)

    rand_btn = gr.Button("Pick a quote")
    quote = gr.Markdown()

    rand_btn.click(fn=select_quote,inputs=[],outputs=quote)
    
    # with gr.Row():
    mf_input = gr.Audio(sources='microphone',type="filepath"),

    btn = gr.Button("Transcribe")
    
    gr.Markdown(f'### Output transcribed using {BASE_MODEL_NAME}')
    with gr.Row():

        out_1 = gr.Textbox(label='Whisper transcription')
        diff_out_1 = gr.HighlightedText(
            label="WhisperDiff",
            combine_adjacent=True,
            show_legend=True,
            color_map={"+": "green", "-": "blue"}
            )
        
    gr.Markdown(f'### Output transcribed using {FT_MODEL_NAME}')
    with gr.Row():    
        out_2 = gr.Textbox(label='Finetune transcription')
        diff_out_2 = gr.HighlightedText(
            label="FinetuneDiff",
            combine_adjacent=True,
            show_legend=True,
            color_map={"+": "green", "-": "blue"}
            )
    
    btn.click(fn=transcribe, inputs=mf_input, outputs=[out_1, diff_out_1, out_2, diff_out_2])

demo.launch()
