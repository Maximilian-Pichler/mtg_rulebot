import gradio
from pathlib import Path

from mtg.bot import MagicGPT
from mtg.data_handler import CardDB
from mtg.objects.llm_chain import LlmChain
from mtg.bot.chat import create_chat_model
from mtg.bot.chat_history import ChatHistory


all_cards_file = Path("data/raw/scryfall_all_cards_with_rulings.json")

def set_temp(input):
    print(input)
    return input

def get_bot(model: str="gpt-3.5-turbo",temp: float=0): 
    print(temp)
    card_db = CardDB(all_cards_file)
    chat_history = ChatHistory()
    llm_chain = create_chat_model(
        model=model, 
        temperature=temp, 
        max_token_limit=2000
    )

    magic_bot = MagicGPT(llm_chain=llm_chain, card_db=card_db, chat_history=chat_history)
    return magic_bot

chain = LlmChain()



# creates a new Blocks app and assigns it to the variable demo.
with gradio.Blocks(scale=2) as ui:
    # creates a new Chatbot instance and assigns it to the variable chatbot.
    temperature = 0.0
    chatbot = gradio.Chatbot(bubble_full_width=False, layout='panel')

    # creates a new Row component, which is a container for other components.
    with gradio.Row(scale=1):
        slider = gradio.Slider(minimum=0.0, maximum=1.0, label="Temperature", value=0.1, interactive=True, step=0.05)
    slider.change(chain.set_temperature, slider)
    with gradio.Row(scale=1):
        txt = gradio.Textbox(show_label=False, placeholder="Enter text and press enter")
    txt.submit(get_bot(temp=temperature).ask, txt, chatbot)
    txt.submit(None, None, txt, _js="() => {''}")

ui.launch()
 