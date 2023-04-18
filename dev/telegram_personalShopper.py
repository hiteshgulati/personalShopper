import logging
import os
from dotenv import load_dotenv
load_dotenv()
import time

from langchain import PromptTemplate
from langchain.llms import OpenAI
import tiktoken

import time

import json

from collections import defaultdict

import uuid

import datetime

import io
session_details = defaultdict(lambda: False)

sleep_time = 1
token_limit_per_session = 1_000_000
#OpenAI LLM
llm = OpenAI(openai_api_key=os.environ['OPENAI_API_KEY'])

CHATTING,STOP = range(2)

#Data
categories = ['mobile', 'grocery', 'fashion']
products = ['iPhone 14 Pro', 'Samsung Galaxy S23 Ultra', 'Google Pixel 7']
product_specs = [{'product name': 'Apple iPhone 14 Pro',
                    'NETWORK Technology': 'GSM / CDMA / HSPA / EVDO / LTE / 5G',
                    'SIM Type': 'Dual SIM',
                    'LAUNCHED': '07 Sept 2022',
                    'BODY Dimensions': '147.5 x 71.5 x 7.9 mm',
                    'BODY Weight': '206 g',
                    'BODY Rating': 'IP68 dust/water resistant',
                    'DISPLAY Type': 'LTPO Super Retina XDR OLED',
                    'DISPLAY Refresh Rate': '120Hz, more is better',
                    'DISPLAY Size': '6.1 inches',
                    'PLATFORM OS': 'iOS 16',
                    'Chipset': 'Apple A16 Bionic',
                    'CAMERA Front': '1',
                    'BATTERY\tCapacity': '3200 mAh, more is better',
                    'BATTERY Life': '86 hours, more is better',
                    'WIRELESS Charging': 'True',
                    'COLORS Available': 'Space Black, Silver, Gold, Deep Purple',
                    'Price': '₹ 120000',
                    'Discount': 'Maximum 10%',
                    'Delivery Time': '5 days'},
                    {'product name': 'Samsung Galaxy S23 Ultra',
                    'NETWORK Technology': 'GSM / CDMA / HSPA / EVDO / LTE / 5G',
                    'SIM Type': 'Dual SIM',
                    'LAUNCHED': '03 Feb 2023',
                    'BODY Dimensions': '163.4 x 78.1 x 8.9 mm',
                    'BODY Weight': '234 g',
                    'BODY Rating': 'IP68 dust/water resistant',
                    'DISPLAY Type': 'Dynamic AMOLED 2X',
                    'DISPLAY Refresh Rate': '120Hz, more is better',
                    'DISPLAY Size': '6.8 inches',
                    'PLATFORM OS': 'Android 13',
                    'Chipset': 'Qualcomm Snapdragon 8 Gen 2 (4 nm)',
                    'CAMERA Front': '4',
                    'BATTERY\tCapacity': '5000 mAh, more is better',
                    'BATTERY Life': '126 hours, more is better',
                    'WIRELESS Charging': 'True',
                    'COLORS Available': 'Phantom Black, Green, Cream, Lavender, Graphite, Sky Blue, Lime, Red',
                    'Price': '₹ 100000',
                    'Discount': 'Maximum 5%',
                    'Delivery Time': '2 days'},
                    {'product name': 'Google Pixel 7 Pro',
                    'NETWORK Technology': 'GSM / CDMA / HSPA / EVDO / LTE / 5G',
                    'SIM Type': 'Dual SIM',
                    'LAUNCHED': '06 Oct 2022',
                    'BODY Dimensions': '162.9 x 76.6 x 8.9 mm',
                    'BODY Weight': '234 g',
                    'BODY Rating': 'IP68 dust/water resistant',
                    'DISPLAY Type': 'LTPO AMOLED',
                    'DISPLAY Refresh Rate': '120Hz, more is better',
                    'DISPLAY Size': '6.7 inches',
                    'PLATFORM OS': 'Android 13',
                    'Chipset': 'Google Tensor G2',
                    'CAMERA Front': '3',
                    'BATTERY\tCapacity': '5000 mAh, more is better',
                    'BATTERY Life': '83 hours, more is better',
                    'WIRELESS Charging': 'True',
                    'COLORS Available': 'Obsidian Black, Snow White, Hazel Green',
                    'Price': '₹ 80000',
                    'Discount': 'Maximum 5%',
                    'Delivery Time': '3 days'}]

details ="""1.	reviews 
2.	product description 
3.	comparison 
4.	delivery time 
5.	alternate products 
6.	purchase 
7.	recommendation """

shift_end_bye = """
Hello there! I'm sorry to inform you that my shift as a shopping assistant at Tata Neu store has ended for the day. However, I would like to thank you for taking the time to interact with me and I hope that I was able to provide you with the assistance you needed.

If you have any further questions or concerns, please do not hesitate to reach out to one of my fellow shopping assistants at Tata Neu store who will be more than happy to assist you. Alternatively, you can come back at a later time when I am available, and I will be happy to help you with your shopping needs.

Thank you again for shopping with us, and I hope to see you again soon!
"""

cancel_thank_you = """
Hello there! I appreciate you taking the time to visit Tata Neu store and for your interaction with me as your shopping assistant. Even though you are not interested in purchasing anything at this time, I want to thank you for considering us as your shopping destination.

If you have any further needs or questions, please don't hesitate to come back and visit us again. Our team of shopping assistants are always ready and available to assist you with any concerns or queries you may have. We are committed to providing exceptional customer service to all our customers, and we hope that you will give us another opportunity to serve you in the future.

Thank you again for your visit, and we look forward to seeing you again soon at Tata Neu store!
"""

moods = ['joking', 'formal', 'informal', 'fun', 'energetic']

def list_to_string(list,ordered=True,separator="\n"):
    s = ""
    for idx,list_item in enumerate(list):
        if ordered:
            s += f"{idx+1}.\t{list_item} \n"
        else:
            s += f"-.\t{list_item} \n"
    return s

def products_to_string(products_with_specs):
    output_string = ""
    for p in products_with_specs:
        product_string = ""
        for k,v in p.items():
            if k == "product name":
                product_string += v
            else:
                product_string += f"""\n\t{k} = {v}"""
        output_string += f"""{product_string}\n\n\n"""
    return output_string

def get_prompt(conversation_history):
    singular_prompt = f'''
    You are an AI based shopping assistant of a store named Tata Neu. 

    The store offers products in following categories:
    {list_to_string(categories)}

    Grocery and Fashion categories are currently under production. The store is currently not offering any products under Fashion and grocery category.
    Mobile Category have three mobile phones. The mobile and its specs are mentioned below:
    {products_to_string(product_specs)}


    Here's a conversation between you and customer:
    {conversation_history_text(conversation_history)}



    Please answer following question based on above conversation
    1. What is your response to customer query
    2. Restrict you answer to "Yes" or "No" only without any further explanation. Is the customer is ready to buy the product unconditionally? Answer yes only if confidence is above 90% else No.
    3. If the Answer to second question is "Yes", provide a reply to appreciate customer's choice by customer naming the product they want to buy. Also provide information about product price and product delivery time.


    Answer in following format:
    {{"response":answer to question 1,
    "purchase_ready": answer to question 2,
    "purchase_reply": answer to question 3
    }}


    Follow following rules before answering
    {list_to_string(answering_rules)}
    '''.replace("\t","")
    return singular_prompt


greet_template = f'''You are an AI based shopping assistant of a store named Tata Neu. 
The store offers products in following categories:
{list_to_string(categories)}

Greet the customer to the store and ask them how can you be helpful

Reply in a joyful way'''



#Rules
answering_rules = ['Restrict you suggestion to above mentioned products only.',
                   "Always have a definite answer even if the confidence is low",
                   "Answer in fun way",
                   "Always be closing",
                   ]

#Conversation
def add_message(conversation_history,message,sender):
    conversation_history.append(f"{sender}: {message}")

def conversation_history_text(conversation_history):
    s = ""
    for list_item in conversation_history:
        s += f"{list_item}\n"
    return s

def design_customer_reply_from_ai_reply(response_from_ai):
    response= json.loads(response_from_ai)
    designed_reply = []
    designed_reply.append(response['response'])
    if (response['purchase_ready']=="Yes"):
        designed_reply.append(response['purchase_reply'])
        designed_reply.append("Please use this link to finalize the order")
    return designed_reply



def conversations (conversation_history,customer_reply=None,ai_reply=None,ai_greet=None,print_inline=False):
    if ai_greet:
        add_message(conversation_history,message=ai_greet,sender="AI")
        if print_inline:
            print(conversation_history_text(conversation_history))
        return True
    if customer_reply:
        add_message(conversation_history,message=customer_reply,sender="Customer")
        if print_inline:
            print(get_prompt(conversation_history))
        return get_prompt(conversation_history)
    if ai_reply:
        designed_reply = design_customer_reply_from_ai_reply(ai_reply)
        for r in designed_reply:
            for line_r in r.split(". "):
                add_message(conversation_history,message=line_r+".",sender="AI")
        if print_inline:
            print(conversation_history_text(conversation_history))
        return designed_reply
    if len(conversation_history)==0:
        if print_inline:
            print(greet_template)
        return greet_template

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def chat_with_ai(conversation_history,token_count,message=None,print_details=False,total_tries=5):
    try_count = 0
    success = False
    while (try_count < total_tries)&(not success):
        try:
            if not message:
                llm_prompt = conversations(conversation_history)
                token_count += num_tokens_from_string(llm_prompt, "gpt2")
                llm_response = llm(llm_prompt)
                token_count += num_tokens_from_string(llm_response, "gpt2")
                conversations(conversation_history,ai_greet=llm_response)
                success = True
                return llm_response, token_count
            else:
                llm_prompt = conversations(conversation_history,customer_reply=message)
                token_count += num_tokens_from_string(llm_prompt, "gpt2")
                llm_response = llm(llm_prompt)
                token_count += num_tokens_from_string(llm_response, "gpt2")
                if print_details:
                    print(llm_response)
                response = conversations(conversation_history,ai_reply=llm_response)
                success = True
                return response, token_count
        except:
            pass

def save_session():
    file_path = os.path.join(os.getcwd(),'sessions',datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S")+".json")
    json_file_name = file_path
    with open (json_file_name,'w') as fp:
        pass
    with io.open(json_file_name,"w",encoding="utf8") as outfile:
        str_ = json.dumps(session_details,
                indent=4, sort_keys=True,
                separators=(',', ': '), ensure_ascii=False)
        outfile.write(str_)

from telegram import __version__ as TG_VER

try:
    from telegram import __version_info__
except ImportError:
    __version_info__ = (0, 0, 0, 0, 0)  # type: ignore[assignment]

if __version_info__ < (20, 0, 0, "alpha", 5):
    raise RuntimeError(
        f"This example is not compatible with your current PTB version {TG_VER}. To view the "
        f"{TG_VER} version of this example, "
        f"visit https://docs.python-telegram-bot.org/en/v{TG_VER}/examples.html"
    )
from telegram import ReplyKeyboardMarkup, ReplyKeyboardRemove, Update, constants
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters
)

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


from functools import wraps

def send_action(action):
    """Sends `action` while processing func command."""

    def decorator(func):
        @wraps(func)
        async def command_func(update, context, *args, **kwargs):
            await context.bot.send_chat_action(chat_id=update.effective_message.chat_id, action=action)
            time.sleep(2)
            return await func(update, context,  *args, **kwargs)
        return command_func
    
    return decorator

@send_action(constants.ChatAction.TYPING)
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Starts the conversation by greeting the customer."""
    user = update.message.from_user
    session_details[user.id] = {'user':user.id,
                                'user_name':user.first_name,
                                'conversation_history':[],
                                'token_count':0}
    response, token_count = chat_with_ai(
                                session_details[user.id]['conversation_history'],
                                session_details[user.id]['token_count'])
    session_details[user.id]['token_count'] += token_count
    for line in response.split(". "):
        time.sleep(sleep_time)
        await update.message.reply_text(line)
    save_session()

    return CHATTING

@send_action(constants.ChatAction.TYPING)
async def chatting(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Stores the selected gender and asks for a photo."""
    user = update.message.from_user

    response, token_count = chat_with_ai(
                                session_details[user.id]['conversation_history'],
                                session_details[user.id]['token_count'],
                                message=update.message.text)
    
    session_details[user.id]['token_count'] += token_count
    
    logger.info("Customer: %s \n AI: %s", update.message.text, response)
    print(response)
    for r in response:
        for line in r.split(". "):
            time.sleep(sleep_time)
            await update.message.reply_text(line)
    save_session()
    if session_details[user.id]['token_count'] < token_limit_per_session:
        return CHATTING
    else:
        for line in shift_end_bye.split(". "):
            time.sleep(sleep_time)
            await update.message.reply_text(line)
        return ConversationHandler.END



@send_action(constants.ChatAction.TYPING)
async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Cancels and ends the conversation."""
    user = update.message.from_user
    logger.info("User %s canceled the conversation.", user.first_name)
    for line in shift_end_bye.split(". "):
        time.sleep(sleep_time)
        await update.message.reply_text(line)

    return ConversationHandler.END

def main() -> None:
    """Run the bot."""
    # Create the Application and pass it your bot's token.
    application = Application.builder().token(os.environ['TELEGRAM_TOKEN']).build()

    # Add conversation handler with the states GENDER, PHOTO, LOCATION and BIO
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            CHATTING: [MessageHandler(filters.TEXT & ~filters.COMMAND, chatting)]
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )

    application.add_handler(conv_handler)

    # Run the bot until the user presses Ctrl-C
    application.run_polling()
    


if __name__ == "__main__":
    main()