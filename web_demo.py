from transformers import AutoModel, AutoTokenizer
import gradio as gr
import mdtex2html
import json
import requests
import pandas as pd

tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True, cache_dir="/nfs/volume-76-1/cache")
model = AutoModel.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True, cache_dir="/nfs/volume-76-1/cache").half().cuda()
model = model.eval()


import torch
from transformers.models.bert import BertTokenizer, BertForSequenceClassification


tokenizer_toxic = BertTokenizer.from_pretrained('/nfs/volume-76-1/zhangjian/Documents/ChatGLM-6B/toxic', cache_dir='/nfs/volume-76-1/cache')
model_toxic = BertForSequenceClassification.from_pretrained('/nfs/volume-76-1/zhangjian/Documents/ChatGLM-6B/toxic', cache_dir='/nfs/volume-76-1/cache')
model_toxic.eval()

texts = ['你就是个傻逼！','黑人很多都好吃懒做，偷奸耍滑！','男女平等，黑人也很优秀。']



def check_toxic(texts):
    """
    """
    model_input = tokenizer_toxic(texts, return_tensors="pt", padding=True)
    model_output = model_toxic(**model_input, return_dict=False)
    prediction = torch.argmax(model_output[0].cpu(), dim=-1)
    
    return prediction.item()


"""Override Chatbot.postprocess"""

def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y


gr.Chatbot.postprocess = postprocess

df = pd.read_csv('/nfs/volume-76-1/zhangjian/Documents/ChatGLM-6B/指令数据集.csv')

def searchEbot(text):
    """
    """
    r = requests.post('http://10.186.4.164:8097/vector', data = json.dumps({'text': text}))
    result = json.loads(r.text)
    r = requests.post('http://10.186.4.164:8081/faiss/search', data = json.dumps({'vectors': result["data"]["vector"], 'k': 20}))
    result = json.loads(r.text)[0]

    history = []
    results = list()
    for item in result['neighbors']:

        if item['score'] <= 0.75:
            continue
        ds = df[df['id'] == item['id']]
        results.append(ds)
        if item['score'] >= 0.95:
            break
    if not len(results):
        return 0, '', ''
    dd = pd.concat(results)
    dd.drop_duplicates(subset='output', inplace=True)

    fitext = ''
    for idx, row in dd.iterrows():
        fitext += row['output']
        history.append((row['instruction'], row['output']))
    
    return len(results), fitext, history



def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>"+line
    text = "".join(lines)
    return text


def predict(input, chatbot, max_length, top_p, temperature, history, useEbot, toxicCheck):
    chatbot.append((parse_text(input), ""))
    model_input = input
    toxicnumber = 0
    search_history = history
    if useEbot:
        search_num, search_text, search_history = searchEbot(input)
        model_input = "你是人工智能助手小滴, 请按照以下内容: " + search_text + '\n' + "回答该问题: " + input + '\n'
    if toxicCheck:
        toxicnumber = check_toxic(input)
    
    print(history)
    print(model_input)
    for response, history in model.stream_chat(tokenizer, model_input, history, max_length=max_length, top_p=top_p,
                                               temperature=temperature):
        if useEbot:
            if search_num == 0:
                response = response
        if toxicCheck and toxicnumber:
            response = '请礼貌性地提问问题哦，要开心哦😄'
        chatbot[-1] = (parse_text(input), parse_text(response))       

        yield chatbot, history


def reset_user_input():
    return gr.update(value='')


def reset_state():
    return [], []


with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center">ChatGLM</h1>""")

    chatbot = gr.Chatbot()
    with gr.Row():
        with gr.Column(scale=4):
            with gr.Column(scale=12):
                user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=10).style(
                    container=False)
            with gr.Column(min_width=32, scale=1):
                submitBtn = gr.Button("Submit", variant="primary")
        with gr.Column(scale=1):
            emptyBtn = gr.Button("Clear History")
            max_length = gr.Slider(0, 4096, value=2048, step=1.0, label="Maximum length", interactive=True)
            top_p = gr.Slider(0, 1, value=0.7, step=0.01, label="Top P", interactive=True)
            temperature = gr.Slider(0, 1, value=0.95, step=0.01, label="Temperature", interactive=True)
            useEbot = gr.Slider(0, 1, value=1, step=1, label="是否用Ebot知识库", interactive=True)
            toxicCheck = gr.Slider(0, 1, value=1, step=1, label="有害性检测", interactive=True)

    history = gr.State([])

    submitBtn.click(predict, [user_input, chatbot, max_length, top_p, temperature, history, useEbot, toxicCheck], [chatbot, history],
                    show_progress=True)
    submitBtn.click(reset_user_input, [], [user_input])

    emptyBtn.click(reset_state, outputs=[chatbot, history], show_progress=True)

demo.queue().launch(share=False, inbrowser=True, server_name='0.0.0.0', server_port=8098)
