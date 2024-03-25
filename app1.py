from flask import Flask, render_template, request
from transformers import BartForConditionalGeneration, BartTokenizer, RobertaForCausalLM, RobertaTokenizer
from googletrans import Translator
from transformers import AdamW
import torch
app = Flask(__name__)

def bart_summarize(text):
    model_name = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    inputs = tokenizer([text], max_length=1024, return_tensors="pt")

    summary_ids = model.generate(inputs["input_ids"], max_length=150, num_beams=4, length_penalty=2.0, early_stopping=True)

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def generate_story(seed_text):
    model_name = "roberta-base"
    model = RobertaForCausalLM.from_pretrained(model_name)
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    input_ids = tokenizer.encode(seed_text, return_tensors="pt")
    max_length = 200
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1)
    generated_story = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_story

def train_story_generation_model():
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaForCausalLM.from_pretrained("roberta-base")
    input_ids = tokenizer.encode("Once upon a time", return_tensors="pt")
    num_train_epochs = 5
    learning_rate = 2e-5
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()
    for epoch in range(num_train_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, labels=input_ids)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
    model.save_pretrained("story_generation_model")

def translate_to_indian_language(english_text, target_language):
    translator = Translator()
    translated = translator.translate(english_text, dest=target_language)
    return translated.text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    text = request.form['text']
    summary = bart_summarize(text)
    return render_template('index.html', summary=summary)
@app.route('/ar')
def ar():
    return render_template('ar.html')
@app.route('/generate_story', methods=['POST'])
def generate_story_endpoint():
    seed_text = request.form['seed_text']
    generated_story = generate_story(seed_text)
    return render_template('index.html', generated_story=generated_story)

@app.route('/train_model', methods=['GET'])
def train_model():
    train_story_generation_model()
    return "Training completed successfully!"

@app.route('/translate', methods=['POST'])
def translate():
    english_text = request.form['english_text']
    target_language = request.form['target_language']
    translated_text = translate_to_indian_language(english_text, target_language)
    return render_template('index.html', translated_text=translated_text)

if __name__ == '__main__':
    app.run(debug=True)
