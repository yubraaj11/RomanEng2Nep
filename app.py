import gradio as gr
from transformers import AutoTokenizer, MT5ForConditionalGeneration

# Load tokenizer and model
checkpoint = "syubraj/RomanEng2Nep-v2"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = MT5ForConditionalGeneration.from_pretrained(checkpoint)

# Set max sequence length
max_seq_len = 20

# Define the translation function
def translate(text):
    # Tokenize the input text with a max length of 20
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_seq_len)

    # Generate translation
    translated = model.generate(**inputs)

    # Decode the translated tokens back to text
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    return translated_text

# Gradio interface
iface = gr.Interface(
    fn=translate,  # function to use for inference
    inputs="text",  # input type
    outputs="text",  # output type
    title="Romanized English to Nepali Transliterator",
    description="Translate Romanized English text into Nepali.",
    examples=[["ahile", "prakriti"]]
)

# Launch the Gradio app
iface.launch()
