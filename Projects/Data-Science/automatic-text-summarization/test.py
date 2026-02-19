# from transformers import pipeline

# summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# text = """
# Artificial Intelligence (AI) is rapidly transforming various sectors including healthcare, education, finance, and transportation. 
# With the ability to process massive datasets, identify patterns, and automate decision-making, AI is streamlining operations 
# and improving efficiency. In healthcare, AI assists in diagnosing diseases and personalizing treatments. In finance, it's used 
# to detect fraud and make investment decisions. As AI continues to evolve, it’s essential to address ethical concerns, including 
# bias, privacy, and job displacement.
# """

# summary = summarizer(text, max_length=60, min_length=25, do_sample=False)

# #Output result
# print("🧠 Original Text:\n")
# print(text)
# print("\n✅ Abstractive Summary:\n")
# print(summary[0]['summary_text'])

from transformers import BartForConditionalGeneration, BartTokenizer, pipeline

model_name = "facebook/bart-large-cnn"

print("⏳ Cargando modelo y tokenizador...")
model = BartForConditionalGeneration.from_pretrained(model_name)
tokenizer = BartTokenizer.from_pretrained(model_name)

# Intentamos usar el pipeline, si falla, usamos la lógica manual
try:
    summarizer_pipe = pipeline("summarization", model=model, tokenizer=tokenizer)
    use_manual = False
except Exception:
    print("⚠️ Modo manual activado.")
    use_manual = True

text = """
Artificial Intelligence (AI) is rapidly transforming various sectors including healthcare, education, finance, and transportation. 
With the ability to process massive datasets, identify patterns, and automate decision-making, AI is streamlining operations 
and improving efficiency. In healthcare, AI assists in diagnosing diseases and personalizing treatments. In finance, it's used 
to detect fraud and make investment decisions.
"""

print("✨ Generando resumen...")

if not use_manual:
    # Caso 1: Usando Pipeline
    result = summarizer_pipe(text, max_length=60, min_length=25, do_sample=False)
    summary_text = result[0]['summary_text']
else:
    # Caso 2: Lógica Manual corregida
    inputs = tokenizer([text], max_length=1024, return_tensors="pt", truncation=True)
    summary_ids = model.generate(inputs["input_ids"], num_beams=4, max_length=60, min_length=25, early_stopping=True)
    # Decodificamos el primer resultado de la lista
    summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print("\n✅ Resultado Final:")
print("-" * 30)
print(summary_text)
print("-" * 30)