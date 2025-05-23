from PyPDF2 import PdfReader

def load_pdf(file):
    reader = PdfReader(file)
    texts = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            texts.append(text)
    return texts
