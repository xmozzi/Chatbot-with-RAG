import PyPDF2

def load_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
    chunks = text.split(". ")
    return chunks