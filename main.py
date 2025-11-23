from dotenv import load_dotenv
from textExtractor import TextExtractor

load_dotenv()

async def main():
    model_name = "gemini-2.5-flash-lite"
    extractor = TextExtractor(model_name)
    pdf_path = "Resume.pdf"  # Replace with your PDF path
    extracted_text = await extractor.extractFromPdf(pdf_path)
    with open("resumeText.txt", "w", encoding="utf-8") as f:
        f.write(extracted_text)

if __name__ == "__main__":
    main()