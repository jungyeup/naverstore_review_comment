import os
from docx import Document
import pandas as pd
from datetime import datetime
import openai

class ReportGenerator:
    def __init__(self, api_key, folder_name='history'):
        self.folder_name = folder_name
        if not os.path.exists(self.folder_name):
            os.makedirs(self.folder_name)

        openai.api_key = api_key

    def generate_gpt4_summary(self, interaction_data):
        """
        Generate a summary using GPT-4 for a given interaction.
        """
        try:
            prompt = f"""
            Please provide a detailed description based on the following interaction data:
            {interaction_data}
            """
            response = openai.Completion.create(
                engine="gpt-4",
                prompt=prompt,
                max_tokens=500,
                n=1,
                stop=None,
                temperature=0.7
            )
            return response.choices[0].text.strip()
        except Exception as e:
            return f"Error generating summary: {e}"

    def generate_docx_report(self, data, file_path):
        doc = Document()

        doc.add_heading('Daily Report', 0)

        for record in data:
            doc.add_heading(record.get('type', 'Unknown'), level=1)
            doc.add_paragraph(f"Timestamp: {record.get('timestamp', 'N/A')}")
            doc.add_paragraph(f"Product Name: {record.get('product_name', 'N/A')}")
            doc.add_paragraph(f"User: {record.get('user', 'N/A')}")
            doc.add_paragraph(f"URL: {record.get('url', 'N/A')}")
            doc.add_paragraph(f"Question: {record.get('question', 'N/A')}")
            doc.add_paragraph(f"Answer: {record.get('answer', 'N/A')}")
            doc.add_paragraph(f"OCR Summaries: {record.get('ocr_summaries', 'N/A')}")
            doc.add_paragraph(f"Status: {record.get('status', 'N/A')}")
            doc.add_paragraph(f"Error Message: {record.get('error_message', 'None')}")

            summary = self.generate_gpt4_summary(record)
            doc.add_paragraph(f"Summary: {summary}")

        doc.save(file_path)

    def generate_xlsx_report(self, data, file_path):
        df = pd.DataFrame(data)
        df.to_excel(file_path, index=False)

    def generate_reports(self, data):
        current_date = datetime.now().strftime('%Y-%m-%d')
        docx_file_path = os.path.join(self.folder_name, f'Daily_Report_{current_date}.docx')
        xlsx_file_path = os.path.join(self.folder_name, f'Daily_Report_{current_date}.xlsx')

        self.generate_docx_report(data, docx_file_path)
        self.generate_xlsx_report(data, xlsx_file_path)