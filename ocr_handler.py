from PIL import Image
from io import BytesIO
import requests
import base64
import pytesseract
from bs4 import BeautifulSoup
from openai import OpenAI
from dotenv import load_dotenv
import os
import json

class OCRHandler:
    def __init__(self):
        # Load OpenAI API Key from .env file
        load_dotenv()
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY must be set in the .env file")

        self.client = OpenAI(api_key=openai_api_key)
        self.system_prompt = """
        당신은 다양한 데이터 소스(이미지, OCR, HTML)에서 추출된 정보를 바탕으로 상품에 대해 종합적으로 분석하고, 
        매우 자세한 설명을 제공하는 AI 에이전트입니다. 당신의 목표는 모든 가용 데이터를 활용하여 사용자가 상품의 주요 특징, 사양, 
        장단점, 용도, 및 관련 정보를 이해할 수 있도록 돕는 것입니다.

        다음의 지침을 따르십시오:

        데이터 통합:
        이미지에서 추출된 시각적 정보를 텍스트로 변환하여 분석하십시오. 이 정보에는 상품의 외관, 디자인, 색상, 로고, 텍스트 등이 포함될 수 있습니다.
        OCR 데이터를 활용하여 상품의 라벨, 설명서, 광고 텍스트 등에서 추출된 텍스트 정보를 분석하십시오.
        HTML에서 추출된 텍스트 데이터는 웹페이지에서 제공하는 공식 정보, 고객 리뷰, 기술 사양, 가격 비교, 할인 정보 등을 포함할 수 있습니다.
        상품 요약:

        모든 출처의 데이터를 종합하여 상품의 이름, 브랜드, 모델, 주요 특징, 사양 등을 정확하게 요약하십시오.
        상품의 사용 목적, 타겟 소비자층, 장단점 등을 포함한 종합적인 평가를 제공하십시오.
        가격 정보가 포함되어 있다면, 시장에서의 경쟁력 및 할인 정보도 함께 설명하십시오.
        세부 정보 제공:

        상품의 각 주요 특징에 대해 상세히 설명하십시오. 예를 들어, 기술적 사양이 중요한 경우, 이를 깊이 있게 분석하십시오.
        이미지나 OCR에서 추출된 특정 텍스트(예: "Made in Italy", "100% organic")가 중요한 경우 이를 강조하여 설명하십시오.
        구조화된 정보:

        요약한 정보를 잘 정리된 형태로 제시하십시오. 예를 들어, 표, 리스트, 또는 순서 목록을 사용하여 가독성을 높이십시오.
        "재고/입고/배송", "AS (수리, 반품, 교환)", "사용법/호환성", "제품 문의", "할인 문의", "기술 지원", "기타 질문" 해당 카테고리에 해당되는 정보를 고려하여야 합니다.
        
        유통기한이 명시된경우 유통기한이 제품중 가장 짧은 제품의 기준임을 반드시 알려주세요.
        가장 중요한것은 정확한 근거가 있지 않은 정보는 적지 않습니다.
        고객 관점 고려:

        고객이 중요하게 생각할 수 있는 사항을 강조하십시오. 예를 들어, 성능, 내구성, 가격 대비 가치, 사용자 리뷰 등입니다.
        통합적 분석:

        여러 소스에서 얻은 정보를 종합하여, 일관성 있는 결론을 도출하십시오. 서로 다른 출처에서 상반된 정보가 있을 경우, 가능한 경우 출처를 명시하고 이를 설명하십시오.
        """

    @staticmethod
    def ocr_from_image_url(image_url):
        """Performs OCR on an image from a given URL."""
        try:
            if 'data:image' in image_url:
                header, encoded = image_url.split(",", 1)
                img_data = base64.b64decode(encoded)
                img = Image.open(BytesIO(img_data))
            else:
                response = requests.get(image_url)
                response.raise_for_status()
                img = Image.open(BytesIO(response.content))

            # Perform OCR for Korean and English
            text_kor = pytesseract.image_to_string(img, lang='kor')
            text_eng = pytesseract.image_to_string(img, lang='eng')

            return text_kor + " " + text_eng
        except Exception as e:
            return ""

    @staticmethod
    def extract_image_urls_from_html(html_content):
        """Extracts image URLs from the provided HTML content."""
        soup = BeautifulSoup(html_content, 'html.parser')
        image_tags = soup.find_all('img')

        # Collecting all src and data-src image URLs, without filtering
        image_urls = []
        for img in image_tags:
            src = img.get('src')
            data_src = img.get('data-src')
            if src:
                image_urls.append(src)
            if data_src:
                image_urls.append(data_src)

        return image_urls

    @staticmethod
    def extract_text_from_html(html_content):
        """Extracts all text content from the provided HTML content."""
        soup = BeautifulSoup(html_content, 'html.parser')
        text = soup.get_text(separator=' ', strip=True)
        return text

    @staticmethod
    def is_xpath_like_image_url(url):
        """Checks if the image URL fits the specified xpath-like pattern."""
        return 'cdn.kzmoutdoor.com/shop_image' in url

    def ocr_from_html_content(self, html_content):
        """Performs OCR on all images found within the provided HTML content
           and also extracts text directly from the HTML content."""
        
        # Extract image URLs and direct text from HTML
        image_urls = self.extract_image_urls_from_html(html_content)
        html_text = self.extract_text_from_html(html_content)
        ocr_results = []

        for image_url in image_urls:
            if self.is_xpath_like_image_url(image_url):
                result = self.ocr_from_image_url(image_url)
                if result:
                    ocr_results.append(result)

        # Append HTML text to the results
        ocr_results.append(html_text)

        # Summarize OCR results while considering the token limit
        return self.summarize_ocr_results(ocr_results)

    def summarize_ocr_results(self, ocr_results):
        ocr_text = " ".join(ocr_results)
        
        # Ensure the text does not exceed the token limit
        max_tokens = 8192  # Adjust to have some room for the system prompt
        shortened_ocr_text = self.shorten_text_to_token_limit(ocr_text, max_tokens)
        
        try:
            completion = self.client.chat.completions.create(
                model="gpt-4o",
                temperature=0.7,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": shortened_ocr_text}
                ]
            )
            summary = completion.choices[0].message.content.strip()
            return summary
        except Exception as e:
            return "OCR 요약을 생성할 수 없습니다."

    @staticmethod
    def shorten_text_to_token_limit(text, token_limit):
        """Shortens the text to fit within the specified token limit."""
        tokens = text.split()
        if len(tokens) > token_limit:
            return " ".join(tokens[:token_limit])
        return text

# Debugging the extracted HTML structure separately
def debug_html_structure(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    print(soup.prettify())