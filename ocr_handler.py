import os
import requests
import base64
from PIL import Image, ImageEnhance
from io import BytesIO
from bs4 import BeautifulSoup
from openai import OpenAI
import pytesseract
import numpy as np
import cv2
from dotenv import load_dotenv

class OCRHandler:
    def __init__(self):
        # Load OpenAI API Key from .env file
        load_dotenv()
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY must be set in the .env file")
        
        self.client = OpenAI(api_key=openai_api_key)
        self.system_prompt = """
        당신은 다양한 데이터 소스(이미지, OCR, HTML)에서 추출된 정보를 바탕으로 상품에 대해 종합적으로 분석하고, 매우 자세한 설명을 제공하는 AI 에이전트입니다. 
        당신의 목표는 모든 가용 데이터를 활용하여 사용자가 상품의 주요 특징, 사양, 장단점, 용도, 및 관련 정보를 이해할 수 있도록 돕는 것입니다.
        해당 제품과 제품의 옵션별로 사이즈가 여러가지로 나올 수 있습니다. 해당 본제품과 옵션의 사이즈를 구분하여 정리하십시오.

        제품과 사이즈는 두가지 이상일 수 있습니다. 반드시 구분하여 정리해주십시오.

        PRODUCT SPEC 밑에 주로 중요한 정보가 나옵니다.
        size는 보통 290x220x(h)195cm 라는것의 의미는 가로의 길이가 290 세로가 220 높이는 195 단위는 cm라는 뜻입니다.
        특수한 케이스로는 텐트종류는 모양이 마름모인 경우가 많습니다.
        예를들어 size가 290(240)x220x(h)195cm 라는것의 의미는 가로의 길이는 긴측 가로가 290이고 짧은측은 240 그리고 세로는 220 높이는 195라는 의미고 단위는 cm입니다. 마름모 모양을 의미합니다. 반드시 참고해주십시오.
        위 사항을 반드시 주의하여 요약하십시오.

        상세 페이지 참조 혹은 참고 라는 말을 절대 사용하지 않습니다.
        무조건 입력받은 TEXT를 기반으로 답변을 작성합니다.
        질문과 관계 없이 모든 데이터를 정리합니다.

        다음의 지침을 따르십시오:

        데이터 통합:
        이미지에서 추출된 시각적 정보를 텍스트로 변환하여 분석하십시오. 이 정보에는 상품의 사이즈, 상품의 크기, 상품의 외관, 디자인, 색상, 로고, 텍스트 등이 포함될 수 있습니다.
        OCR 데이터를 활용하여 상품의 라벨, 설명서, 광고 텍스트 등에서 추출된 텍스트 정보를 분석하십시오.
        HTML에서 추출된 텍스트 데이터는 웹페이지에서 제공하는 공식 정보, 고객 리뷰, 기술 사양, 가격 비교, 상품의 사이즈, 상품의 크기, 할인 정보 등을 포함할 수 있습니다.
        상품 요약:

        모든 출처의 데이터를 종합하여 상품의 이름, 브랜드, 모델, 주요 특징, 사양 등을 정확하게 요약하십시오.
        상품의 사용 목적, 타겟 소비자층, 장단점 등을 포함한 종합적인 평가를 제공하십시오.
        가격 정보가 포함되어 있다면, 시장에서의 경쟁력 및 할인 정보도 함께 설명하십시오.
        세부 정보 제공:

        크기와 사이즈를 깊이 있게 분석하십시오.
        이미지나 OCR에서 추출된 특정 텍스트(예: 가로, 세로, 높이, 측면, 안쪽, 외관과 특정 위치에 대한 사이즈, 이너텐트)가 중요한 경우 이를 강조하여 설명하십시오.
        구조화된 정보:

        요약한 정보를 잘 정리된 형태로 제시하십시오. 예를 들어, 표, 리스트, 또는 순서 목록을 사용하여 가독성을 높이십시오.
        "재고/입고/배송", "AS (수리, 반품, 교환)", "사용법/호환성", "제품 문의", "할인 문의", "기술 지원", "기타 질문" 해당 카테고리에 해당되는 정보를 고려하여야 합니다.
        
        유통기한이 명시된 경우 유통기한이 제품 중 가장 짧은 제품의 기준임을 반드시 알려주세요.
        가장 중요한 것은 정확한 근거가 있지 않은 정보는 적지 않습니다.
        고객 관점 고려:

        고객이 중요하게 생각할 수 있는 사항을 강조하십시오. 예를 들어, 이너텐트 사이즈, 제품의 사이즈, 크기, 성능, 내구성, 가격 대비 가치, 사용자 리뷰 등입니다.
        통합적 분석:

        여러 소스에서 얻은 정보를 종합하여, 일관성 있는 결론을 도출하십시오. 서로 다른 출처에서 상반된 정보가 있을 경우, 가능한 경우 출처를 명시하고 이를 설명하십시오.
        """

    @staticmethod
    def preprocess_image(img, upscale_factor=2):
        """Preprocess the image to improve OCR results."""
        try:
            # Increase DPI and upscale the image
            img = img.resize((img.width * upscale_factor, img.height * upscale_factor), Image.LANCZOS)
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(img)
            enhanced_img = enhancer.enhance(2)

            # Convert to numpy array
            enhanced_img_np = np.array(enhanced_img)

            # Convert to grayscale
            gray_img = cv2.cvtColor(enhanced_img_np, cv2.COLOR_BGR2GRAY)
            
            # Apply adaptive thresholding
            thresh_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

            # Apply denoising
            denoised_img = cv2.fastNlMeansDenoising(thresh_img, None, 30, 7, 21)

            # Convert back to PIL Image
            processed_img = Image.fromarray(denoised_img)

            return processed_img
        except Exception as e:
            raise RuntimeError(f"Error in image preprocessing: {e}")

    @staticmethod
    def split_image_vertically(img, max_height=3000):
        """Splits an image vertically into smaller parts if it exceeds max_height."""
        width, height = img.size
        if height <= max_height:
            return [img]
        
        split_images = []
        for i in range(0, height, max_height):
            box = (0, i, width, min(i + max_height, height))
            split_images.append(img.crop(box))
        
        return split_images

    @staticmethod
    def ocr_from_image(img):
        """Performs OCR on an image and returns detected text."""
        try:
            preprocessed_img = OCRHandler.preprocess_image(img)
            
            # If the image is too tall, split it into chunks and perform OCR on each
            split_images = OCRHandler.split_image_vertically(preprocessed_img)
            ocr_results = []
            for idx, part_img in enumerate(split_images):
                text = pytesseract.image_to_string(part_img, lang='kor+eng')
                ocr_results.append(f"Part {idx + 1}:\n{text.strip()}")

            return "\n".join(ocr_results)
        except Exception as e:
            return f"Error in OCR processing: {e}"

    @staticmethod
    def ocr_from_image_url(image_url):
        """Performs OCR on an image from a URL."""
        try:
            if 'data:image' in image_url:
                header, encoded = image_url.split(",", 1)
                img_data = base64.b64decode(encoded)
                img = Image.open(BytesIO(img_data))
            else:
                response = requests.get(image_url)
                response.raise_for_status()
                img = Image.open(BytesIO(response.content))

            return OCRHandler.ocr_from_image(img)
        except Exception as e:
            return f"Error in OCR from URL: {e}"

    @staticmethod
    def extract_image_urls_from_html(html_content):
        """Extracts image URLs from the provided HTML content."""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            image_tags = soup.find_all('img')

            image_urls = []
            for img in image_tags:
                src = img.get('src')
                data_src = img.get('data-src')
                if src:
                    image_urls.append(src)
                if data_src:
                    image_urls.append(data_src)
            return image_urls
        except Exception as e:
            raise RuntimeError(f"Error in extracting image URLs from HTML: {e}")

    @staticmethod
    def extract_text_from_html(html_content):
        """Extracts all text content from the provided HTML content."""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            text = soup.get_text(separator=' ', strip=True)
            return text
        except Exception as e:
            raise RuntimeError(f"Error in extracting text from HTML: {e}")

    @staticmethod
    def is_xpath_like_image_url(url):
        """Checks if the image URL fits the specified xpath-like pattern."""
        return 'cdn.kzmoutdoor.com/shop_image' in url or 'shop-phinf.pstatic.net' in url

    def ocr_from_html_content(self, html_content):
        """Performs OCR on all images found within the provided HTML content and also extracts text directly from the HTML content."""
        try:
            image_urls = self.extract_image_urls_from_html(html_content)
            html_text = self.extract_text_from_html(html_content)
            ocr_results = []

            for image_url in image_urls:
                if self.is_xpath_like_image_url(image_url):
                    result = self.ocr_from_image_url(image_url)
                    if isinstance(result, str) and result:
                        ocr_results.append({'image_url': image_url, 'ocr_text': result})

            # Append extracted text from HTML
            ocr_results.append({'source': 'html_content', 'ocr_text': html_text})

            # Summarize ocr results
            summaries = self.summarize_ocr_results(ocr_results)
            return summaries
        except Exception as e:
            return f"Error in OCR from HTML content: {e}"

    def summarize_ocr_results(self, ocr_results):
        """Summarizes OCR results for all grouped products."""
        summaries = []

        try:
            combined_text = "\n".join([result.get('ocr_text', '') for result in ocr_results])

            max_tokens = 8192
            shortened_ocr_text = self.shorten_text_to_token_limit(combined_text, max_tokens)

            chat_completion = self.client.chat.completions.create(
                model="gpt-4o",
                temperature=0.7,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": shortened_ocr_text}
                ]
            )
            summary = chat_completion.choices[0].message.content
            summaries.append({'summary': summary})
            return summaries
        except Exception as e:
            summaries.append({'summary': f"Error in generating OCR summary: {e}"})
            return summaries

    @staticmethod
    def shorten_text_to_token_limit(text, token_limit):
        """Shortens the text to fit within the specified token limit."""
        tokens = text.split()
        if len(tokens) > token_limit:
            return " ".join(tokens[:token_limit])
        return text


# Example Usage (Debugging the extracted HTML structure separately)
def debug_html_structure(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    print(soup.prettify())

# Example testing for the given large image URL
# image_url = "https://cdn.kzmoutdoor.com/shop_image/K241T3T01.jpg"
# ocr_handler = OCRHandler()
# ocr_result = ocr_handler.ocr_from_image_url(image_url)
# print(ocr_result)