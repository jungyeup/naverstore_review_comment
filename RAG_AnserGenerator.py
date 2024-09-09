from openai import OpenAI
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import os

class AnswerGenerator:
    def __init__(self, openai_api_key, data_folder='data'):
        self.client = OpenAI(api_key=openai_api_key)
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # Any appropriate embedding model
        self.system_prompt = """
            ... [The same system prompt as before]
        """
        self.df = self.load_all_excel_files(data_folder)
        self.index = self.build_faiss_index()
        
        if '문의내용' not in self.df.columns or '답변내용' not in self.df.columns:
            raise ValueError("엑셀 파일에는 '문의내용' 및 '답변내용' 컬럼이 포함되어야 합니다")

    def load_all_excel_files(self, folder_path):
        combined_df = pd.DataFrame()
        try:
            for file_name in os.listdir(folder_path):
                if file_name.endswith('.xls') or file_name.endswith('.xlsx'):
                    file_path = os.path.join(folder_path, file_name)
                    if file_name.endswith('.xls'):
                        df = pd.read_excel(file_path, engine='xlrd')
                    elif file_name.endswith('.xlsx'):
                        df = pd.read_excel(file_path, engine='openpyxl')
                    combined_df = pd.concat([combined_df, df], ignore_index=True)
            return combined_df
        except Exception as e:
            print(f"엑셀 파일 읽기 오류: {e}")
            raise
        
    def build_faiss_index(self):
        # Assuming the column we are interested in is named '문의내용'
        embeddings = self.model.encode(self.df['문의내용'].tolist())
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        return index
    
    def retrieve_similar_responses(self, question, top_k=3):
        question_embedding = self.model.encode([question])
        D, I = self.index.search(question_embedding, top_k)
        retrieved_responses = self.df.iloc[I[0]]
        print(retrieved_responses, D)
        return retrieved_responses, D

    def generate_answer(self, question, ocr_texts, product_name, model_name=None):
        try:
            ocr_summary = " ".join(ocr_texts) if ocr_texts else "해당 제품에 대한 이미지에서 정보를 추출하지 않았습니다."
            image_summary = self.get_image_summary(ocr_summary) if ocr_texts else ocr_summary

            retrieved_responses, distances = self.retrieve_similar_responses(question)
            if retrieved_responses.empty:
                similar_question_prompt = "죄송합니다. 데이터베이스에서 유사한 질문을 찾을 수 없습니다."
                similar_question_answer = None
            else:
                similar_questions_text = "\n\n".join([
                    f"Q: {row['문의내용']}\nA: {row['답변내용']}" for _, row in retrieved_responses.iterrows()
                ])
                similar_question_prompt = f"다음 유사 질문과 그 답변을 참고해 주세요:\n\n{similar_questions_text}"

            category_info = self.classify_question(question)
            category = self.extract_category_from_classification(category_info)
            specific_prompt = self.get_specific_prompt(category)
            print(similar_question_prompt)

            answer_prompt = f"""
            제공된 제품 정보 및 유사 질문 답변을 바탕으로 다음 문의에 답변해 주세요. 제품명과 고객질문을 언급하지 말아주세요.
            제품명: {product_name} 모델명: {model_name}
            고객 질문: {question}
            {similar_question_prompt}
            제품 정보: {image_summary}
            {specific_prompt}
            """
            
            chat_completion = self.client.chat.completions.create(
                model="gpt-4o",
                temperature=0.8,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": answer_prompt}
                ]
            )
            answer = chat_completion.choices[0].message.content
            return answer
        except Exception as e:
            print(f"답변 생성 오류: {e}")
            return "죄송합니다, 답변을 생성할 수 없습니다."

    def revise_answer(self, user_input, original_answer):
        try:
            revision_prompt = f"""
            수정 전 답변: {original_answer}
            사용자 입력: {user_input}
            사용자의 피드백을 반영하여 답변을 다시 생성해주세요.
            """
            chat_completion = self.client.chat.completions.create(
                model="gpt-4o",
                temperature=0.8,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": revision_prompt}
                ]
            )
            revised_answer = chat_completion.choices[0].message.content
            return revised_answer
        except Exception as e:
            print(f"답변 수정 오류: {e}")
            return original_answer

    def get_image_summary(self, ocr_summary):
        try:
            image_info_prompt = f"""
            다음 제품 정보를 요약합니다: ...
            추출된 텍스트: {ocr_summary}
            """
            chat_completion = self.client.chat.completions.create(
                model="gpt-4o",
                temperature=0.8,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": image_info_prompt}
                ]
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            print(f"이미지 요약 생성 오류: {e}")
            return "이미지 요약을 생성할 수 없습니다."

    def classify_question(self, question):
        try:
            classification_prompt = f"""
            다음 고객 문의를 아래 카테고리 중 하나로 분류해 주세요:
            카테고리 목록: [제품 리뷰, 재고/입고/배송, AS (수리, 반품, 교환), 사용법/호환성, 제품 문의, 할인 문의, 기술 지원, 기타 질문]
            고객 문의: {question}
            """
            chat_completion = self.client.chat.completions.create(
                model="gpt-4o",
                temperature=0.8,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": classification_prompt}
                ]
            )
            category_info = chat_completion.choices[0].message.content
            return category_info
        except Exception as e:
            print(f"질문 분류 오류: {e}")
            return "기타 질문"

    def extract_category_from_classification(self, category_info):
        categories = ["제품 리뷰", "재고/입고/배송", "AS (수리, 반품, 교환)", 
                      "사용법/호환성", "제품 문의", "할인 문의", "기술 지원", "기타 질문"]
        for category in categories:
            if category in category_info:
                return category
        return "기타 질문"

    def get_specific_prompt(self, category):
        category_prompts = {
            "제품 리뷰": """
            고객님께서 작성하신 제품 리뷰를 토대로 가능한 친절하고 공손하게 응답해 주세요. 고객님의 피드백을 반영하여 개선될 점이 있다면 공감하며 답변해 주세요.
            제품에관해 전혀 모름으로 제품에 대한 상세 설명을 넣지 않습니다.
            임의로 제품의 사이즈나 특징들을 얘기하지 않습니다.
            """,
            "재고/입고/배송": """
            고객님께서 재고, 입고, 또는 배송에 관한 문의를 주셨습니다. 정확한 재고 상태, 입고 예정일, 배송 가능 여부 및 소요일 등에 대해 상세히 답변해 주세요.
            재고는 항상 있습니다.
            오후1시 이전 주문건은 당일출고가 됩니다.
            오후1시 이후 주문건은 익일출고가 됩니다.
            추석 택배 마감은 9/12 오후 1시 이전 주문 건까지 추석 전 발송 가능하다고 안내합니다.
            """,
            "AS (수리, 반품, 교환)": """
            고객님께서 AS, 수리, 반품 또는 교환 절차에 대해 문의를 주셨습니다. 고객님의 불편을 최소화하기 위해 명확하고 자세한 절차를 안내해 주세요.
            AS에 괄호치거나 부가적 설명을 넣지 않습니다.
            인터넷 주소를 두번적지 않습니다.
            전화번호는 넣지 않습니다.
            ### A/S 비용 안내 시스템

            ## 텐트 수선 비용

            다음 테이블은 텐트의 손상 범위(cm)에 따른 수선 비용입니다. 단위는 원(₩)입니다.

            | 범류(cm) 및 범주  | ~ 3cm | 10cm ~ | 20cm ~ | 30cm ~ |
            |------------------|-------|--------|--------|-------|
            | 덧댐               | 30,000 | 50,000 | 60,000 | 상담   |
            | 웰더 혹은 3cm~7cm | 30,000 | 불가    | 불가    | 상담   |
            | 덧댐 + 박음질      | 50,000 | 70,000 | 90,000 | 상담   |

            *수선 유형 설명:*
            1. **덧댐**: 손상 부분에 패치를 덧대어 수선합니다.
            2. **웰더 혹은 3cm~7cm**: 손상 부위를 용접 방식(습식 용접)으로 수선합니다.
            3. **덧댐 + 박음질**: 패치를 덧댐과 동시에 박음질하여 내구성을 상승시킵니다.

            *주의사항:*
            - 손상된 부분에 따라 기본 수선비가 발생하며, 추천된 수선 방식에 따라 추가 비용이 발생합니다.
            - 천의 상태 및 손상 부위 상황에 따라, 추가적인 지퍼 수선 또는 교체 작업이 권장될 수 있으며 이에 따른 추가비용이 발생할 수 있습니다.
            - 큰 손상이나 수선이 불가한 손상의 경우, 천의 교체가 필요하며 이 경우 상담 후 비용이 결정됩니다.

            ## 폴대 수선 비용

            다음 테이블은 폴대의 마디 교체 비용입니다. 단위는 원(₩)입니다.

            | 수량 및 범주 | 10대 | 20대 | 30대 | 40대 이상 |
            |--------------|------|------|------|-----------|
            | 폴대 테이프  | 10,000 | 20,000 | 30,000 | 40,000 |
            | 수량 추가비용| 50,000 | 60,000 | 70,000 | 상담     |

            *수선 유형 설명:*
            1. **폴대 테이프 교체**: 폴대의 손상된 부분에 테이프를 부착하여 임시 수선합니다.
            2. **수량 추가비용**: 수량이 많을수록 추가비용이 발생하며, 대량일 경우 일부 할인이 적용됩니다.

            *주의사항:*
            - 폴대는 손상 발생 시 철저한 점검을 통해 수선 또는 교체 작업이 필요합니다.
            - 심한 손상으로 인해 수선이 불가한 경우, 폴대 전체를 교체해야 하며 이에 따른 추가비용이 발생할 수 있습니다.
            - 폴대 수량에 따른 정밀 검사를 사전에 진행해야 하며, 검사 결과에 따라 추가 비용이 발생할 수 있습니다.

            이 시스템은 주어진 기준에 따라 A/S 비용을 정확하게 계산하고 안내합니다. 고객이 필요한 정보를 명확하고 간결하게 제공하여 올바른 A/S 비용을 안내하십시오. 고객의 상황에 맞는 최선의 수선 방식을 추천하고, 가능한 한 정확한 비용 예산을 제공합니다.
            """,
            "사용법/호환성": """
            고객님께서 제품의 사용법 또는 다른 제품과의 호환성에 대해 문의를 주셨습니다. 제품의 사용 방법 및 호환 가능 여부를 상세하게 안내해 주세요.
            """,
            "제품 문의": """
            고객님께서 제품에 대한 구체적인 정보를 문의하셨습니다. 제품의 상세 정보 및 특징을 전달해 주세요.
            정확한 계산방법과 근거를 말하세요.
            정확히 기재가 돼 있지 않으면 근거를 찾아 계산하거나 산출하세요.
            선물세트 라는 단어가 제품명에 포함돼 있으면 매장안내나 as안내를 하지 않습니다.
            제품에 관해서는 최대한 긍정적으로 좋게 설명합니다.
            계산을 한다면 모든 수치에 대해서는 본제품과 작은 오차가 있을 수 있습니다. 라는 설명을 넣습니다.
            """,
            "할인 문의": """
            고객님께서 제품 할인에 대한 문의를 주셨습니다. 현재 진행중인 할인 정보 및 적용 조건을 안내해 주세요.
            """,
            "기술 지원": """
            고객님께서 기술 지원에 대해 문의하셨습니다. 가능한 신속하고 정확한 기술 지원을 제공해 주세요.
            """,
            "기타 질문": """
            고객님께서 기타 문의를 주셨습니다. 가능한 한 명확하고 유용한 답변을 제공해 주세요.
            """
        }
        return category_prompts.get(category, "")