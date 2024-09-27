from openai import OpenAI
import pandas as pd
import os
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import ctypes
from ocr_handler import OCRHandler

class AnswerGenerator:
    def __init__(self, openai_api_key, data_folder='data'):
        self.client = OpenAI(api_key=openai_api_key)
        self.system_prompt = """
            당신은 자사 쇼핑몰 고객 문의에 응답하는 친절한 상담원입니다. 한국어 문법을 정확히 지키고, 띄어쓰기도 정확하게 합니다.
            고객의 문의에 대해 친절하고 공손하며 정확한 답변을 제공합니다.
            시작은 모두 친절한 인사말과 함께 시작하며 마무리 인사또한 가이드에 따릅니다. 
            모든 응답은 명확한 문장으로 문제 해결을 지원해야 하며, {question}에 대하여 Chain of Thought 접근법을 활용하여 문의를 단계적으로 분석합니다.
            답변에는 "이미지나 URL을 참고해 보았을 때"라는 말을 포함하지 않습니다.
            상품명을 모를 경우 "해당 제품은"이라는 표현을 사용합니다.
            명시되어 있지 않거나 모른다는 표현을 하지 않고 보다 정확한 정보 확인을 위해 라는 표현을 사용합니다.
            명시되어 있지않거나 모르는 문제나 답변이 어려운 문제는 지어내지 않고 "보다 정확한 정보 확인을 위해 "톡톡상담"을 이용하여 사진 파일과 함께 문의해 주시면 더욱 신속한 도움을 드릴 수 있습니다."
            와 같이 "톡톡상담"을 유도합니다.
            마무리 인사를 할때는 "톡톡상담"으로 유도하지 않습니다.
            최대한 해결하기위해 노력합니다.
            문의 내용을 적지 않습니다.
            [제품 이름을 모르므로 '해당 제품은'으로 시작합니다] 이러한 설명을 넣지 않습니다.
            [고객 상담원 이름] 드림 이라는 표현은 넣지 않습니다.
            확인 되지 않는 정보는 보다 정확한 정보 확인을 위해 라는 표현을 사용합니다.
            가장 중요한 것은 근거가 있는 답변만을 하여야 합니다. 절대 모르는 질문에 임의로 대답하지 않습니다. 
            수집된 정보를 바탕으로 계산하여 정확하게 답변합니다. AI라는 것을 절대 언급하지 않으며 최대한 사람 상담원으로써 역할을 다하며 답변합니다.
            전화번호는 언급하지 않습니다.
            제품에 관해서는 최대한 긍정적으로 좋게 설명합니다.
            같은 내용을 두번 반복하여 적지 않습니다.
            정확한 정보가 제공되지 않았습니다 라는 표현을 쓰지 않습니다.
            실제 제품의 크기는 최대 ±5 정도의 오차가 발생할 수 있습니다. 이는 제조 공정상의 특성으로 인한 것입니다.

            무엇무엇인것 같습니다와 같은 추상적 표현을 쓰지 마십시오.
            답변생성시 질문:{question}이나 상품명:{product_name}에대해 절대 언급 하지 않습니다. 
            예를들어:
            상품명:{product_name}에 대해 문의 주셨군요 와 같이 {product_name},{question}을 절대 언급 하지 않습니다.

            주문을 완료해주시면 이라는 표현을 쓰지 않습니다.
            고객 문의는 다음과 같이 두 가지 카테고리로 분류할 수 있습니다 라는 표현을 쓰지 않습니다.
            선물세트 라는 단어가 제품명에 포함돼 있으면 매장안내나 as안내를 하지 않습니다.
            확인해보겠다는 말을 하지 않습니다. 정보를 기반으로 최대한 문제를 해결합니다.
            계산식이나 수학식을 표시할때 영어를 쓰지않고 기호를 사용합니다. 최대한 간단하게 구성합니다.
            정확한 계산방법과 근거를 말하세요.
            정보가 정확히 기재가 돼 있지 않으면 근거를 찾아 계산하거나 산출하세요. 계산과 산출은 수학적으로 정확해야 합니다.
            만약 계산이나 산출이 안된다면 톡톡문의로 넘기세요.

            고정 정보: 본사 직영매장인 "KZM STORE"에서 직접 보시고 구매하실수 있습니다. 매장 정보는 다음과 같습니다.  
            주소: 경기 김포시 양촌읍 양촌역길 80 "KZM TOWER" 영업시간: 10:00~19:00 (연중무휴)
            AS 접수 페이지 : https://support.kzmoutdoor.com
            자사 쇼핑몰 : https://www.kzmmall.com/
            자가수리 부품 AS 전용 웹사이트: https://www.kzmmall.com/category/as부품/110/

            응답 가이드라인: 
            1. 긍정적인 문의에 대한 답변: 
                - 감사의 인사를 전하고, 고객의 긍정적인 경험에 대해 기뻐하는 내용을 전달합니다. 
                - 제품이나 서비스의 특징을 언급하며, 고객의 피드백이 중요하다는 메시지를 포함합니다. 
                - 예시: "소중한 의견 남겨주셔서 감사합니다. 저희 제품을 만족스럽게 사용해주셔서 기쁩니다. 앞으로도 항상 최선을 다하겠습니다."

            2. 중립적이거나 개선이 필요한 문의에 대한 답변:
                - 감사 인사를 전한 후, 고객이 언급한 개선사항이나 불편에 대해 공감하는 내용을 전달합니다.
                - 문제 해결을 위한 조치를 설명하거나, 추가적인 지원이 필요할 경우 고객 지원팀과 연결될 수 있도록 안내합니다.
                - 예시: "남겨주신 의견 감사합니다. 말씀해주신 사항은 매우 중요하게 생각하며, 더 나은 서비스를 위해 노력하겠습니다."

            3. 부정적인 문의에 대한 답변:
                - 진심으로 사과하며, 고객의 불편 사항에 대해 공감하는 내용을 전달합니다.
                - 문제 해결을 위한 방법을 안내하고, 추가로 도움이 필요할 경우 연락할 방법을 안내합니다.
                - 예시: "죄송합니다. 불편함을 드려 정말 죄송합니다. 해당 문제를 신속히 해결할 수 있도록 최선을 다하겠습니다. 추가 문의가 있으시면 언제든지 고객센터로 연락 부탁드립니다."

            4. 반복적인 문의:
                - 동일한 내용의 문의에 대해서는 비슷한 템플릿을 사용하되, 개인화된 문구를 추가하여 고객이 기계적인 답변처럼 느끼지 않도록 합니다.

            기본 톤과 스타일:
                - 항상 긍정적이고 정중하며, 고객의 입장에서 생각하는 모습을 보여줍니다.
                - 말을 길게 하지 않고 간결하고 깔끔하게 대답합니다.
                - 간결하면서도 진심 어린 답변을 제공합니다. 기술적 용어나 전문 용어는 사용을 피하고, 고객이 이해하기 쉬운 언어로 답변합니다.
                - 조사나 검색된 자료에 근거하여 정확한 근거를 제시합니다.
        """
        self.model = SentenceTransformer('sentence-transformers/xlm-r-large-en-ko-nli-ststb')
        self.df = self.load_all_excel_files(data_folder)
        self.ocr_handler = OCRHandler()

        # 디버그: 데이터 확인을 위해 상위 5개 데이터 출력
        print("데이터프레임 상위 데이터:\n", self.df.head())

        # 필수 컬럼 존재 여부 확인
        if '문의내용' not in self.df.columns or '답변내용' not in self.df.columns:
            raise ValueError("엑셀 파일에는 '문의내용' 및 '답변내용' 컬럼이 포함되어야 합니다")

        self.embeddings = self.generate_embeddings(self.df['문의내용'])
        self.index = self.create_faiss_index(self.embeddings)

    def load_all_excel_files(self, folder_path):
        combined_df = pd.DataFrame()
        try:
            for file_name in os.listdir(folder_path):
                if file_name.endswith('.xls') or file_name.endswith('.xlsx'):
                    file_path = os.path.join(folder_path, file_name)
                    print(f"읽고 있는 파일: {file_path}")
                    if file_name.endswith('.xls'):
                        df = pd.read_excel(file_path, engine='xlrd')
                    else:
                        df = pd.read_excel(file_path, engine='openpyxl')
                    combined_df = pd.concat([combined_df, df], ignore_index=True)

            return combined_df
        except Exception as e:
            print(f"엑셀 파일 읽기 오류: {e}")
            raise

    def generate_embeddings(self, texts):
        """Generate embeddings for a list of texts."""
        return self.model.encode(texts)

    def create_faiss_index(self, embeddings):
        """Create a FAISS index from embeddings."""
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        return index

    def find_similar_question(self, question):
        try:
            question_embedding = self.model.encode([question])
            distances, indices = self.index.search(question_embedding, 1)
            most_similar_idx = indices[0][0]
            most_similar_question = self.df.iloc[most_similar_idx]['문의내용']
            similarity_score = 100 - distances[0][0]  # Adjust the similarity score for easier interpretation
            print(f"가장 유사한 질문: {most_similar_question} (유사도 점수: {similarity_score})")
            similar_question_data = self.df.iloc[most_similar_idx]
            print(similar_question_data)
            return similar_question_data, similarity_score
        except Exception as e:
            print(f"유사한 질문 찾기 오류: {e}")
            return None, 0

    def generate_answer(self, question, ocr_summaries, product_name, comment_time=None, current_time=None):
        try:
            if self.check_if_important_question(question):
                self.show_popup_and_wait(question)

            image_summary = ocr_summaries if ocr_summaries else "해당 제품에 대한 이미지에서 정보를 추출하지 않았습니다."

            similar_question_data, score = self.find_similar_question(question)
            if similar_question_data is None:
                similar_question_prompt = "죄송합니다. 데이터베이스에서 유사한 질문을 찾을 수 없습니다."
                similar_question_answer = None
            else:
                similar_question_answer = similar_question_data['답변내용']
                if 100 >= score >= 90:
                    similar_question_prompt = f"다음 모범 답안을 참고해 주세요 (유사도 점수: {score}):\n\nQ: {similar_question_data['문의내용']}\nA: {similar_question_answer}\n\n"
                else:
                    similar_question_prompt = f"다음 유사 질문과 그 답변을 참고해 주세요 (유사도 점수: {score}):\n\nQ: {similar_question_data['문의내용']}\nA: {similar_question_answer}\n\n"

            category_info = self.classify_question(question)
            categories = self.extract_category_from_classification(category_info)
            specific_prompt = self.get_specific_prompt(categories)

            print("카테고리들:", categories)
            print("적용된 specific_prompt:", specific_prompt)

            answer_prompt = f"""
            제공된 제품 정보 및 유사 질문 답변을 바탕으로 다음 문의에 답변해 주세요. 제품명과 고객질문을 언급하지 말아주세요.
            제품명: {product_name}
            고객 질문: {question}
            """

            # Append times if provided
            if comment_time and current_time:
                answer_prompt += f"""
                질문 시간: {comment_time} 
                현재 시간: {current_time}
                """

            answer_prompt += f"""
            {similar_question_prompt}
            제품 정보: {image_summary}
            {specific_prompt}
            """

            chat_completion = self.client.chat.completions.create(
                model="gpt-4o",
                temperature=0.7,
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

    def check_if_important_question(self, question):
        try:
            check_prompt = f"""
            다음 고객 문의가 아래 카테고리와 관련이 있는지 판단해 주세요.
            - 택배관련 문의 
            - 방문수령 문의
            - 송장관련 문의
            - 주소 변경
            - 고객 이름 변경
            - 받는 주소 변경
            - 주문 취소
            - 입고 관련
            - 재입고 관련

            고객 문의: {question}

            반환 예시 (해당되는 경우): "해당합니다."
            반환 예시 (해당되지 않는 경우): "해당하지 않습니다."
            """

            chat_completion = self.client.chat.completions.create(
                model="gpt-4o",
                temperature=0.7,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": check_prompt}
                ]
            )
            gpt_response = chat_completion.choices[0].message.content.strip()
            return "해당합니다" in gpt_response
        except Exception as e:
            print(f"중요한 질문 판별 오류: {e}")
            return False

    def show_popup_and_wait(self, question):
        message_box_styles = 0x00000040 | 0x00000000 | 0x00040000 | 0x00001000
        ctypes.windll.user32.MessageBoxW(0, f"중요한 고객 문의가 들어왔습니다:\n\n{question}", "중요한 문의 알림", message_box_styles)

    def revise_answer(self, user_input, original_answer):
        try:
            revision_prompt = f"""
            수정 전 답변: {original_answer}
            사용자 입력: {user_input}
            사용자의 피드백을 반영하여 답변을 다시 생성해주세요.
            """

            chat_completion = self.client.chat.completions.create(
                model="gpt-4o",
                temperature=0.7,
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

    def classify_question(self, question):
        try:
            classification_prompt = f"""
            다음 고객 문의를 관련된 모든 카테고리로 분류해 주세요. 여러 카테고리가 관련된 경우 모두 표시해 주세요.
            카테고리 목록: [제품 리뷰, 재고/입고/배송, AS (수리, 반품, 교환), 사용법/호환성, 제품 문의, 할인 문의, 기술 지원, 기타 질문]
            다음 고객 문의가 아래 상황 중 하나에 해당한다면 기타 질문으로 분류합니다.
            - 택배관련 문의 
            - 방문수령 문의
            - 송장관련 문의
            - 주소 변경
            - 고객 이름 변경
            - 받는 주소 변경
            - 주문 취소
            - 입고 관련
            - 재입고 관련
            고객 문의: {question}
            """
            chat_completion = self.client.chat.completions.create(
                model="gpt-4o",
                temperature=0.1,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": classification_prompt}
                ]
            )
            category_info = chat_completion.choices[0].message.content
            print(category_info)
            return category_info
        except Exception as e:
            print(f"질문 분류 오류: {e}")
            return "기타 질문"

    def extract_category_from_classification(self, category_info):
        categories = ["제품 리뷰", "재고/입고/배송", "AS (수리, 반품, 교환)", 
                      "사용법/호환성", "제품 문의", "할인 문의", "기술 지원", "기타 질문"]
        detected_categories = [category for category in categories if category in category_info]
        return detected_categories

    def get_specific_prompt(self, categories):
        category_prompts = {
            "제품 리뷰": """
            고객님께서 작성하신 제품 리뷰를 토대로 가능한 친절하고 공손하게 응답해 주세요. 고객님의 피드백을 반영하여 개선될 점이 있다면 공감하며 답변해 주세요.
            제품에관해 전혀 모름으로 제품에 대한 상세 설명을 넣지 않습니다.
            임의로 제품의 사이즈나 특징들을 얘기하지 않습니다.
            """,
            "재고/입고/배송": """
            고객님께서 재고, 입고, 또는 배송에 관한 문의를 주셨습니다. 정확한 재고 상태, 입고 예정일, 배송 가능 여부 및 소요일 등에 대해 상세히 답변해 주세요.
            주문가능하거나 주문된 제품의 재고는 항상 있습니다.
            단종은 아니지만 재입고 날짜가 확정되지 않아 당분간은 구매가 어렵습니다.
            오후1시 이전 주문건은 당일출고가 됩니다.
            오후1시 이후 주문건은 익일출고가 됩니다.
            """,
            "AS (수리, 반품, 교환)": """
            고객님께서 AS, 수리, 반품 또는 교환 절차에 대해 문의를 주셨습니다. 고객님의 불편을 최소화하기 위해 명확하고 자세한 절차를 안내해 주세요.
            AS에 괄호치거나 부가적 설명을 넣지 않습니다.
            인터넷 주소를 두번 적지 않습니다.
            전화번호는 넣지 않습니다.
            금액을 절대 표기하지 않습니다.
            실제 제품의 크기는 최대 ±5 정도의 오차가 발생할 수 있습니다. 이는 제조 공정상의 특성으로 인한 것입니다.
            
            카즈미 텐트의 폴대는 서비스가 가능합니다.

            필요한 부품을 구매해서 자가수리하는 방법과 AS센터에서 진행하는 방식이 있습니다.

            가장 추천하는 방법: 자가수리의 경우 시간과 비용을 절감할 수 있기 때문에 자가수리를 추천드리고 있습니다. as부품 구매와 수리 동영상은 아래의 URL에서 확인 가능하십니다.
            https://www.kzmmall.com/category/as부품/110/
            자가 수리를 최대한 권유합니다.

            AS센터를 통한 서비스 비용 안내와 접수는 아래 URL을 참고하시기 바랍니다. 텐트 수선비용은 수선 방법과 범위에 따라 금액이 달라집니다. 아래 주소에서 예상 수리비용 확인이 가능하십니다.
            https://support.kzmoutdoor.com/ 
            """,
            "사용법/호환성": """
            고객님께서 제품의 사용법 또는 다른 제품과의 호환성에 대해 문의를 주셨습니다. 제품의 사용 방법 및 호환 가능 여부를 상세하게 안내해 주세요.
            """,
            "제품 문의": """
            정확한 계산방법과 근거를 합니다.
            정보가 정확히 기재가 돼 있지 않으면 근거를 찾아 계산하거나 산출하세요. 계산이나 산출이 안된다면 톡톡문의로 넘기세요.
            선물세트 라는 단어가 제품명에 포함돼 있으면 매장안내나 as안내를 하지 않습니다.
            제품에 관해서는 최대한 긍정적으로 좋게 설명합니다.
            계산을 한다면 모든 수치에 대해서는 본 제품과 작은 오차가 있을 수 있습니다. 라는 설명을 넣습니다.
            실제 제품의 크기는 최대 ±5 정도의 오차가 발생할 수 있습니다. 이는 제조 공정상의 특성으로 인한 것입니다.
            단종은 아니지만 재입고 날짜가 확정되지 않아 당분간은 구매가 어렵습니다.

            식기세척기 사용 가능 제품 이름과 모델명:
                에센셜 커틀러리세트, K24T3K02식기세트 22P, K22T3K07
                캠핑 식기세트 17P, K22T3K06캠핑 식기세트 15P, K22T3K05
                캠핑 식기세트 25P, K21T3K11웨스턴 커틀러리 세트, K22T3K01
                트라이 커틀러리 세트, K9T3K004
                쉐프 키친툴세트, K9T3K011
                더블 머그컵 6Pset, K4T3K004
                에그 텀블러 2P, K9T3K010
                프리미엄 STS 푸드 플레이트, K20T3K003프리미엄 코펠세트 XL, K8T3K003
                프리미엄 코펠세트 L, K8T3K002
                프리미엄 STS 패밀리 캠핑 식기세트, K20T3K002
                프리미엄 STS 식기세트 커플, K20T3K001

            식기세척기 사용 불가능한 제품 이름과 모델명:
                이그니스 디자인 팟 그리들 X 얼, K24T3K05이그니스 디자인 그리들 (얼),  K23T3G03
                필드 크레프트 시에라컵 2Pset, K23T3K05GR / K23T3K05BK
                와일드 필드 캠핑컵 8P, K23T3K03
                NEW 블랙 머그 5Pset, K21T3K03웨이브 콜드컵 2Pset, K8T3K007RD
                필드 650 텀블러, K23T3K06
                트윈 우드 커틀러리세트, K21T3K10
                스텐레스 캠핑 주전자 0.8L, K21T3K08 

            - 식기세척기 사용 주의 문구 내용
                식기 세척기 사용 시 코팅(가공)된 제품의 경우 제품 손상 및 변형이 발생할 수 있어 권장하지 않습니다. 
                이는 불량 사유에 해당하지 않으므로, 무상 교환 처리 진행이 불가능합니다. 
                All 스테인리스 제품의 경우 식기 세척기 사용이 가능하나, 
                코팅 또는 나무, 플라스틱 등 재질이 추가된 경우 식기 세척기 사용 시 
                상기 사용 주의 문구 내용처럼 제품의 손상 및 변형이 발생할 수 있습니다.
            """,
            "할인 문의": """
            고객님께서 제품 할인에 대한 문의를 주셨습니다. 현재 진행 중인 할인 정보 및 적용 조건을 안내해 주세요.
            """,
            "기술 지원": """
            고객님께서 기술 지원에 대해 문의하셨습니다. 가능한 신속하고 정확한 기술 지원을 제공해 주세요.
            """,
            "기타 질문": """
            고객님께서 기타 문의를 주셨습니다. 질문의 의도를 다시 한번 파악하고 질문과 상황에 맞는 답변을 생성해주세요.
            모른다거나 확인이 필요하다 라는 답변을 하지 않습니다.
            추가정보를 요구하지 않습니다.
            주문가능하거나 주문된 제품의 재고는 항상 있습니다.
            단종은 아니지만 재입고 날짜가 확정되지 않아 당분간은 구매가 어렵습니다.
            오후1시 이전 주문건은 당일출고가 됩니다.
            오후1시 이후 주문건은 익일출고가 됩니다.
            """
        }
        prompt = "\n".join([category_prompts[cat] for cat in categories])
        return prompt