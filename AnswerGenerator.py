from openai import OpenAI
import pandas as pd
from fuzzywuzzy import process
import os

class AnswerGenerator:
    def __init__(self, openai_api_key, data_folder='data'):
        self.client = OpenAI(api_key=openai_api_key)
        self.system_prompt = """
            당신은 자사 쇼핑몰 고객 문의에 응답하는 친절한 상담원입니다. 한국어 문법을 정확히 지키고, 
            고객의 문의에 대해 친절하고 공손하며 정확한 답변을 제공합니다.
            시작은 모두 친절한 인사말과 함께 시작하며 마무리 인사또한 가이드에 따릅니다. 
            모든 응답은 명확한 문장으로 문제 해결을 지원해야 하며, {question}에 대하여 Chain of Thought 접근법을 활용하여 문의를 단계적으로 분석합니다.
            답변에는 "이미지나 URL을 참고해 보았을 때"라는 말을 포함하지 않습니다. 
            상품명을 모를 경우 "해당 제품은"이라는 표현을 사용합니다.
            명시되어 있지 않거나 모른다는 표현을 하지 않고 보다 정확한 정보 확인을 위해 라는 표현을 사용합니다.
            명시되어 있지않거나 모르는 문제나 답변이 어려운 문제는 지어내지 않고 "보다 정확한 정보 확인을 위해 "톡톡상담"을 이용하여 사진 파일과 함께 문의해 주시면 더욱 신속한 도움을 드릴 수 있습니다."
            와 같이 "톡톡상담"을 유도합니다.
            [제품 이름을 모르므로 '해당 제품은'으로 시작합니다] 이러한 설명을 넣지 않습니다.
            [고객 상담원 이름] 드림 이라는 표현은 넣지 않습니다.
            확인 되지 않는 정보는 보다 정확한 정보 확인을 위해 라는 표현을 사용합니다.
            가장 중요한 것은 근거가 있는 답변만을 하여야 합니다. 절대 모르는 질문에 임의로 대답하지 않습니다. 
            수집된 정보를 바탕으로 계산하여 정확하게 답변합니다. AI라는 것을 절대 언급하지 않으며 최대한 사람 상담원으로써 역할을 다하며 답변합니다.
            전화번호는 언급하지 않습니다.
            제품에 관해서는 최대한 긍정적으로 좋게 설명합니다.
            같은 내용을 두번 반복하여 적지 않습니다.
            정확한 정보가 제공되지 않았습니다 라는 표현을 쓰지 않습니다.

            중요한건 답변생성시 질문:{question}이나 상품명:{product_name}에대해 반복하여 말하지 않습니다. 
            상품명:{product_name}에 대해 문의 주셨군요 와 같이 {product_name},{question}을 말하지 않습니다.

            주문을 완료해주시면 이라는 표현을 쓰지 않습니다.
            고객 문의는 다음과 같이 두 가지 카테고리로 분류할 수 있습니다 라는 표현을 쓰지 않습니다.
            선물세트 라는 단어가 제품명에 포함돼 있으면 매장안내나 as안내를 하지 않습니다.
            확인해보겠다는 말을 하지 않습니다. 정보를 기반으로 최대한 문제를 해결합니다.
            계산식이나 수학식을 표시할때 영어를 쓰지않고 기호를 사용합니다. 최대한 간단하게 구성합니다.

            고정 정보: 본사 직영매장인 "KZM STORE"에서 직접 보시고 구매하실수 있습니다. 매장 정보는 다음과 같습니다.  
            주소: 경기 김포시 양촌읍 양촌역길 80 "KZM TOWER" 영업시간: 10:00~19:00 (연중무휴)
            AS 접수 페이지 : https://support.kzmoutdoor.com
            자사 쇼핑몰 : https://www.kzmmall.com/
            부품 AS 전용 웹사이트: https://www.kzmmall.com/category/as부품/110/

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
                - 간결하면서도 진심 어린 답변을 제공합니다. 기술적 용어나 전문 용어는 사용을 피하고, 고객이 이해하기 쉬운 언어로 답변합니다.
                - 조사나 검색된 자료에 근거하여 정확한 근거를 제시합니다.
        """
        self.df = self.load_all_excel_files(data_folder)

        # 디버그: 데이터 확인을 위해 상위 5개 데이터 출력
        print("데이터프레임 상위 데이터:\n", self.df.head())

        # 필수 컬럼 존재 여부 확인
        if '문의내용' not in self.df.columns or '답변내용' not in self.df.columns:
            raise ValueError("엑셀 파일에는 '문의내용' 및 '답변내용' 컬럼이 포함되어야 합니다")

    def load_all_excel_files(self, folder_path):
        combined_df = pd.DataFrame()

        try:
            for file_name in os.listdir(folder_path):
                if file_name.endswith('.xls') or file_name.endswith('.xlsx'):
                    file_path = os.path.join(folder_path, file_name)
                    print(f"읽고 있는 파일: {file_path}")
                    if file_name.endswith('.xls'):
                        df = pd.read_excel(file_path, engine='xlrd')
                    elif file_name.endswith('.xlsx'):
                        df = pd.read_excel(file_path, engine='openpyxl')

                    combined_df = pd.concat([combined_df, df], ignore_index=True)

            return combined_df
        except Exception as e:
            print(f"엑셀 파일 읽기 오류: {e}")
            raise

    def find_similar_question(self, question):
        try:
            question_titles = self.df['문의내용'].tolist()
            most_similar, score = process.extractOne(question, question_titles)
            print(f"가장 유사한 질문: {most_similar} (유사도 점수: {score})")
            similar_question_data = self.df[self.df['문의내용'] == most_similar].iloc[0]
            print(similar_question_data)
            return similar_question_data, score
        except Exception as e:
            print(f"유사한 질문 찾기 오류: {e}")
            return None, 0

    def generate_answer(self, question, ocr_texts, product_name):
        try:
            # Always include OCR summaries if they exist
            ocr_summary = " ".join(ocr_texts) if ocr_texts else "해당 제품에 대한 이미지에서 정보를 추출하지 않았습니다."
            image_summary = self.get_image_summary(ocr_summary) if ocr_texts else ocr_summary

            similar_question_data, score = self.find_similar_question(question)
            if similar_question_data is None:
                similar_question_prompt = "죄송합니다. 데이터베이스에서 유사한 질문을 찾을 수 없습니다."
                similar_question_answer = None
            else:
                similar_question_answer = similar_question_data['답변내용']
                if score >= 90:
                    similar_question_prompt = f"다음 모범 답안을 참고해 주세요 (유사도 점수: {score}):\n\nQ: {similar_question_data['문의내용']}\nA: {similar_question_answer}\n\n"
                else:
                    similar_question_prompt = f"다음 유사 질문과 그 답변을 참고해 주세요 (유사도 점수: {score}):\n\nQ: {similar_question_data['문의내용']}\nA: {similar_question_answer}\n\n"

            # 여기서 카테고리를 분류합니다
            category_info = self.classify_question(question)
            category = self.extract_category_from_classification(category_info)
            specific_prompt = self.get_specific_prompt(category)

            # specific_prompt 디버그 출력
            print("카테고리:", category)
            print("적용된 specific_prompt:", specific_prompt)

            answer_prompt = f"""
            제공된 제품 정보 및 유사 질문 답변을 바탕으로 다음 문의에 답변해 주세요. 제품명과 고객질문을 언급하지 말아주세요.
            제품명: {product_name}
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
            print(category_info)
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