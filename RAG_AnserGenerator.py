import os
import pickle

import numpy as np
import pandas as pd
import PyPDF2
import docx
import pytesseract
from PIL import Image
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util


class AnswerGenerator:
    def __init__(self, openai_api_key, data_folder='data', vector_db_file='vector_db.pkl', processed_files_file='processed_files.pkl'):
        self.model = SentenceTransformer('sentence-transformers/xlm-r-large-en-ko-nli-ststb')
        self.client = OpenAI(api_key=openai_api_key)
        self.system_prompt = "기본 프롬프트 내용을 여기에 기재하세요."
        self.vector_db_file = vector_db_file
        self.processed_files_file = processed_files_file
        self.data_folder = data_folder
        self.df, self.all_files = self.load_all_data_files(data_folder)
        self.vector_db = self.load_vector_db()
        self.debug()

    def debug(self):
        print(f"모든 파일들: {self.all_files}")
        print(f"처리된 파일들: {self.load_processed_files()}")
        print(f"벡터 데이터베이스: {self.vector_db}")

    def load_all_data_files(self, folder_path):
        combined_df = pd.DataFrame()
        all_files = set()
        try:
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                if file_name.endswith('.xls') or file_name.endswith('.xlsx'):
                    df = pd.read_excel(file_path)
                elif file_name.endswith('.pdf'):
                    text = self.extract_text_from_pdf(file_path)
                    if text:
                        data = {'문의내용': text, '파일명': file_name, '답변내용': ''}
                        df = pd.DataFrame([data])
                elif file_name.endswith('.docx'):
                    q_text, a_text = self.extract_text_from_docx(file_path)
                    if q_text and a_text:
                        data = {'문의내용': q_text, '파일명': file_name, '답변내용': a_text}
                        df = pd.DataFrame([data])
                elif file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    text = self.extract_text_from_image(file_path)
                    if text:
                        data = {'문의내용': text, '파일명': file_name, '답변내용': ''}
                        df = pd.DataFrame([data])
                else:
                    continue

                df['파일명'] = file_name  # Add a column to keep track of file names
                combined_df = pd.concat([combined_df, df], ignore_index=True)
                all_files.add(file_name)
            print(f"총 파일 로드됨: {len(all_files)}")
            # Ensure '답변내용' column exists in the dataframe, even if it's empty.
            if '답변내용' not in combined_df.columns:
                combined_df['답변내용'] = ''
            return combined_df, all_files
        except Exception as e:
            print(f"파일 읽기 오류: {e}")
            raise

    def save_vector_db(self, vector_db):
        try:
            with open(self.vector_db_file, 'wb') as f:
                pickle.dump(vector_db, f)
            print("벡터 데이터베이스 저장 성공")
        except Exception as e:
            print(f"벡터 데이터베이스 저장 오류: {e}")
            raise

    def save_processed_files(self, processed_files):
        try:
            with open(self.processed_files_file, 'wb') as f:
                pickle.dump(processed_files, f)
            print("처리된 파일 목록 저장 성공")
        except Exception as e:
            print(f"처리된 파일 저장 오류: {e}")
            raise

    def load_processed_files(self):
        try:
            if os.path.exists(self.processed_files_file):
                with open(self.processed_files_file, 'rb') as f:
                    processed_files = pickle.load(f)
                    print(f"처리된 파일 불러오기: {processed_files}")
                    return processed_files
            else:
                print("처리된 파일 없음, 새 목록 생성")
                return set()
        except Exception as e:
            print(f"처리된 파일 로드 오류: {e}")
            raise

    def load_vector_db(self):
        try:
            processed_files = self.load_processed_files()
            if os.path.exists(self.vector_db_file) and processed_files == self.all_files:
                with open(self.vector_db_file, 'rb') as f:
                    vector_db = pickle.load(f)
                    print("벡터 데이터베이스 성공적으로 불러옴")
                    return vector_db
            else:
                return self.create_vector_db(processed_files)
        except Exception as e:
            print(f"벡터 데이터베이스 로드 오류: {e}")
            raise

    def create_vector_db(self, processed_files):
        try:
            new_files = self.all_files - processed_files
            if new_files:
                new_data = self.df[self.df['파일명'].isin(new_files)]
                questions_new = new_data['문의내용'].tolist()
                answers_new = new_data['답변내용'].tolist()

                embeddings_new = self.model.encode(questions_new, convert_to_tensor=True)
                embeddings_new = embeddings_new.cpu().numpy()

                if os.path.exists(self.vector_db_file):
                    vector_db_old = self.load_vector_db_from_saved()
                    vector_db_old['questions'].extend(questions_new)
                    vector_db_old['embeddings'] = np.concatenate([vector_db_old['embeddings'], embeddings_new])
                    vector_db_old['answers'].extend(answers_new)
                    vector_db = vector_db_old
                    print(f"새로운 데이터 추가: {len(questions_new)} 문서")
                else:
                    vector_db = {'questions': questions_new, 'embeddings': embeddings_new, 'answers': answers_new}
                    print(f"초기 데이터베이스 생성: {len(questions_new)} 문서")

                self.save_vector_db(vector_db)
                self.save_processed_files(self.all_files)
                return vector_db
            else:
                return self.load_vector_db_from_saved()
        except Exception as e:
            print(f"벡터 데이터베이스 생성 오류: {e}")
            raise

    def load_vector_db_from_saved(self):
        try:
            with open(self.vector_db_file, 'rb') as f:
                vector_db = pickle.load(f)
                return vector_db
        except Exception as e:
            print(f"벡터 데이터베이스 불러오기 오류: {e}")
            raise

    def retrieve_relevant_docs(self, question, top_n=5):
        try:
            print(f"Question: {question}")

            # Encode the question to get its embedding
            question_embedding = self.model.encode(question, convert_to_tensor=True)
            question_embedding = question_embedding.cpu().numpy()

            # Ensure the_embeddings are not empty
            if self.vector_db['embeddings'].size == 0:
                print("벡터 데이터베이스에 임베딩이 없습니다.")
                return [], []

            print(f"Vector DB Embeddings shape: {self.vector_db['embeddings'].shape}")
            similarities = util.cos_sim(np.array([question_embedding]), np.array(self.vector_db['embeddings']))
            similarities = similarities.numpy()  # Convert similarities to NumPy array for further processing
            print(f"Similarities: {similarities}")

            # Check if top_n is valid
            if top_n <= 0:
                print("유효한 top_n 값 아닙니다.")
                return [], []

            # Ensure there are enough similarities to process
            if similarities.shape[1] > 0:
                # Extract top N indices based on the highest similarity scores
                top_indices = np.argsort(similarities[0])[::-1][:top_n]
                print(f"Top Indices: {top_indices}")
                # Retrieve the relevant documents and their corresponding answers
                relevant_docs = []
                relevant_answers = []
                for idx in top_indices:
                    if idx < len(self.vector_db['questions']):
                        relevant_docs.append(self.vector_db['questions'][idx])
                        relevant_answers.append(self.vector_db['answers'][idx])
                print(f"Retrieved relevant docs: {relevant_docs}")
                print(f"Retrieved relevant answers: {relevant_answers}")
                return relevant_docs, relevant_answers
            else:
                print("비교할 임베딩이 없습니다.")
                return [], []
        except IndexError as e:
            print(f"IndexError 발생: {e}")
            print(f"similarities shape: {similarities.shape}")
            return [], []
        except Exception as e:
            print(f"문서 검색 오류: {e}")
            return [], []

    def extract_answer_from_docs(self, relevant_docs, question):
        try:
            # 문서 내에서 질문과 관련된 구절을 찾고 추출
            question_embedding = self.model.encode(question, convert_to_tensor=True)
            question_embedding = question_embedding.cpu().numpy()

            closest_match = None
            highest_similarity = -1

            for doc in relevant_docs:
                doc_embedding = self.model.encode(doc, convert_to_tensor=True)
                doc_embedding = doc_embedding.cpu().numpy()
                similarity = util.cos_sim(np.array([question_embedding]), np.array([doc_embedding])).item()

                if similarity > highest_similarity:
                    highest_similarity = similarity
                    closest_match = doc

            return closest_match
        except Exception as e:
            print(f"문서에서 답변 추출 오류: {e}")
            return "죄송합니다, 문서에서 답변을 추출할 수 없습니다."

    def generate_answer(self, question, ocr_texts, product_name):
        try:
            ocr_summary = " ".join(ocr_texts) if ocr_texts else "해당 제품에 대한 이미지에서 정보를 추출하지 않았습니다."
            image_summary = self.get_image_summary(ocr_summary) if ocr_texts else ocr_summary
            relevant_docs, relevant_answers = self.retrieve_relevant_docs(question)  # Now this returns both docs and answers

            relevant_info = "\n".join(relevant_docs)
            reference_answers = "\n".join(relevant_answers)

            category_info = self.classify_question(question)
            category = self.extract_category_from_classification(category_info)
            specific_prompt = self.get_specific_prompt(category)

            extracted_answer = self.extract_answer_from_docs(relevant_docs, question)

            answer_prompt = f"""
            제공된 제품 정보 및 관련 문서를 바탕으로 다음 문의에 답변해 주세요. 제품명과 고객질문을 언급하지 말아주세요.
            제품명: {product_name}
            고객 질문: {question}
            관련 문서: {relevant_info}
            참고 답변: {reference_answers}
            제품 정보: {image_summary}
            {specific_prompt}
            문서에서 추출된 답변: {extracted_answer}
            """

            chat_completion = self.client.chat.completions.create(
                model="gpt-4o",
                temperature=0.8,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": answer_prompt}
                ]
            )

            return chat_completion.choices[0].message.content
        except Exception as e:
            print(f"답변 생성 오류: {e}")
            return "죄송합니다, 답변을 생성할 수 없습니다."

    def get_image_summary(self, ocr_summary):
        try:
            image_info_prompt = f"""
            다음 제품 정보를 요약합니다: ...(특히 숫자, 치수, 사이즈, 스펙 그리고 특징 등을 자세히 기록합니다)
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
        categories = ["제품 리뷰", "재고/입고/배송", "AS (수리, 반품, 교환)", "사용법/호환성", "제품 문의", "할인 문의", "기술 지원", "기타 질문"]
        for category in categories:
            if category in category_info:
                return category
        return "기타 질문"

    def get_specific_prompt(self, category):
        category_prompts = {
            "제품 리뷰": "제품 리뷰에 대한 구체적인 내용을 여기에 기재하세요.",
            "재고/입고/배송": "재고/입고/배송에 대한 구체적인 내용을 여기에 기재하세요.",
            "AS (수리, 반품, 교환)": "AS (수리, 반품, 교환)에 대한 구체적인 내용을 여기에 기재하세요.",
            "사용법/호환성": "사용법/호환성에 대한 구체적인 내용을 여기에 기재하세요.",
            "제품 문의": "제품 문의에 대한 구체적인 내용을 여기에 기재하세요.",
            "할인 문의": "할인 문의에 대한 구체적인 내용을 여기에 기재하세요.",
            "기술 지원": "기술 지원에 대한 구체적인 내용을 여기에 기재하세요.",
            "기타 질문": "기타 질문에 대한 구체적인 내용을 여기에 기재하세요."
        }
        return category_prompts.get(category, "")

    def extract_text_from_pdf(self, file_path):
        texts = []
        try:
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfFileReader(f)
                for page_num in range(reader.numPages):
                    page = reader.getPage(page_num)
                    texts.append(page.extract_text())
            return " ".join(texts)
        except Exception as e:
            print(f"PDF 텍스트 추출 오류: {e}")
            return None

    def extract_text_from_image(self, file_path):
        try:
            img = Image.open(file_path)
            text = pytesseract.image_to_string(img, lang='kor+eng')  # Explicitly setting the language
            return text
        except Exception as e:
            print(f"이미지 텍스트 추출 오류: {e}")
            return None

    def extract_text_from_docx(self, file_path):
        q_text = ""
        a_text = ""
        try:
            doc = docx.Document(file_path)
            for para in doc.paragraphs:
                if para.text.startswith("Q:"):
                    q_text += para.text[2:].strip() + " "
                elif para.text.startswith("A:"):
                    a_text += para.text[2:].strip() + " "
            return q_text.strip(), a_text.strip()
        except Exception as e:
            print(f"DOCX 텍스트 추출 오류: {e}")
            return None, None