import os
import platform
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException, WebDriverException
import pandas as pd
import time
import ast

class QuestionHandler:
    def __init__(self, driver, ocr_handler, answer_generator, report_generator):
        self.driver = driver
        self.ocr_handler = ocr_handler
        self.answer_generator = answer_generator
        self.report_generator = report_generator
        self.excel_file_path = "data/questions_answers.xlsx"
        self.df = pd.read_excel(self.excel_file_path) if os.path.exists(self.excel_file_path) else pd.DataFrame(columns=["작성시간", "상품명", "문의내용", "답변내용", "수정내용", "OCR내용", "특이사항"])

    def is_unanswered(self, i):
        label_xpaths = [
            f'/html/body/ui-view[1]/div[3]/div/div[4]/div/ui-view/div/div/div[2]/ui-view[2]/ul/li[{i}]/div[2]/div[1]/span[1]',
            f'/html/body/ui-view[1]/div[3]/div/div[4]/div/ui-view[2]/ul/li[{i}]/div[2]/div[1]/span[1]'
        ]

        for label_xpath in label_xpaths:
            try:
                label = self.driver.find_element(By.XPATH, label_xpath)
                label_text = label.text.strip()
                if label_text == "미답변":
                    print(f"Question {i} is unanswered.")
                    return True
                elif label_text == "답변완료":
                    print(f"Question {i} is already answered.")
                    return False
                else:
                    print(f"Unexpected label text for question {i}: {label_text}")
                    return False
            except NoSuchElementException:
                print(f"Element not found for XPath: {label_xpath}")
            except Exception as e:
                print(f"Error checking unanswered status for question {i} with XPath {label_xpath}: {e}")
        return False

    def scroll_down_fully(self, increment=1000, pause_time=1):
        """Scroll down the webpage incrementally to ensure all content is loaded."""
        last_height = self.driver.execute_script("return document.body.scrollHeight")

        while True:
            self.driver.execute_script(f"window.scrollBy(0, {increment});")
            time.sleep(pause_time)
            new_height = self.driver.execute_script("return document.body.scrollHeight")

            if new_height == last_height:
                break

            last_height = new_height

        while True:
            self.driver.execute_script(f"window.scrollBy(0, {increment});")
            time.sleep(pause_time)

            WebDriverWait(self.driver, 10).until(lambda driver: driver.execute_script("return document.readyState") == "complete")

            new_height = self.driver.execute_script("return document.body.scrollHeight")

            if new_height == last_height:
                break

            last_height = new_height

    def extract_and_ocr_images(self):
        """Extract images and perform OCR on them."""
        try:
            def collect_ocr_summaries():
                self.scroll_down_fully()
                ocr_summaries = []

                html_content = self.driver.page_source
                ocr_summaries = self.ocr_handler.ocr_from_html_content(html_content)

                return ocr_summaries

            return collect_ocr_summaries()
        except Exception as e:
            print(f"Error scrolling and collecting OCR summaries: {e}")
            return ["Failed to collect OCR summaries"]

    def dismiss_popup(self):
        """Dismiss any open popup windows."""
        try:
            popup_close_selectors = [
                "//button[contains(text(), 'OK')]",
                "//button[contains(text(), '확인')]",
                "//button[contains(text(), '닫기')]",
                "//button[@type='button' and @class='close' and @data-dismiss='modal' and @ng-click='vm.func.closeModal()' and @aria-label='닫기']/span[@aria-hidden='true']"
            ]
            for selector in popup_close_selectors:
                try:
                    popup_close_button = WebDriverWait(self.driver, 5).until(EC.element_to_be_clickable((By.XPATH, selector)))
                    if popup_close_button:
                        popup_close_button.click()
                        print("Popup closed.")
                        return True
                except TimeoutException:
                    continue
                except Exception as e:
                    print(f"Could not close popup with selector {selector}: {e}")
            print("No popups to close or unhandled popup.")
            return False
        except Exception as e:
            print(f"Error checking or closing popup: {e}")
            return False

    def get_ocr_summaries_for_product(self, product_name):
        """Get OCR summaries for a given product name."""
        if not self.df.empty:
            existing_entry = self.df[self.df['상품명'] == product_name]
            if not existing_entry.empty:
                ocr_summaries = existing_entry.iloc[0]['OCR내용']
                print(f"Using existing OCR summaries for product: {product_name}")
                try:
                    return ast.literal_eval(ocr_summaries)
                except Exception as e:
                    print(f"Error parsing OCR summaries for product: {product_name}. Details: {e}")
                    return []

        print("No existing data found in the Excel file.")

        ocr_summaries = self.extract_and_ocr_images()
        if 'Failed to collect OCR summaries' in ocr_summaries:
            print(f"Could not collect OCR summaries for product: {product_name}")
            ocr_summaries = []
        return ocr_summaries

    def handle_question(self, i, product_name_xpath, question_xpaths, click_to_answer_xpaths, typing_area_xpaths, upload_button_xpaths, comment_time_xpath):
        """Handles an individual question."""
        try:
            question = None
            for xpath in question_xpaths:
                try:
                    question_element = self.driver.find_element(By.XPATH, xpath)
                    question = question_element.text
                    break
                except NoSuchElementException:
                    continue

            if not question:
                print(f"Could not find question element for index {i}")
                return

            product_name_element = WebDriverWait(self.driver, 10).until(EC.presence_of_element_located((By.XPATH, product_name_xpath)))
            product_name = product_name_element.text if product_name_element else "해당 제품"

            comment_time_element = self.driver.find_element(By.XPATH, comment_time_xpath)
            comment_time = comment_time_element.text

            product_num_element = self.driver.find_element(By.XPATH, product_name_xpath)
            self.driver.execute_script("arguments[0].removeAttribute('target')", product_num_element)
            product_num_element.click()
            time.sleep(1)

            scroll_button_xpaths = [
                '//button[text()="상세정보 펼쳐보기"]',
                '//button[@class="_1gG8JHE9Zc _nlog_click"][@data-shp-page-key="100329229"][@data-shp-area="detailitm.more"]',
                '//button[@class="_1gG8JHE9Zc _nlog_click" and @data-shp-page-key="100356693" and @data-shp-area="detailitm.more"]'
            ]

            try:
                self.scroll_down_fully()  # Scroll down the page fully before trying to find the scroll button

                scroll_button = None
                for xpath in scroll_button_xpaths:
                    try:
                        scroll_button = WebDriverWait(self.driver, 10).until(EC.element_to_be_clickable((By.XPATH, xpath)))
                        if scroll_button:
                            self.driver.execute_script("arguments[0].scrollIntoView(true);", scroll_button)
                            time.sleep(1)
                            scroll_button.click()
                            time.sleep(1)
                            self.scroll_down_fully()

                            WebDriverWait(self.driver, 10).until(lambda driver: driver.execute_script("return document.readyState") == "complete")

                            break
                    except Exception as e:
                        continue

                if not scroll_button:
                    print("Scroll button not found or not clickable with provided XPaths.")
            except Exception as e:
                print(f"Scroll button processing error: {e}")

            ocr_summaries = self.get_ocr_summaries_for_product(product_name)

            print(f"Extracted OCR summaries: {ocr_summaries}")

            current_time = time.strftime("%Y-%m-%d %H:%M:%S")

            answer = self.answer_generator.generate_answer(question, ocr_summaries, product_name, comment_time, current_time)
            modification_note = "자동 생성된 답변"
            status = "Generated"
            special_note = ""  # Initialize special note

            if not answer:
                similar_question_data = self.answer_generator.find_similar_question(question)
                if similar_question_data is None:
                    answer = "죄송합니다. 데이터베이스에서 유사한 질문을 찾을 수 없습니다."
                else:
                    answer = f"아래 유사 질문의 답변을 참고해 주세요:\n\nQ: {similar_question_data['문의내용']}\nA: {similar_question_data['답변내용']} 내용."
                modification_note = "유사한 질문에 대한 답변으로 대체됨"

            original_answer = answer
            print("\n\nGenerated Answer:", original_answer)

            self.beep_sound()

            special_note = ""

            while True:
                print(f"\n\n==================================================\n상품명:\n{product_name}에 대한 \n\n질문:\n'{question}'의 답변입니다.\n\n")
                user_input = input("답변을 수정하거나 추가하십시오. 업로드를 취소하려면 '취소' 혹은 'c'를 입력하고, 재생성을 원하시면 '재생성' 혹은 're', 그대로 두려면 Enter를 누르세요: ").strip().lower()
                if user_input in ['c', '취소']:
                    print("업로드가 취소되었습니다.")
                    self.driver.back()
                    time.sleep(5)
                    
                    # Add cancellation report data
                    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
                    report_data = {
                        'type': 'Comment',
                        'timestamp': current_time,
                        'product_name': product_name,
                        'question': question,
                        'answer': "N/A",
                        'ocr_summaries': str(ocr_summaries),
                        'status': "Cancelled",
                        'notes': "User cancelled the upload."
                    }
                    self.record_report(report_data)
                    
                    return
                if user_input in ['re', '재생성']:
                    answer = self.answer_generator.generate_answer(question, ocr_summaries, product_name, comment_time, current_time)
                    original_answer = answer
                    print("Regenerated Answer:", answer)
                    modification_note = "자동 생성된 답변 (재생성)"
                    
                    # Add regeneration report data
                    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
                    report_data = {
                        'type': 'Comment',
                        'timestamp': current_time,
                        'product_name': product_name,
                        'question': question,
                        'answer': answer,
                        'ocr_summaries': str(ocr_summaries),
                        'status': "Regenerated",
                        'notes': "User requested to regenerate the answer."
                    }
                    self.record_report(report_data)

                elif user_input:
                    answer = self.answer_generator.revise_answer(user_input, original_answer)
                    print("Revised Answer:", answer)
                    modification_note = f"수정된 답변: {user_input}"
                    
                    # Add revision report data
                    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
                    report_data = {
                        'type': 'Comment',
                        'timestamp': current_time,
                        'product_name': product_name,
                        'question': question,
                        'answer': answer,
                        'ocr_summaries': str(ocr_summaries),
                        'status': "Revised",
                        'notes': f"User revised the answer manually with input: {user_input}"
                    }
                    self.record_report(report_data)

                else:
                    break

            current_time = time.strftime("%Y-%m-%d %H:%M:%S")
            new_entry = pd.DataFrame({
                "작성시간": [current_time],
                "상품명": [product_name],
                "문의내용": [question],
                "답변내용": [answer],
                "수정내용": [modification_note],
                "OCR내용": [str(ocr_summaries)],
                "특이사항": [special_note]
            })
            self.df = pd.concat([self.df, new_entry], ignore_index=True)
            self.save_to_excel()

            self.driver.back()
            time.sleep(5)

            for xpath in click_to_answer_xpaths:
                try:
                    answer_button = self.driver.find_element(By.XPATH, xpath)
                    answer_button.click()
                    time.sleep(1)
                    break
                except NoSuchElementException:
                    continue
                except Exception as e:
                    print(f"Could not click answer button with XPath {xpath}: {e}")
            time.sleep(1)
            typing_area = None
            for xpath in typing_area_xpaths:
                try:
                    typing_area = WebDriverWait(self.driver, 10).until(EC.element_to_be_clickable((By.XPATH, xpath)))
                    typing_area.send_keys(answer)
                    break
                except NoSuchElementException:
                    continue
                except Exception as e:
                    print(f"Could not find typing area with XPath {xpath}: {e}")
            time.sleep(1)
            for xpath in upload_button_xpaths:
                try:
                    upload_button = self.driver.find_element(By.XPATH, xpath)
                    upload_button.click()
                    self.dismiss_popup()
                    answer_button.click()
                    print(f"Answered question {i}")
                    break
                except NoSuchElementException:
                    continue
                except Exception as e:
                    print(f"Could not find upload button with XPath {xpath}: {e}")

            self.record_report({
                'type': 'Comment',
                'timestamp': current_time,
                'product_name': product_name,
                'question': question,
                'answer': answer,
                'ocr_summaries': str(ocr_summaries),
                'status': status,
                'notes': special_note  # Add special notes to report data
            })

        except WebDriverException as e:
            error_message = f"Selenium-related error answering question {i} (not included in report): {e}"
            print(error_message)
            error_report = {
                'type': 'Comment',
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'product_name': 'N/A',
                'question': f"Error processing question {i}",
                'answer': 'N/A',
                'ocr_summaries': 'N/A',
                'status': 'Error',
                'error_message': error_message
            }
            self.record_report(error_report)
            self.driver.back()
            time.sleep(2)
            return None

        except Exception as e:
            error_message = f"Error answering question {i}: {e}"
            print(error_message)
            error_report = {
                'type': 'Comment',
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'product_name': 'N/A',
                'question': f"Error processing question {i}",
                'answer': 'N/A',
                'ocr_summaries': 'N/A',
                'status': 'Error',
                'error_message': error_message
            }
            self.record_report(error_report)
            self.driver.back()
            time.sleep(2)
            return error_report

    def handle_review(self, product_name_xpaths, review_xpaths, review_text_xpath, reply_textarea_xpath, reply_button_xpath):
        """Handles an individual review."""
        try:
            self.scroll_down_fully()

            product_name_element = None
            for product_name_xpath in product_name_xpaths:
                try:
                    product_name_element = self.driver.find_element(By.XPATH, product_name_xpath)
                    product_name = product_name_element.text if product_name_element else "해당 제품"
                    break
                except NoSuchElementException:
                    continue

            if not product_name_element:
                print("Product name element not found with provided XPaths.")
                return

            review_element = None
            for review_xpath in review_xpaths:
                try:
                    review_element = self.driver.find_element(By.XPATH, review_xpath)
                    review_element.click()
                    time.sleep(1)
                    break
                except NoSuchElementException:
                    continue

            if not review_element:
                print("Review element not found with provided XPaths.")
                return

            review_text_element = self.driver.find_element(By.XPATH, review_text_xpath)
            review_text = review_text_element.text

            reply_textarea = self.driver.find_element(By.XPATH, reply_textarea_xpath)
            placeholder = reply_textarea.getAttribute('placeholder')

            if placeholder != "반복적인 답글이 아닌 정성스러운 답글을 남겨주세요. 낮은 평점의 리뷰에도 귀 기울여 진심을 담아 구매자와 소통해주시면 스토어 만족도가 높아집니다.^^":
                print("Reply already exists. Moving to the next review.")
                self.driver.findElement(By.XPATH, '//button[@type="button" and @class="close" and @data-dismiss="modal" and @ng-click="vm.func.closeModal()"]/span[@aria-hidden="true"]').click()
                return

            print(f"Review text: {review_text}")

            try:
                reply_button = self.driver.findElement(By.XPATH, reply_button_xpath)
            except NoSuchElementException:
                print("Reply button not found. This review might already have a reply.")
                self.driver.findElement(By.XPATH, '//button[@type="button" and @class="close" and @data-dismiss="modal" and @ng-click="vm.func.closeModal()"]/span[@aria-hidden="true"]').click()
                return

            ocr_summaries = self.get_ocr_summaries_for_product(product_name)

            answer = self.answer_generator.generate_answer(review_text, ocr_summaries, product_name)
            modification_note = "자동 생성된 답글"
            status = "Generated"
            special_note = ""  # Initialize special note

            original_answer = answer
            print("Generated Answer:", original_answer)

            self.beep_sound()

            while True:
                print(f"\n\n==================================================\n상품명:\n{product_name}에 대한 \n\n리뷰:\n'{review_text}'의 답글입니다.\n\n")
                user_input = input("답글을 수정하거나 추가하십시오. 업로드를 취소하려면 '취소' 혹은 'c'를 입력하고, 재생성을 원하시면 '재생성' 혹은 're', 그대로 두려면 Enter를 누르세요:").strip().lower()
                if user_input in ['c', '취소']:
                    print("업로드가 취소되었습니다.")
                    self.driver.find_element(By.XPATH, '//button[@type="button" and @class="close" and @data-dismiss="modal" and @ng-click="vm.func.closeModal()"]/span[@aria-hidden="true"]').click()
                    
                    # Add cancellation report data
                    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
                    report_data = {
                        'type': 'Review',
                        'timestamp': current_time,
                        'product_name': product_name,
                        'question': review_text,
                        'answer': "N/A",
                        'ocr_summaries': str(ocr_summaries),
                        'status': "Cancelled",
                        'notes': "User cancelled the upload."
                    }
                    self.record_report(report_data)

                    return
                if user_input in ['re', '재생성']:
                    answer = self.answer_generator.generate_answer(review_text, ocr_summaries, product_name)
                    original_answer = answer
                    print("Regenerated Answer:", answer)
                    modification_note = "자동 생성된 답글 (재생성)"
                    
                    # Add regeneration report data
                    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
                    report_data = {
                        'type': 'Review',
                        'timestamp': current_time,
                        'product_name': product_name,
                        'question': review_text,
                        'answer': answer,
                        'ocr_summaries': str(ocr_summaries),
                        'status': "Regenerated",
                        'notes': "User requested to regenerate the answer."
                    }
                    self.record_report(report_data)
                    
                elif user_input:
                    answer = self.answer_generator.revise_answer(user_input, original_answer)
                    print("Revised Answer:", answer)
                    modification_note = f"수정된 답글: {user_input}"
                    
                    # Add revision report data
                    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
                    report_data = {
                        'type': 'Review',
                        'timestamp': current_time,
                        'product_name': product_name,
                        'question': review_text,
                        'answer': answer,
                        'ocr_summaries': str(ocr_summaries),
                        'status': "Revised",
                        'notes': f"User revised the answer manually with input: {user_input}"
                    }
                    self.record_report(report_data)
                    
                else:
                    break

            current_time = time.strftime("%Y-%m-%d %H:%M:%S")

            new_entry = pd.DataFrame({
                "작성시간": [current_time],
                "상품명": [product_name],
                "문의내용": [review_text],
                "답변내용": [answer],
                "수정내용": [modification_note],
                "OCR내용": [str(ocr_summaries)],
                "특이사항": [special_note]
            })
            self.df = pd.concat([self.df, new_entry], ignore_index=True)
            self.save_to_excel()

            reply_textarea.send_keys(answer)

            try:
                reply_button.click()
                time.sleep(1)
            except NoSuchElementException:
                print("Reply button not found. This review might already have a reply.")
                self.driver.find_element(By.XPATH, '//button[@type="button" and @class="close" and @data-dismiss="modal" and @ng-click="vm.func.closeModal()"]/span[@aria-hidden="true"]').click()
                return
            
            self.dismiss_popup()

            print(f"Replied to review with text: {answer}")

            new_entry = pd.DataFrame({
                "작성시간": [current_time],
                "상품명": [product_name],
                "문의내용": [review_text],
                "답변내용": [answer],
                "수정내용": [modification_note],
                "OCR내용": [str(ocr_summaries)],
                "특이사항": [special_note]
            })
            self.df = pd.concat([self.df, new_entry], ignore_index=True)
            self.save_to_excel()

            report_data = {
                'type': 'Review',
                'timestamp': current_time,
                'product_name': product_name,
                'question': review_text,
                'answer': answer,
                'ocr_summaries': str(ocr_summaries),
                'status': status,
                'notes': special_note  # Add special notes to report data
            }

            # Save report after handling each review
            self.report_generator.generate_reports([report_data])
            return report_data

        except WebDriverException as e:
            error_message = f"Selenium-related error handling review (not included in report): {e}"
            print(error_message)
            error_report = {
                'type': 'Review',
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'product_name': 'N/A',
                'question': "Error processing review",
                'answer': 'N/A',
                'ocr_summaries': 'N/A',
                'status': 'Error',
                'error_message': error_message
            }
            self.record_report(error_report)
            self.driver.find_element(By.XPATH, '//button[@type="button" and @class="close" and @data-dismiss="modal" and @ng-click="vm.func.closeModal()"]/span[@aria-hidden="true"]').click()
            return None

        except Exception as e:
            error_message = f"Error handling review: {e}"
            print(error_message)
            error_report = {
                'type': 'Review',
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'product_name': 'N/A',
                'question': "Error processing review",
                'answer': 'N/A',
                'ocr_summaries': 'N/A',
                'status': 'Error',
                'error_message': error_message
            }
            self.record_report(error_report)
            self.driver.find_element(By.XPATH, '//button[@type="button" and @class="close" and @data-dismiss="modal" and @ng-click="vm.func.closeModal()"]/span[@aria-hidden="true"]').click()
            return error_report

    def record_report(self, report):
        """Record a report (both successful and errors) in the DataFrame and save to Excel."""
        self.df = pd.concat([self.df, pd.DataFrame([report])], ignore_index=True)
        self.save_to_excel()
        self.report_generator.generate_reports([report])

    def beep_sound(self):
        """Produce a beep sound."""
        if platform.system() == "Windows":
            import winsound
            winsound.MessageBeep()
        elif platform.system() == "Darwin":
            os.system('say "beep"')
        else:
            print('\a')

    def save_to_excel(self):
        """Save the DataFrame to an Excel file."""
        self.df.to_excel(self.excel_file_path, index=False)
        print(f"Saved questions and answers to {self.excel_file_path}")