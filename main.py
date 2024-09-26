import time
import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

from ConfigurationHandler import ConfigurationHandler
from LoginHandler import LoginHandler
from ocr_handler import OCRHandler
from AnswerGenerator import AnswerGenerator
from QuestionHandler import QuestionHandler
from report_generator import ReportGenerator


class NaverSmartStoreBot:
    def __init__(self):
        self.config = ConfigurationHandler.load_environment_variables()

        if not self.config['OPENAI_API_KEY'] or not self.config['USER_ID'] or not self.config['PASSWORD']:
            raise ValueError("API key, user ID, or password not set in environment variables")

        chrome_options = Options()
        chrome_options.add_argument("--start-fullscreen")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")

        self.driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=chrome_options)
        self.driver.maximize_window()

        self.login_handler = LoginHandler(self.driver, self.config['USER_ID'], self.config['PASSWORD'])
        self.ocr_handler = OCRHandler()

        data_folder = os.path.join(os.getcwd(), 'data')
        self.answer_generator = AnswerGenerator(self.config['OPENAI_API_KEY'], data_folder=data_folder)

        self.question_handler = QuestionHandler(self.driver, self.ocr_handler, self.answer_generator)

        self.report_generator = ReportGenerator(self.config['OPENAI_API_KEY'])

    def run(self):
        comment_url = "https://sell.smartstore.naver.com/#/comment/"
        review_url = "https://sell.smartstore.naver.com/#/review/search"

        self.driver.get(comment_url)

        self.login_handler.login(
            '/html/body/div/div/div[1]/div/div/div[4]/div[1]/div/ul[1]/li[1]/input',  # id_xpath
            '/html/body/div/div/div[1]/div/div/div[4]/div[1]/div/ul[1]/li[2]/input',  # password_xpath
            '/html/body/div/div/div[1]/div/div/div[4]/div[1]/div/div/button'  # login_button_xpath
        )
        time.sleep(2)

        try:
            store_select_button = self.driver.find_element(By.XPATH, '//a[@role="button" and @ui-sref="work.channel-select" and @data-action-location-id="selectStore"]')
            store_select_button.click()
            time.sleep(1)
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, 'seller-list-scroll'))
            )

            kazmi_store_option = self.driver.find_element(By.XPATH, '//label[.//span[contains(text(), "카즈미")]]')
            time.sleep(1)
            kazmi_store_option.click()
            time.sleep(2)
        except Exception as e:
            print(f"Error during initial store selection: {e}")

        daily_report_data = []

        while True:
            try:
                unanswered_comments_exist = self.process_comments(daily_report_data)

                if not unanswered_comments_exist:
                    self.driver.get(review_url)
                    self.process_reviews(daily_report_data)

                    self.driver.get(comment_url)
                time.sleep(3)
            except Exception as e:
                daily_report_data.append({
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                    'type': "Error",
                    'url': self.driver.current_url,
                    'error_message': str(e)
                })
                print(f"Error in main loop: {e}")
                time.sleep(1)

        self.report_generator.generate_reports(daily_report_data)

    def process_comments(self, report_data):
        self.driver.refresh()

        WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located(
                (By.XPATH, '/html/body/ui-view[1]/div[3]/div/div[4]/div/ui-view/div/div/div[2]/ui-view[2]/ul/li[1]')
            )
        )

        any_unanswered = False
        for i in range(1, 9):
            if self.question_handler.is_unanswered(i):
                comment_time_xpath = f'/html/body/ui-view[1]/div[3]/div/div[4]/div/ui-view/div/div/div[2]/ui-view[2]/ul/li[{i}]/div[2]/div[2]/span[4]'
                
                comment_data = self.question_handler.handle_question(
                    i,
                    f'/html/body/ui-view[1]/div[3]/div/div[4]/div/ui-view/div/div/div[2]/ui-view[2]/ul/li[{i}]/div[2]/div[1]/a',  # product_name_xpath
                    [
                        f'/html/body/ui-view[1]/div[3]/div/div[4]/div/ui-view/div/div/div[2]/ui-view[2]/ul/li[{i}]/div[2]/p',  # question_xpaths
                        f'/html/body/ui-view[1]/div[3]/div/div[4]/div/ui-view[2]/ul/li[{i}]/div[2]/p'
                    ],
                    [
                        f'/html/body/ui-view[1]/div[3]/div/div[4]/div/ui-view[2]/ul/li[{i}]/div[2]/div[3]/button',  # click_to_answer_xpaths
                        f'/html/body/ui-view[1]/div[3]/div/div[4]/div/ui-view/div/div/div[2]/ui-view[2]/ul/li[{i}]/div[2]/div[3]/button'
                    ],
                    [
                        f'/html/body/ui-view[1]/div[3]/div/div[4]/div/ui-view[2]/ul/li[{i}]/ncp-comment-reply/div/form/div/div[1]/div/textarea',  # typing_area_xpaths
                        f'/html/body/ui-view[1]/div[3]/div/div[4]/div/ui-view/div/div/div[2]/ui-view[2]/ul/li[{i}]/ncp-comment-reply/div/form/div/div[1]/div/textarea'
                    ],
                    [
                        f'/html/body/ui-view[1]/div[3]/div/div[4]/div/ui-view[2]/ul/li[{i}]/ncp-comment-reply/div/form/div/div[1]/span/button',  # upload_button_xpaths
                        f'/html/body/ui-view[1]/div[3]/div/div[4]/div/ui-view/div/div/div[2]/ui-view[2]/ul/li[{i}]/ncp-comment-reply/div/form/div/div[1]/span/button'
                    ],
                    comment_time_xpath  # New argument for comment time
                )

                report_data.append(comment_data)
                any_unanswered = True
        return any_unanswered

    def get_total_review_count(self):
        """ Get the total number of reviews available. """
        try:
            total_reviews_element = self.driver.find_element(By.XPATH, '//h3[@class="panel-title"]/span[@class="text-primary"]')
            total_reviews = int(total_reviews_element.text)
            return total_reviews
        except Exception as e:
            print(f"Error finding total reviews count: {e}")
            return 0

    def process_reviews(self, report_data):
        self.scroll_down_page()

        consecutive_failures = 0
        max_consecutive_failures = 3

        total_reviews = self.get_total_review_count()
        print(total_reviews)

        for j in range(total_reviews + 1):  # 첫 번째 리뷰도 포함하기 위해 0부터 시작
            if consecutive_failures >= max_consecutive_failures:
                print(f"Too many consecutive failures. Returning to comment processing.")
                return

            try:
                # Identifying XPaths for the required elements
                product_name_xpaths = [
                    f'//div[@row-index="{j}"]//div[@col-id="productName"]/span/a'
                ]

                review_xpaths = [
                    f'//div[@row-index="{j}"]//div[@col-id="reviewContent"]/span/a'
                ]

                review_text_xpath = f'//div[@row-index="{j}"]//div[@col-id="reviewContent"]/span/a/div'

                reply_textarea_xpath = '//textarea[@placeholder="반복적인 답글이 아닌 정성스러운 답글을 남겨주세요. 낮은 평점의 리뷰에도 귀 기울여 진심을 담아 구매자와 소통해주시면 스토어 만족도가 높아집니다.^^"]'

                reply_button_xpath = '//button[contains(@class, "btn btn-xs btn-default progress-button progress-button-dir-horizontal progress-button-style-top-line") and @ng-if="!vm.data.reviewComment || !vm.data.reviewComment.commentId"]'

                review_data = self.question_handler.handle_review(
                    product_name_xpaths, review_xpaths, review_text_xpath, reply_textarea_xpath, reply_button_xpath
                )
                report_data.append(review_data)

                consecutive_failures = 0  # Reset failure count upon successful processing
            except Exception as e:
                error_data = {
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                    'type': "Error",
                    'row_index': j,
                    'error_message': str(e)
                }
                report_data.append(error_data)
                print(f"Error processing review at row {j}: {e}")
                consecutive_failures += 1
                continue  # Skip to next review if an error occurs

    def scroll_down_page(self):
        """ Scroll down the page to load all reviews. """
        last_height = self.driver.execute_script("return document.body.scrollHeight")

        while True:
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

            # Wait to load the page
            time.sleep(1)

            # Calculate new scroll height and compare with last scroll height
            new_height = self.driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height


if __name__ == "__main__":
    bot = NaverSmartStoreBot()
    bot.run()