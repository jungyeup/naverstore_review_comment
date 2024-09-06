import time
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
import os

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

    def run(self):
        comment_url = "https://sell.smartstore.naver.com/#/comment/"
        review_url = "https://sell.smartstore.naver.com/#/review/search"

        self.driver.get(comment_url)

        self.login_handler.login(
            '/html/body/div/div/div[1]/div/div/div[4]/div[1]/div/ul[1]/li[1]/input',
            '/html/body/div/div/div[1]/div/div/div[4]/div[1]/div/ul[1]/li[2]/input',
            '/html/body/div/div/div[1]/div/div/div[4]/div[1]/div/div/button'
        )
        time.sleep(10)

        while True:
            try:
                unanswered_comments_exist = self.process_comments()

                if not unanswered_comments_exist:
                    self.driver.get(review_url)
                    self.process_reviews()

                    self.driver.get(comment_url)
                time.sleep(30)
            except Exception as e:
                print(f"Error in main loop: {e}")
                time.sleep(10)

        self.driver.quit()

    def process_comments(self):
        self.driver.refresh()

        WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located(
                (By.XPATH, '/html/body/ui-view[1]/div[3]/div/div[4]/div/ui-view/div/div/div[2]/ui-view[2]/ul/li[1]')
            )
        )

        any_unanswered = False
        for i in range(1, 9):
            if self.question_handler.is_unanswered(i):
                self.question_handler.handle_question(
                    i,
                    f'/html/body/ui-view[1]/div[3]/div/div[4]/div/ui-view/div/div/div[2]/ui-view[2]/ul/li[{i}]/div[2]/div[1]/a',
                    [
                        f'/html/body/ui-view[1]/div[3]/div/div[4]/div/ui-view/div/div/div[2]/ui-view[2]/ul/li[{i}]/div[2]/p',
                        f'/html/body/ui-view[1]/div[3]/div/div[4]/div/ui-view[2]/ul/li[{i}]/div[2]/p'
                    ],
                    [
                        f'/html/body/ui-view[1]/div[3]/div/div[4]/div/ui-view[2]/ul/li[{i}]/div[2]/div[3]/button',
                        f'/html/body/ui-view[1]/div[3]/div/div[4]/div/ui-view[2]/ul/li[{i}]/div[2]/div[3]/button'
                    ],
                    [
                        f'/html/body/ui-view[1]/div[3]/div/div[4]/div/ui-view[2]/ul/li[{i}]/ncp-comment-reply/div/form/div/div[1]/div/textarea',
                        f'/html/body/ui-view[1]/div[3]/div/div[4]/div/ui-view/div/div/div[2]/ui-view[2]/ul/li[{i}]/ncp-comment-reply/div/form/div/div[1]/div/textarea'
                    ],
                    [
                        f'/html/body/ui-view[1]/div[3]/div/div[4]/div/ui-view[2]/ul/li[{i}]/ncp-comment-reply/div/form/div/div[1]/span/button',
                        f'/html/body/ui-view[1]/div[3]/div/div[4]/div/ui-view/div/div/div[2]/ui-view[2]/ul/li[{i}]/ncp-comment-reply/div/form/div/div[1]/span/button'
                    ],
                )
                any_unanswered = True
        return any_unanswered

    def process_reviews(self):
        consecutive_failures = 0
        max_consecutive_failures = 3

        for j in range(1, 501):
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

                self.question_handler.handle_review(
                    product_name_xpaths, review_xpaths, review_text_xpath, reply_textarea_xpath, reply_button_xpath
                )

                consecutive_failures = 0  # Reset failure count upon successful processing
            except Exception as e:
                print(f"Error processing review at row {j}: {e}")
                consecutive_failures += 1
                continue  # Skip to next review if an error occurs

if __name__ == "__main__":
    bot = NaverSmartStoreBot()
    bot.run()