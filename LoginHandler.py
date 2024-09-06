from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

class LoginHandler:
    def __init__(self, driver, user_id, password):
        self.driver = driver
        self.user_id = user_id
        self.password = password

    def login(self, id_xpath, password_xpath, login_button_xpath):
        try:
            WebDriverWait(self.driver, 10).until(EC.element_to_be_clickable((By.XPATH, id_xpath))).send_keys(self.user_id)
            time.sleep(1)
            self.driver.find_element(By.XPATH, password_xpath).send_keys(self.password)
            time.sleep(1)
            self.driver.find_element(By.XPATH, login_button_xpath).click()
        except Exception as e:
            print(f"Error during login: {e}")