from selenium import webdriver
from selenium.webdriver import ChromeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException
import os
import time
import pandas as pd
import numpy as np
import re
import sys
import google.protobuf.text_format as tf
from bareunpy import Tagger

def get_chrome_options() -> ChromeOptions:
    '''
    ChromeOptions을 반환하는 함수, configuration
    '''
    # 크롬 브라우저 설정을 위한 options 객체 생성
    options = webdriver.ChromeOptions()
    # 크롬 브라우저의 창 크기를 (1920 * 1080)으로 설정
    options.add_argument('--window-size=1920,1080')
    # 셀레니움 스크립트 실행 후에도 열려있도록 설정
    options.add_experimental_option("detach", True)
    # 크롬 브라우저가 직접적으로 열리지 않도록 설정
    options.add_argument('--headless')
    # 불필요한 이미지 로딩 없앰 (시간 단축)
    options.add_argument('--disable-logging')
    options.add_argument('--disable-images')
    
    user_agent=f'user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'
    options.add_argument(user_agent)

    return options


def recursive_visit_links(driver, url, depth, last_links, total_order, is_max_depth=True):
    """
    재귀적으로 URL을 탐색하면서 마지막 URL을 만났을 때 링크를
    반환하는 함수
    """

    s = time.time()
    driver.get(url)     # 링크 열기
    time.sleep(3)
    
    # 가장 먼저 <tr>요소에 접근
    tr_XPATH = r'//*[@id="content"]/section/table/tbody/tr'
    tr_elements = driver.find_elements(By.XPATH, tr_XPATH)

    # 각 요소의 href 속성값 가져오기
    # 구조를 살펴본 결과, <tr> -> <th> -> <tr>의 형식으로 접근해야 함
    url_links = []
    if depth > 2:
        for th_element in tr_elements:
            # 요소가 없을 경우, 예외 처리
            try:
                td_element = th_element.find_element(By.CSS_SELECTOR, "th")
                a_element = td_element.find_element(By.CSS_SELECTOR, "a")
                url_links.append(a_element.get_attribute("href"))
            except NoSuchElementException as e:
                pass
    else:
        for tr_element in tr_elements:
            # 요소가 없을 경우, 예외 처리
            try:
                td_element = tr_element.find_element(By.CSS_SELECTOR, "td")
                a_element = td_element.find_element(By.CSS_SELECTOR, "a")
                url_links.append(a_element.get_attribute("href"))
            except NoSuchElementException as e:
                pass
    
    # 최종 깊이에 도달하면 링크 반환 & 함수 종료
    if depth <= 1:
        # driver.quit()   # 웹 드라이버 종료
        last_links.append(url_links[1:])
        driver.back()
        return True

    # 전체 개수
    total_order = total_order + "-" + str(len(url_links))
    # 함수를 재귀적으로 돌기
    for i, url in enumerate(url_links, start=1):
        new_total_order = total_order + "|" + str(i)
        recursive_visit_links(driver, url, depth - 1, last_links, new_total_order, False)
        e = time.time()
        print(f"-------------------- {new_total_order} => Elapsed time: {e - s:.2f}s --------------------")

    if is_max_depth:
        driver.quit()


def txt_to_list(path):
    '''
    여러 줄의 txt 파일을 불러와서 list로 저장하는 함수
    '''
    link_list = []

    with open(path, "r") as f:
        lines = f.readlines()
    
    for line in lines:
        link_list.append(line.strip())
    return link_list


def double_to_single(double_list):
    '''
    이중 리스트를 단일 리스트로 만드는 함수
    '''
    return [item for sublist in double_list for item in sublist]


def worker(url, options, xpath):
    '''
    병렬 scraping을 위한 worker 함수. 비동기적 크롤링을 위한 함수입니다
    '''
    driver = webdriver.Chrome(options=options)
    driver.get(url)

    # 페이지가 완전히 로드될 때까지 대기 (예: body 태그가 나타날 때까지 대기)
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.TAG_NAME, "body")))

    # PATH 찾는 방법
    PATH = driver.find_element(By.XPATH, xpath[0]).text
    # TITLE 찾는 방법
    TITLE = driver.find_element(By.XPATH, xpath[1]).text
    # txt body 찾는 방법
    text_list = driver.find_elements(By.XPATH, xpath[2])
    TXT = ""
    for text in text_list:
        TXT += text.text + "\n"
    # txt desc 찾는 방법
    text_desc_list = driver.find_elements(By.XPATH, xpath[3])
    TXT_DESC = ""
    for text in text_desc_list:
        jusok_dt = text.find_element(By.CLASS_NAME, "jusok-dt").text
        jusok_dd = text.find_element(By.CLASS_NAME, "jusok-dd").text
        TXT_DESC += jusok_dt + jusok_dd + "\n"
    # current url
    URL = driver.current_url

    driver.quit()
    
    result = {"path": PATH,
              "title": TITLE,
              "text_body": TXT,
              "text_desc": TXT_DESC,
              "url": URL}
    return result


def split_sentences(passage, tagger):
    """
    문장을 여러 개로 분리하는 함수
    passage: 여러 개의 문장
    tagger: 바른 tagger
    """
    tagged = tagger.tags([passage], auto_split=True)
    
    sentences = []
    m = tagged.msg()
    for s in m.sentences:
        sentences.append(s.text.content)
    return sentences

def softmax(score_list):
    """
    Softmax 계산하는 함수
    """
    score_list = np.array(score_list)
    return np.exp(score_list) / np.sum(np.exp(score_list))

def log_normalize(score_list):
    """
    Log Normalize 변환
    """
    score_list = np.array(score_list)
    return np.log(1+score_list) / np.sum(np.log(1+score_list))

def normalize(score_list):
    """
    Normalize
    """
    score_list = np.array(score_list)
    return score_list / np.sum(score_list)