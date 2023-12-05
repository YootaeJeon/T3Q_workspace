# inference_service_sub.py

import os
import re

import logging
from PIL import Image

from t3qai_client import DownloadFile
import t3qai_client as tc
from t3qai_client import T3QAI_TRAIN_OUTPUT_PATH, T3QAI_TRAIN_MODEL_PATH, T3QAI_TRAIN_DATA_PATH, T3QAI_TEST_DATA_PATH, T3QAI_MODULE_PATH, T3QAI_INIT_MODEL_PATH

import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

plt.rc('font', family='Malgun Gothic') 

# 저장 파일 확인
def list_files_directories(path):
    # Get the list of all files and directories in current working directory
    dir_list = os.listdir(path)
    logging.info('[search log] ★ Files and directories in {} :'.format(path))
    logging.info('[search log] ★ dir_list : {}'.format(dir_list))  

def exec_init_model():
    logging.info('[search log] the start line of the function [exec_init_model]')
    
    list_files_directories(T3QAI_INIT_MODEL_PATH)
 
    model_path = os.path.join(T3QAI_INIT_MODEL_PATH, 'topkeyword.pkl')
    logging.info(f'[search log] 모델불러오는 경로 확인 : {model_path}')

    model = pd.read_pickle(model_path) # tfidf_df
    model_info_dict = {
        "model": model

    }
    return model_info_dict

def remove_stopwords_from_list(content_list, stopwords):
    """Remove stopwords"""
    filtered_list = []

    filtered_sentence = " ".join([word for word in content_list.split() if word not in stopwords])
    filtered_list.append(filtered_sentence)
    
    return filtered_list

# 요약 문장 전처리 함수 

def exec_inference_dataframe(input_data, model_info_dict):
    
    logging.info('[search log] the start line of the function [exec_inference_dataframe]')
    
    ## 학습 모델 준비(데이터프레임)
    model = model_info_dict['model'] 
    # 데이터 프레임 변환 
    tfidf_df = model
    logging.info(f'[search log] tfidf_df:{tfidf_df}')
    
    tfidf_matrix = np.array(list(tfidf_df['TFIDF_unigram_english']), dtype = object)
    
    doc = tfidf_df['content_processing']
    tfidf_unigram = TfidfVectorizer(max_features=3000).fit(doc)
  
    ################################
    # input data 
    ###############################
    logging.info(f'[search log] input_data :{input_data}')
    logging.info(f'[search log] input_data :{input_data.__class__}')

    line = input_data[0]
    
    logging.info(f'[search log] len(line) :{len(line)}')
    logging.info(f'[search log] line :{line}')
    
    # 한글과 띄어쓰기, 영어를 제외한 모든 글자
    compile = re.compile("[^ \+|a-z|A-Z|ㄱ-ㅣ가-힣]+")
    
    for i in range(len(line)):
        a = compile.sub("",line[i])
        line[i] = a
        
    logging.info(f'[search log] line :{line}')
    logging.info(f'[search log] line.__class__ :{line.__class__}')

    # 불용어 제거
    in_person_stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
    stop_words = in_person_stopwords
        
    def remove_stopwords_from_list(input_data, stopwords):
        """Remove stopwords"""
        filtered_list = []

        for sentence in input_data:
            filtered_sentence = " ".join([word for word in sentence.split() if word not in stopwords])
            filtered_list.append(filtered_sentence)
            
        return filtered_list

    input_data = remove_stopwords_from_list(line, stop_words)
    
    input_data = tfidf_unigram.transform(input_data)
    
    # Convert the sparse matrix to a dense NumPy array
    dense_array = input_data.toarray()
    
    # Convert the NumPy array to a regular Python list
    dense_list = dense_array.tolist()
 
    # result = {'TF-IDF Vec':input_result, 'TF-IDF Values' : input_result2 }
    
    result = {'TF-IDF Vec': dense_list}
    
    return result
 
def exec_inference_file(files, model_info_dict):
    
    """파일기반 추론함수는 files와 로드한 model을 전달받습니다."""
    logging.info('[search log] the start line of the function [exec_inference_file]')
    model = model_info_dict['model']
 
    inference_result = []

    result = {'inference' : inference_result}
    return result