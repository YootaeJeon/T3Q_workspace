# train_sub.py

import logging, os
import t3qai_client as tc
from t3qai_client import T3QAI_TRAIN_OUTPUT_PATH, T3QAI_TRAIN_MODEL_PATH, T3QAI_INIT_MODEL_PATH, T3QAI_TRAIN_DATA_PATH, T3QAI_TEST_DATA_PATH, T3QAI_MODULE_PATH
        
# 저장 파일 확인
def list_files_directories(path):
    # Get the list of all files and directories in current working directory
    dir_list = os.listdir(path)
    logging.info('[search log] ★ Files and directories in {} :'.format(path))
    logging.info('[search log] ★ dir_list : {}'.format(dir_list))  

def exec_train():
    logging.info('[search log] the start line of the function [exec_train]')
    logging.info('[search log] T3QAI_TRAIN_DATA_PATH : {}'.format(T3QAI_TRAIN_DATA_PATH))

    logging.info('[Search lo] ★ Files and directories in [T3QAI_TRAIN_DATA_PATH]')
    list_files_directories(T3QAI_TRAIN_DATA_PATH)
    
    logging.info('[Search log] ★ Files and directories in [T3QAI_TRAIN_MODEL_PATH]')
    list_files_directories(T3QAI_TRAIN_MODEL_PATH)
    
    logging.info('[Search log] ★ Files and directories in [T3QAI_MODULE_PATH]')
    list_files_directories(T3QAI_MODULE_PATH)
    
    logging.info('[Search log] ★ Files and directories in [T3QAI_INIT_MODEL_PATH]')
    list_files_directories(T3QAI_INIT_MODEL_PATH)
    
    logging.info('[Search log] ★ Files and directories in [T3QAI_TRAIN_OUTPUT_PATH]')
    list_files_directories(T3QAI_TRAIN_OUTPUT_PATH)
    
    logging.info('[Search log] ★ Files and directories in [T3QAI_TEST_DATA_PATH]')
    list_files_directories(T3QAI_TEST_DATA_PATH)
    
    # 저장 파일 확인
    list_files_directories(T3QAI_TRAIN_DATA_PATH)
    # text_file = os.path.join(T3QAI_TRAIN_DATA_PATH, '') + '/'+'./json_to_text(noun).txt'
    # logging.info(f'{doc_table['content_processing']}')
    # text_file= ['a','b']
    #text_file = doc_table['content_processing']
    
    from tqdm import tqdm

    #documents = [line.strip() for line in open(text_file, encoding="utf-8").readlines()]
    documents = doc_table['content_processing']
    logging.info('[search log] 전처리 함수 문서 읽기 TEST:{}'.format(documents[1]))

    preprocessed_documents = []

    for line in tqdm(documents):
        # 빈 문자열이거나 숫자로만 이루어진 줄은 제외
        if line and not line.replace(' ', '').isdecimal():
            preprocessed_documents.append(line)

    from sklearn.feature_extraction.text import TfidfVectorizer
    import pandas as pd
    import numpy as np   

    # TF-IDF 모델 학습
    doc = documents
    tfidf_unigram = TfidfVectorizer(max_features=3000).fit(doc)
    logging.info('[search log] tfidf_unigram TEST :{}'.format(tfidf_unigram.vocabulary_))

    import pickle
    # TF-IDF vocabulary_unigram_english.pkl저장
    with open(T3QAI_TRAIN_DATA_PATH +'/'+ f'vocabulary_unigram_english.pkl', 'wb') as v:
        pickle.dump(tfidf_unigram.vocabulary_, v)  # n

    # TF-IDF vocabulary_unigram_english 불러오기
    with open(T3QAI_TRAIN_DATA_PATH +'/'+f'vocabulary_unigram_english.pkl', 'rb') as v:
        loaded_vocab_unigram = pickle.load(v)

    # 문서를 TF-IDF로 변환
    mat_unigram = tfidf_unigram.transform(doc).toarray()
    logging.info('[search log] TF-IDF Unigram shape :{}'.format(mat_unigram.shape))


    # TFIDF 벡터 데이터프레임 저장
    # doc_df['TFIDF_unigram'] >>Series
    doc_table['TFIDF_unigram_english'] = mat_unigram.tolist()  # array > list  # tolist, np.array
    logging.info('[search log] TFIDF_unigram_english :{}'.format(mat_unigram.shape))
    logging.info('[search log] TFIDF_unigram_english :{}'.format(doc_table['TFIDF_unigram_english']))
    logging.info('[search log] doc_table :{}'.format(doc_table))


    # TF-IDF tfidf_unigram_english저장
    with open(T3QAI_TRAIN_DATA_PATH +'/'+ f'tfidf_unigram_english.pkl', 'wb') as v:
        pickle.dump(doc_table, v)  # n

    # tfidf vocab 불러오기
    # Unigram
    with open(T3QAI_TRAIN_DATA_PATH +'/'+ f'vocabulary_unigram_english.pkl', 'rb') as f:
        loaded_vectorizer_unigram = pickle.load(f)


    import pandas as pd
    df = pd.read_pickle(T3QAI_TRAIN_DATA_PATH +'/'+ f'tfidf_unigram_english.pkl')


    # vocabulary key, value 반전시키기
    reverse_vocabulary_unigram = {v: k for k, v in loaded_vectorizer_unigram.items()}
    logging.info('[search log] TF-IDF Unigram shape :{}'.format(reverse_vocabulary_unigram))

    # 중요 키워드를 가중치 높은 순으로 얻는 함수
    def get_top_keywords(tfidf_values, n=5):
        sorted_indices = sorted(range(len(tfidf_values)), key=lambda i: tfidf_values[i], reverse=True)
        return [reverse_vocabulary_unigram[i] for i in sorted_indices[:]]

    # 중요 키워드 얻기
    df['Top_Keywords_unigram'] = df['TFIDF_unigram_english'].apply(get_top_keywords)

    # tfidf 가중치를 높은 순으로 내림차순
    df['TFIDF_sorted_unigram'] = df['TFIDF_unigram_english'].apply(lambda x: sorted(x, reverse=True))

    # 값들을 리스트로 변환
    # ast 
    import ast
    if type(df.Top_Keywords_unigram[0])==str:
        df['Top_Keywords_unigram'] = df.Top_Keywords_unigram.apply(ast.literal_eval)
    # type(df.Top_Keywords_unigram[0])

    df['Top_Keywords_unigram_str'] = df.Top_Keywords_unigram.apply(' '.join)

    logging.info('[search log] df:{}'.format(df))


    # TF-IDF vocabulary 저장
    with open(T3QAI_TRAIN_DATA_PATH +'/'+ f'topkeyword.pkl', 'wb') as v:
        pickle.dump(df, v)  
    
    # TF-IDF vocabulary 저장
    with open(T3QAI_TRAIN_MODEL_PATH +'/'+ f'topkeyword.pkl', 'wb') as v:
        pickle.dump(df, v)  
    


    # 모델 저장하기
    logging.info('[search log] T3QAI_TRAOM_MODEL_PATH : {}'.format(T3QAI_TRAIN_MODEL_PATH))
    list_files_directories(T3QAI_TRAIN_MODEL_PATH)
    
    # 저장 파일 확인
    # logging.info('[search log] ★모델저장 완료★')


    logging.info('[search log] the finish line of the function [exec_train]')

    ###########################################################################
    ## 시각화 구현
    ###########################################################################
    file_name = df['name']
    tfidf_matrix = np.array(list(df['TFIDF_unigram_english']), dtype = object)

    from sklearn.metrics.pairwise import cosine_similarity
    similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)

    import matplotlib.pyplot as plt
    plt.rc('font', family='Malgun Gothic')

    import seaborn as sns

    sns.heatmap(similarity, xticklabels=file_name, yticklabels=file_name, cmap='viridis')

    # save result IMAGE
    filepath = os.path.join(T3QAI_TRAIN_OUTPUT_PATH, 'tfidf_english_doc_heatmap.jpg')

    plt.savefig(filepath)
    
    logging.info('[search log] the finish line of the function [exec_train]')


    
    # from sklearn.metrics import confusion_matrix
    # import seaborn as sns
    # import matplotlib.pyplot as plt

    # plt.savefig(os.path.join(T3QAI_TRAIN_OUTPUT_PATH,'confusion matrix.png'))

    
###########################################################################
## exec_train() 호출 함수 끝                       
###########################################################################

###########################################################################
# 데이터 전처리하기
###########################################################################
import os
import numpy as np
import json
import pandas as pd
from collections import Counter

import os
import numpy as np
import json
import pandas as pd
from collections import Counter

def listToString(str_list):
    result = ""
    for s in str_list:
        result += s + " "
    return result.strip()

with open(T3QAI_TRAIN_DATA_PATH+"/"+'english_data.txt',encoding='UTF8') as file:
    data = []
    for line in file:
        # ':' 기호를 기준으로 파일 이름과 내용을 분리
        name, content = line.split(':', 1)
        
        if len(content)>100:
            data.append({'name': name.strip(), 'content': content.strip()})

df = pd.DataFrame(data)

# 불용어 제거하기

import re

# 한글과 띄어쓰기, 영어를 제외한 모든 글자
compile = re.compile("[^\+|a-z|A-Z|ㄱ-ㅣ가-힣]+")

line = df['content']

in_person_stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

stop_words = in_person_stopwords

def remove_stopwords_from_list(line, stopwords):
    """Remove stopwords"""
    filtered_list = []

    for sentence in line:
        filtered_sentence = " ".join([word for word in sentence.split() if word not in stopwords])
        filtered_list.append(filtered_sentence)
        
    return filtered_list

processing_line = remove_stopwords_from_list(line, stop_words)

processing_line  = pd.concat([pd.DataFrame(processing_line)], axis = 1)
processing_line.columns = ['content_processing'] #

doc_table = pd.concat([df,pd.DataFrame(processing_line)], axis = 1)

###########################################################################
# 데이터 전처리하기 끝
###########################################################################
