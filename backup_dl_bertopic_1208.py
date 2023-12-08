# train_sub.py

import t3qai_client as tc
from t3qai_client import T3QAI_TRAIN_OUTPUT_PATH, T3QAI_TRAIN_MODEL_PATH, T3QAI_TRAIN_DATA_PATH, T3QAI_TEST_DATA_PATH, T3QAI_MODULE_PATH, T3QAI_INIT_MODEL_PATH

# Imports
import os
import numpy as np
import logging
import matplotlib.pyplot as plt
import tensorflow as tf
from kiwipiepy import Kiwi


# 저장 파일 확인
def list_files_directories(path):
    # Get the list of all files and directories in current working directory
    dir_list = os.listdir(path)
    logging.info('[hunmin log] Files and directories in {} :'.format(path))
    logging.info('[hunmin log] dir_list : {}'.format(dir_list))


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

## user algorithm 
# T3Q.ai 공통, 알고리즘 파라미터 불러오기(dictionary 형태)
params = tc.train_load_param()
logging.info('params : {}'.format(params))

# logging.info(f'[hunmin log] tensorflow ver : {tf.__version__}')

# 사용할 gpu 번호를 적는다.
# os.environ["CUDA_VISIBLE_DEVICES"]='0'

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus, 'GPU')
        logging.info('[hunmin log] gpu set complete')
        logging.info('[hunmin log] num of gpu: {}'.format(len(gpus)))
    
    except RuntimeError as e:
        logging.info('[hunmin log] gpu set failed')
        logging.info(e)

# 저장 파일 확인
def list_files_directories(path):
    # Get the list of all files and directories in current working directory
    dir_list = os.listdir(path)
    logging.info('[hunmin log] Files and directories in {} :'.format(path))
    logging.info('[hunmin log] dir_list : {}'.format(dir_list))



#불용처리 함수
in_person_stopwords= str(params.get('stop_words', '재배법 수확량 수확 분야 조사 지역 특성 분석 개발 방법 으로 유형 위한 규모 연구 항목 에서 구축 자료 평가 처리 이용 재료 활용 작업 구조 지표 기준 설계 조건 유용 성분 성능 개소')) # params['n_components'])
# in_person_stopwords = '재배법 수확량 수확 분야 조사 지역 특성 분석 개발 방법 으로 유형 위한 규모 연구 항목 에서 구축 자료 평가 처리 이용 재료 활용 작업 구조 지표 기준 설계 조건 유용 성분 성능 개소'
logging.info('[stop words : default] : {}'.format(in_person_stopwords)) 

stop_words = in_person_stopwords
stop_words = stop_words.split(' ')

# mecab, kiwi과 SBERT를 이용한  BERTopic
class CustomTokenizer:
    def __init__(self, tagger):
        self.tagger = tagger
    def __call__(self, sent):
        sent = sent[:1000000]
        word_tokens = self.tagger.analyze(sent) # 형태소 출력
        forms_list = [token.form for token in word_tokens[0][0]] # '(토큰화결과,형태, , ) > [0][0] 뽑기
        #word_tokens = self.tagger.morphs(sent)
        #word_tokens = self.tagger.nouns(sent)
        #result = [word for word in word_tokens if len(word) > 1]
        result = [word for word in forms_list if len(word) > 1]
        final_result = remove_stopwords_from_list(result, stop_words)
        last_result = [re_word for re_word in final_result if len(re_word) > 1]
        return last_result
    
def remove_stopwords_from_list(content_list, stopwords):
    """Remove stopwords"""
    filtered_list = []
    for sentence in content_list:
        filtered_sentence = " ".join([word for word in sentence.split() if word not in stopwords])
        filtered_list.append(filtered_sentence)
    return filtered_list

def exec_train():
    logging.info('[hunmin log] the start line of the function [exec_train]')
    logging.info('[hunmin log] T3QAI_TRAIN_DATA_PATH : {}'.format(T3QAI_TRAIN_DATA_PATH))
    
    # 저장 파일 확인
    list_files_directories(T3QAI_TRAIN_DATA_PATH)
    # my_path = os.path.join(T3QAI_TRAIN_DATA_PATH, 'dataset') + '/'       #  ./meta_data\dataset/dataset.txt
    # 데이터 zip파일의 압축 해제가 안되기 때문에 데이터셋 txt파일 업로드 후 바로 불러온다.
    my_path = os.path.join(T3QAI_TRAIN_DATA_PATH, '') + '/'  # ./meta_data/dataset.txt
    
    # 카테고리
    dataset=['dataset'] ##335개의 문서가 한 줄로 들어간 txt 파일
    dataset_num = len(dataset) #10

    # 경로에 있는 문서데이터(txt에 한 줄 씩 335개 있는 1개 문서)를 load하고 dataset_numpy list에 추가한다.
    dataset_txt = []
    for i in range (dataset_num):
        ad = my_path + str(dataset[i]) + '.txt'
        dataset_txt.append(ad)
  
    for i in range (dataset_num):
        logging.info('[hunmin log] : {}'.format(dataset_txt[i]))
        
    # text_file = "./meta_data\dataset/dataset.txt"
    text_file = my_path + str(dataset[i]) + '.txt'   

    documents = [line.strip() for line in open(text_file, encoding="utf-8").readlines()]
    logging.info('[hunmin log] Input 산림청 첫번째 문서 확인 예시: {}'.format(documents[0]))

    preprocessed_documents = []
    logging.info('[hunmin log] 총 문서 개수: {}'.format(len(documents)))

    for line in tqdm(documents):
    # 빈 문자열이거나 숫자로만 이루어진 줄은 제외
        if line and not line.replace(' ', '').isdecimal():
            preprocessed_documents.append(line)

    ###########################################################################
    ## 불용어 제거 단어
    ## in_person_stopwords = '재배법 수확량 수확 분야 조사 지역 특성 분석 개발 방법 으로 유형 위한 규모 연구 항목 에서 구축 자료 평가 처리 이용 재료 활용 작업 구조 지표 기준 설계 조건 유용 성분 성능 개소'
    ## stop_words = in_person_stopwords
    ## stop_words = stop_words.split(' ')
    ## logging.info('[hunmin log]  불용어 제거 단어: {}'.format(stop_words))
    logging.info('[hunmin log]  불용어 제거 단어: {}'.format(stop_words))
    ###########################################################################

    # 모델 구축
    # 단일 gpu 혹은 cpu학습
    num_classes = 1

    if len(gpus) < 2:
        model = model_build_and_compile(num_classes)
    # multi-gpu
    else:
        # strategy = tf.distribute.MirroredStrategy()
        logging.info('[humin log] gpu devices num {}'.format(strategy.num_replicas_in_sync))
        with strategy.scope():
            model = model_build_and_compile(num_classes)


    # Create embeddings
    # embeddings = embedding_model.encode(preprocessed_documents, show_progress_bar=True)

    # Create topic model
    topics, probs = model.fit_transform(preprocessed_documents)

    ###########################################################################
    ## 플랫폼 시각화
    ###########################################################################
    """
    plot_metrics(tc, history, model, X_test_cnn, Y_test_cnn)
    """

    # 모델 저장하기
    logging.info('[hunmin log] T3QAI_BERTopic_MODEL_PATH : {}'.format(T3QAI_TRAIN_MODEL_PATH))
    custom_model_name= str(params.get('model_name', 'bertopic_model')) # params['model_name'])
    logging.info('[custaom_model_name log] : {}'.format(custom_model_name))
    model.save(os.path.join(T3QAI_TRAIN_MODEL_PATH, custom_model_name),serialization="safetensors", save_embedding_model=embedding_model)
  
    # 저장 파일 확인
    logging.info('[hunmin log] ★모델저장 완료★')
    list_files_directories(T3QAI_TRAIN_DATA_PATH)

    logging.info('[hunmin log] the finish line of the function [exec_train]')

    # 모델 토픽 결과 출력
    topics: List[int] = None
    top_n_topics: int = None

    # 전체 주제 수 출력하기
    freq_df = model.get_topic_freq()
    freq_df = freq_df.loc[freq_df.Topic != -1, :]

    if topics is not None:
        topics = list(topics)
    elif top_n_topics is not None:
        topics = sorted(freq_df.Topic.to_list()[:top_n_topics])
    else:
        topics = sorted(freq_df.Topic.to_list()[0:])

    logging.info('[★ 주제 수 ] : {} 개로 예상합니다.'.format(len(topics)))
    
    print('\n')
    print('Result topic 결과입니다.')
    model_get_topic_info(model)
    
    #model_visualize_barchart(model)
    #visualize_topics2(model)

    # Show wordcloud
    # create_wordcloud(model, topic=1)
    filepath1 = os.path.join(T3QAI_TRAIN_OUTPUT_PATH, "heatmap.html")
    filepath2 = os.path.join(T3QAI_TRAIN_OUTPUT_PATH, "visualize_barchart.html")
    
    fig = model.visualize_heatmap()
    fig.write_html(filepath1)
    
    fig = model.visualize_barchart()
    fig.write_html(filepath2)
    
    # b = JEONTopic()
    
    # filepath3 = os.path.join(T3QAI_TRAIN_OUTPUT_PATH, "visualize_document.html")
    # fig = b.visualize_documents2(preprocessed_documents, embeddings=embeddings)
    # fig.write_html(filepath3)
    
###########################################################################
## exec_train() 호출 함수 끝
###########################################################################

from konlpy.tag import Mecab
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer

#custom_tokenizer = CustomTokenizer(Mecab(dicpath=r'C:\mecab\mecab-ko-dic'))
custom_tokenizer = CustomTokenizer(Kiwi())

vectorizer = CountVectorizer( tokenizer=custom_tokenizer, max_features=3000,  ngram_range=(1, 1))

from umap import UMAP
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from bertopic.vectorizers import ClassTfidfTransformer

# ## user algorithm 
# # T3Q.ai 공통, 알고리즘 파라미터 불러오기(dictionary 형태)
# params = tc.train_load_param()
# logging.info('params : {}'.format(params))

n_neighbors = int(params.get('n_neighbors', '10')) # params['n_neighbors'])
n_components = int(params.get('n_components', '5')) # params['n_components'])
min_dist = float(params.get('min_dist', '0.2')) # params['min_dist'])
random_seed = int(params["random_seed"])
min_cluster_size = int(params.get('min_cluster_size', '2')) # params['min_cluster_size'])


# Step 1 - Extract embeddings
embedding_model = SentenceTransformer("sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens")

# Step 2 - Reduce dimensionality
umap_model = UMAP(n_neighbors=n_neighbors, n_components=n_components, min_dist=min_dist, metric='cosine',random_state=random_seed)

# Step 3 - Cluster reduced embeddings
hdbscan_model = HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean', cluster_selection_method='eom', prediction_data=True)

# Step 4 - Tokenize topics
vectorizer_model = vectorizer
#CountVectorizer(tokenizer=custom_tokenizer, max_features=3000,  ngram_range=(1, 1))

# Step 5 - Create topic representation
ctfidf_model = ClassTfidfTransformer()

# Step 6 - (Optional) Fine-tune topic representations with 
# a `bertopic.representation` model
representation_model = KeyBERTInspired()

def model_build_and_compile(num_classes):
    #모델 구축
    # All steps together
    
    # 사용자로부터 입력 받기
    user_input = params.get('n_topics', 'auto')
    
    # 입력 값이 숫자인지 문자열인지 확인
    if user_input.isdigit():
        # 입력 값이 숫자일 경우 실행할 코드
        nr_topics_input = int(params.get('n_topics', '10')) # params['n_topics'])
    else:
        # 입력 값이 문자열일 경우 실행할 코드
        nr_topics_input = str(params.get('n_topics', 'auto')) # params['n_topics'])
    
    
    logging.info('[n_topics] : {} '.format(nr_topics_input))
    top_n_words_input = int(params.get('top_n_words', '10')) # params['top_n_words'])
    logging.info('[top_n_words] : {} '.format(top_n_words_input))
    
    model = BERTopic(
    nr_topics=nr_topics_input,
    top_n_words=top_n_words_input,
    embedding_model=embedding_model,          # Step 1 - Extract embeddings
    umap_model=umap_model,                    # Step 2 - Reduce dimensionality
    hdbscan_model=hdbscan_model,              # Step 3 - Cluster reduced embeddings
    vectorizer_model=vectorizer_model,        # Step 4 - Tokenize topics
    ctfidf_model=ctfidf_model,                # Step 5 - Extract topic words
    representation_model=representation_model, # Step 6 - (Optional) Fine-tune topic represenations
    calculate_probabilities=True              # 문서당 할당된 주제의 확률 대신 문서당 모든 주제의 확률을 계산합니다. 문서가 많은 경우(> 100_000) 주제 추출 속도가 느려질 수 있습니다. 
                                                # 참고: false인 경우 해당 시각화 방법을 사용할 수 없습니다 visualize_probabilities. 
                                                # 참고: 이는 HDBSCAN에 사용된 주제 확률의 근사치이며 정확한 표현은 아닙니다.
    )

    return model
    
#######################
## 시각화 구현
#######################
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# FONT_PATH =r"C:\Windows\Fonts\malgun.ttf"

# 토픽 정보 출력 함수
def model_get_topic_info(model):
    logging.info('[★ Result topic 결과 준비가 완료되었습니다.]')
    print(model.get_topic_info())
    print(model.get_topic_info()['Count'])
    print(model.get_topic_info()['Name'])
    print(model.get_topic_info()['Representation'])
    print(model.get_topic_info()['Representative_Docs'])
    

# BERTopic barchart 결과화면
def model_visualize_barchart(model):
    model.visualize_barchart(top_n_topics=50, n_words=5, custom_labels=model.set_topic_labels)

#워드 클라우드 시각화 함수
def create_wordcloud(model, topic):
    text = {word: value for word, value in model.get_topic(topic)}
    logging.info('[★ Word Cloud 시각화 결과 화면 준비가 완료되었습니다.]')
    print(text)
    FONT_PATH =r"malgun.ttf"
    
    wc = WordCloud(font_path=FONT_PATH,background_color="white", max_words=1000)
    wc.generate_from_frequencies(text)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
 
    filepath = os.path.join(T3QAI_TRAIN_OUTPUT_PATH, 'wordcloud.jpg')
    plt.savefig(filepath)

from typing import List
from typing import Union
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


from sklearn.preprocessing import StandardScaler, MinMaxScaler

def _plotly_topic_visualization(df: pd.DataFrame,
                                topic_list: List[str],
                                title: str,
                                width: int,
                                height: int):
    """ Create plotly-based visualization of topics with a slider for topic selection """

    def get_color(topic_selected):
        if topic_selected == -1:
            marker_color = ["#B0BEC5" for _ in topic_list]
        else:
            marker_color = ["red" if topic == topic_selected else "#B0BEC5" for topic in topic_list]
        return [{'marker.color': [marker_color]}]

    # Prepare figure range
    x_range = (df.x.min() - abs((df.x.min()) * .15), df.x.max() + abs((df.x.max()) * .15))
    y_range = (df.y.min() - abs((df.y.min()) * .15), df.y.max() + abs((df.y.max()) * .15))

    # Plot topics
    fig = px.scatter(df, x="x", y="y", size="Size", size_max=40, template="simple_white", labels={"x": "", "y": ""},
                    hover_data={"Topic": True, "Words": True, "Size": True, "x": False, "y": False})
    fig.update_traces(marker=dict(color="#B0BEC5", line=dict(width=2, color='DarkSlateGrey')))

    # Update hover order
    fig.update_traces(hovertemplate="<br>".join(["<b>Topic %{customdata[0]}</b>",
                                                "%{customdata[1]}",
                                                "Size: %{customdata[2]}"]))

    # Create a slider for topic selection
    steps = [dict(label=f"Topic {topic}", method="update", args=get_color(topic)) for topic in topic_list]
    sliders = [dict(active=0, pad={"t": 50}, steps=steps)]

    # Stylize layout
    fig.update_layout(
        title={
            'text': f"{title}",
            'y': .95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                size=22,
                color="Black")
        },
        width=width,
        height=height,
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Rockwell"
        ),
        xaxis={"visible": False},
        yaxis={"visible": False},
        sliders=sliders
    )

    # Update axes ranges
    fig.update_xaxes(range=x_range)
    fig.update_yaxes(range=y_range)

    # Add grid in a 'plus' shape
    fig.add_shape(type="line",
                x0=sum(x_range) / 2, y0=y_range[0], x1=sum(x_range) / 2, y1=y_range[1],
                line=dict(color="#CFD8DC", width=2))
    fig.add_shape(type="line",
                x0=x_range[0], y0=sum(y_range) / 2, x1=x_range[1], y1=sum(y_range) / 2,
                line=dict(color="#9E9E9E", width=2))
    fig.add_annotation(x=x_range[0], y=sum(y_range) / 2, text="D1", showarrow=False, yshift=10)
    fig.add_annotation(y=y_range[1], x=sum(x_range) / 2, text="D2", showarrow=False, xshift=10)
    fig.data = fig.data[::-1]

    return fig    


def visualize_topics2(model,
                    topics: List[int] = None,
                    top_n_topics: int = None,
                    custom_labels: Union[bool, str] = False,
                    title: str = "<b>Intertopic Distance Map</b>",
                    width: int = 650,
                    height: int = 650) -> go.Figure:

    # Select topics based on top_n and topics args
    freq_df = model.get_topic_freq()              #                  Topic Count
    freq_df = freq_df.loc[freq_df.Topic != -1, :] # -1 이상치 주제의 index 1 제거  
    # index, Topic , Count
    
    # print(topics) # None 상태
    if topics is not None:
        topics = list(topics)
    elif top_n_topics is not None:
        topics = sorted(freq_df.Topic.to_list()[:top_n_topics])
    else:
        topics = sorted(freq_df.Topic.to_list())

    # Extract topic words and their frequencies
    topic_list = sorted(topics)
    frequencies = [model.topic_sizes_[topic] for topic in topic_list]
    if isinstance(custom_labels, str):
        words = [[[str(topic), None]] + model.topic_aspects_[custom_labels][topic] for topic in topic_list]
        words = ["_".join([label[0] for label in labels[:4]]) for labels in words]
        words = [label if len(label) < 30 else label[:27] + "..." for label in words]
    elif custom_labels and topic_model.custom_labels_ is not None:
        words = [model.custom_labels_[topic + model._outliers] for topic in topic_list]
    else:
        words = [" | ".join([word[0] for word in model.get_topic(topic)[:5]]) for topic in topic_list]

    # Embed c-TF-IDF into 2D
    all_topics = sorted(list(model.get_topics().keys()))
    indices = np.array([all_topics.index(topic) for topic in topics])

    if model.topic_embeddings_ is not None:
        embeddings = model.topic_embeddings_[indices]
        embeddings = UMAP(n_neighbors=2, n_components=4, metric='cosine', random_state=42).fit_transform(embeddings)

    else:
        embeddings = topic_model.c_tf_idf_.toarray()[indices]
        embeddings = MinMaxScaler().fit_transform(embeddings)
        embeddings = UMAP(n_neighbors=2, n_components=4, metric='hellinger', random_state=42).fit_transform(embeddings)

    # Visualize with plotly
    df = pd.DataFrame({"x": embeddings[:, 0], "y": embeddings[:, 1],
                    "Topic": topic_list, "Words": words, "Size": frequencies})
    
    return _plotly_topic_visualization(df, topic_list, title, width, height)
    
def preprocessed_document():
    logging.info('[hunmin log] the start line of the function [exec_train]')
    logging.info('[hunmin log] T3QAI_TRAIN_DATA_PATH : {}'.format(T3QAI_TRAIN_DATA_PATH))
    
    # 저장 파일 확인
    list_files_directories(T3QAI_TRAIN_DATA_PATH)
    # my_path = os.path.join(T3QAI_TRAIN_DATA_PATH, 'dataset') + '/'       #  ./meta_data\dataset/dataset.txt
    # 데이터 zip파일의 압축 해제가 안되기 때문에 데이터셋 txt파일 업로드 후 바로 불러온다.
    my_path = os.path.join(T3QAI_TRAIN_DATA_PATH, '') + '/'  # ./meta_data/dataset.txt
    
    # 카테고리
    dataset=['dataset'] ##335개의 문서가 한 줄로 들어간 txt 파일
    dataset_num = len(dataset) #10

    # 경로에 있는 문서데이터(txt에 한 줄 씩 335개 있는 1개 문서)를 load하고 dataset_numpy list에 추가한다.
    dataset_txt = []
    for i in range (dataset_num):
        ad = my_path + str(dataset[i]) + '.txt'
        dataset_txt.append(ad)
  
    for i in range (dataset_num):
        logging.info('[hunmin log] : {}'.format(dataset_txt[i]))
        
    # text_file = "./meta_data\dataset/dataset.txt"
    text_file = my_path + str(dataset[i]) + '.txt'   

    documents = [line.strip() for line in open(text_file, encoding="utf-8").readlines()]
    logging.info('[hunmin log] Input 산림청 첫번째 문서 확인 예시: {}'.format(documents[0]))

    preprocessed_documents = []
    logging.info('[hunmin log] 총 문서 개수: {}'.format(len(documents)))

    for line in tqdm(documents):
    # 빈 문자열이거나 숫자로만 이루어진 줄은 제외
        if line and not line.replace(' ', '').isdecimal():
            preprocessed_documents.append(line)
            
    return preprocessed_documents
    
    
    # 시간오래걸리는 것을 방지 하기위해 다음과 같이 코드를 미리 선언해두면 속도가 빠르다.
    from umap import UMAP
    from sentence_transformers import SentenceTransformer
    
    # Prepare embeddings preprocessed_documents(산림청데이터)
    sentence_model = SentenceTransformer("sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens")
    embeddings = sentence_model.encode(preprocessed_documents, show_progress_bar=True)
    
    # Train BERTopic
    topic_model = BERTopic(embedding_model="sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens", \
                     vectorizer_model=vectorizer,
                     nr_topics='auto',
                     top_n_words=10,
                     calculate_probabilities=True).fit(preprocessed_documents, embeddings)
    
    topic_model.get_topic_info()
    
    hierarchical_topics = topic_model.hierarchical_topics(preprocessed_documents)
    
    # Reduce dimensionality of embeddings, this step is optional
    reduced_embeddings = UMAP(n_neighbors=15, n_components=2, min_dist=0.25, metric='cosine').fit_transform(embeddings)
    
    topic_model.visualize_documents(preprocessed_documents, embeddings=embeddings)
    
## jeontopic
from typing import List
from typing import Union

import pandas as pd
import numpy as np
import plotly.graph_objects as go

my_path = os.path.join(T3QAI_TRAIN_DATA_PATH, '') + '/'  # ./meta_data/dataset.txt

# 카테고리
dataset=['dataset'] ##335개의 문서가 한 줄로 들어간 txt 파일
dataset_num = len(dataset) #10

# 경로에 있는 문서데이터(txt에 한 줄 씩 335개 있는 1개 문서)를 load하고 dataset_numpy list에 추가한다.
dataset_txt = []
for i in range (dataset_num):
    ad = my_path + str(dataset[i]) + '.txt'
    dataset_txt.append(ad)

for i in range (dataset_num):
    logging.info('[hunmin log] : {}'.format(dataset_txt[i]))
    
# text_file = "./meta_data\dataset/dataset.txt"
text_file = my_path + str(dataset[i]) + '.txt'   

documents = [line.strip() for line in open(text_file, encoding="utf-8").readlines()]
logging.info('[hunmin log] Input 산림청 첫번째 문서 확인 예시: {}'.format(documents[0]))

preprocessed_documents = []
logging.info('[hunmin log] 총 문서 개수: {}'.format(len(documents)))

for line in tqdm(documents):
# 빈 문자열이거나 숫자로만 이루어진 줄은 제외
    if line and not line.replace(' ', '').isdecimal():
        preprocessed_documents.append(line)

sentence_model = SentenceTransformer("sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens")
embeddings = sentence_model.encode(preprocessed_documents, show_progress_bar=True)

# class JEONTopic(BERTopic):
#     def visualize_documents2(topic_model,
#                         docs: List[str],
#                         topics: List[int] = None,
#                         embeddings: np.ndarray = None,
#                         reduced_embeddings: np.ndarray = None,
#                         sample: float = None,
#                         hide_annotations: bool = False,
#                         hide_document_hover: bool = False,
#                         custom_labels: Union[bool, str] = False,
#                         title: str = "<b>Documents and Topics</b>",
#                         width: int = 1200,
#                         height: int = 750):

    
#         # topic_per_doc = model.topics_

#         doc_info = model.get_document_info(docs)
#         topic_per_doc = doc_info['Topic']

#         # Sample the data to optimize for visualization and dimensionality reduction
#         if sample is None or sample > 1:
#             sample = 1

#         indices = []

#         for topic in set(topic_per_doc):
#             s = np.where(np.array(topic_per_doc) == topic)[0]        # np.where 조건을 만족하는 값 찾기
#             size = len(s) if len(s) < 100 else int(len(s) * sample)
#             np.random.seed(42)
#             indices.extend(np.random.choice(s, size=size, replace=False))    # 랜덤으로 선택, 결과 값 달라지기 때문에 umap에서 42고정
#         indices = np.array(indices)

        
#         print('indices :' , indices)
        
#         #print('len(indices)', len(indices))

#         df = pd.DataFrame({"topic": np.array(topic_per_doc)[indices]})
#         #df["doc_psg"] = [docs[index] for index in indices]
#         df["topic"] = [topic_per_doc[index] for index in indices]
#         df["doc"] = [f"문서번호 : {i}_" + "".join(docs[i][0:20]) for i in indices]


#         # Extract embeddings if not already done
#         if sample is None:
#             if embeddings is None and reduced_embeddings is None:
#                 embeddings_to_reduce = model._extract_embeddings(df.doc.to_list(), method="document")
#             else:
#                 embeddings_to_reduce = embeddings
#         else:
#             if embeddings is not None:
#                 embeddings_to_reduce = embeddings[indices]
#             elif embeddings is None and reduced_embeddings is None:
#                 embeddings_to_reduce = model._extract_embeddings(df.doc.to_list(), method="document")

#         print('차원 감소 임베딩:', reduced_embeddings)

#         # Reduce input embeddings
#         if reduced_embeddings is None:
#             umap_model = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine',random_state=42).fit(embeddings_to_reduce)
#             embeddings_2d = umap_model.embedding_
#         elif sample is not None and reduced_embeddings is not None:
#             embeddings_2d = reduced_embeddings[indices]
#         elif sample is None and reduced_embeddings is not None:
#             embeddings_2d = reduced_embeddings

#         print(embeddings_2d)

#         unique_topics = set(topic_per_doc)
#         if topics is None:
#             topics = unique_topics

#         # Combine data
#         df["x"] = embeddings_2d[:, 0]
#         df["y"] = embeddings_2d[:, 1]

#         # Prepare text and names 주제 이름 넣어주기
#         if isinstance(custom_labels, str):
#             names = [[[str(topic), None]] + model.topic_aspects_[custom_labels][topic] for topic in unique_topics]
#             names = ["_".join([label[0] for label in labels[:4]]) for labels in names]
#             names = [label if len(label) < 30 else label[:27] + "..." for label in names]
#         elif model.custom_labels_ is not None and custom_labels:
#             names = [model.custom_labels_[topic + model._outliers] for topic in unique_topics]
#         else:
#             names = [f"{topic}_" + "_".join([word for word, value in model.get_topic(topic)][:3]) for topic in unique_topics]
         
#         # Visualize
#         fig = go.Figure()

#         # Outliers and non-selected topics
#         non_selected_topics = set(unique_topics).difference(topics)

#         if len(non_selected_topics) == 0:
#             non_selected_topics = [-1]

#         selection = df.loc[df.topic.isin(non_selected_topics), :]
#         selection["text"] = ""
#         selection.loc[len(selection), :] = [None, None, selection.x.mean(), selection.y.mean(), "Other documents"]

#         fig.add_trace(
#             go.Scattergl(
#                 x=selection.x,
#                 y=selection.y,
#                 hovertext=selection.doc if not hide_document_hover else None,
#                 hoverinfo="text",
#                 mode='markers+text',
#                 name="other",
#                 showlegend=False,
#                 marker=dict(color='#CFD8DC', size=5, opacity=0.5)
#             )
#         )

#         # Selected topics
#         for name, topic in zip(names, unique_topics): # 0_산림_목재_참나무, 0 
#             print(topic)
#             if topic in topics and topic != -1:
#                 selection = df.loc[df.topic == topic, :]
#                 print(selection)
#                 selection["text"] = ""

#                 if not hide_annotations:
#                     selection.loc[len(selection), :] = [None, None, selection.x.mean(), selection.y.mean(), name]

#                 fig.add_trace(
#                     go.Scattergl(
#                         x=selection.x,
#                         y=selection.y,
#                         hovertext=selection.doc if not hide_document_hover else None,
#                         hoverinfo="text",
#                         text=selection.text,
#                         mode='markers+text',
#                         name=name,
#                         textfont=dict(
#                             size=12,
#                         ),
#                         marker=dict(size=5, opacity=0.5)
#                     )
#                 )

#         # Add grid in a 'plus' shape
#         x_range = (df.x.min() - abs((df.x.min()) * .15), df.x.max() + abs((df.x.max()) * .15))
#         y_range = (df.y.min() - abs((df.y.min()) * .15), df.y.max() + abs((df.y.max()) * .15))
#         fig.add_shape(type="line",
#                     x0=sum(x_range) / 2, y0=y_range[0], x1=sum(x_range) / 2, y1=y_range[1],
#                     line=dict(color="#CFD8DC", width=2))
#         fig.add_shape(type="line",
#                     x0=x_range[0], y0=sum(y_range) / 2, x1=x_range[1], y1=sum(y_range) / 2,
#                     line=dict(color="#9E9E9E", width=2))
#         fig.add_annotation(x=x_range[0], y=sum(y_range) / 2, text="D1", showarrow=False, yshift=10)
#         fig.add_annotation(y=y_range[1], x=sum(x_range) / 2, text="D2", showarrow=False, xshift=10)

#         # Stylize layout
#         fig.update_layout(
#             template="simple_white",
#             title={
#                 'text': f"{title}",
#                 'x': 0.5,
#                 'xanchor': 'center',
#                 'yanchor': 'top',
#                 'font': dict(
#                     size=22,
#                     color="Black")
#             },
#             width=width,
#             height=height
#         )

#         fig.update_xaxes(visible=False)
#         fig.update_yaxes(visible=False)

#         return fig


    