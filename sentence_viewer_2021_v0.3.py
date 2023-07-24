from ntpath import join
import json
import sys
import os

from PyQt5 import QtCore, QtGui, QtWidgets


class SentenceMainWindow(QtWidgets.QMainWindow):
    
    list_json_data = []
    total_count = 0
    list_color = ["#0000ff", "#dc143c", "#8a2be2", "#5f9ea0", "#d2691e",
                  "#6495ed", "#a52a2a", "#00008b", "#b8860b", "#008b8b",
                  "#8b008b", "#006400", "#ff8c00", "#556b2f", "#9932cc",
                  "#8b0000", "#e9967a", "#483d8b", "#2f4f4f", "#00ced1", 
                  "#9400d3", "#ff1493", "#00bfff", "#696969", "#1e90ff",
                  "#b22222", "#228b22", "#daa520", "#808080", "#008000",
                  "#ff69b4", "#4b0082", "#cd5c5c", "#20b2aa", "#f08080",
                  "#87cefa", "#778899", "#ff00ff", "#800000", "#66cdaa",
                  "#0000cd", "#ba55d3", "#3cb371", "#9370d8", "#7b68ee",
                  "#800080", "#663399", "#ff0000", "#bc8f8f", "#4169e1",
                  "#008080", "#8b4513", "#fa8072", "#2e8b57", "#a0522d",
                  "#6a5acd", "#708090", "#4682b4"]

    selection_text_start = 0
    selection_text_end = 0

    origin_context = ''

    check_select = False
    check_cursor = False

    def __init__(self, *args, **kwargs):

        super(SentenceMainWindow, self).__init__(*args, **kwargs)
        # main
        self.main_groupbox = QtWidgets.QGroupBox()
        self.setCentralWidget(self.main_groupbox)
        self.setWindowIcon(QtGui.QIcon('./icon/favicon-32x32'))

        # layout
        self.main_layout = QtWidgets.QGridLayout(self.main_groupbox)
        
        self.btn_prev = QtWidgets.QPushButton()
        self.btn_prev.setFixedHeight(50)
        self.btn_next = QtWidgets.QPushButton()
        self.btn_next.setFixedHeight(50)

        self.txt_edit = QtWidgets.QTextEdit("")
        self.txt_edit.setObjectName("MainText")
        self.txt_edit.setEnabled(True)
        self.txt_edit.setReadOnly(True)
        self.txt_edit.setAcceptRichText(True)
        self.txt_edit.selectionChanged.connect(self.text_selection_changed)
        self.txt_edit.cursorPositionChanged.connect(self.text_cursor_changed)
        
        self.qa_groupbox = QtWidgets.QGroupBox()
        self.qa_groupbox.setStyleSheet("QGroupBox{border:1px solid black;}")
        self.qa_groupbox.setFixedWidth(600)

        self.label_title = QtWidgets.QLabel("T3Q 인턴 전유태, 조영래 >> ★★★요약텍스트([sentence])★★★")
        self.sentence_cnt = QtWidgets.QLabel("글자수 : 0") 
        self.sentence_cnt.setAlignment(QtCore.Qt.AlignCenter)
        self.sentence_cnt.setFixedWidth(150)
                
        self.btn_fileopen = QtWidgets.QPushButton("파일 열기")
        self.btn_fileopen.setFixedSize(120, 30)

        self.btn_submit = QtWidgets.QPushButton("저장")
        self.btn_submit.setFixedHeight(50)
        self.btn_submit.clicked.connect(self.save_file)
        
        self.main_layout.addWidget(self.btn_fileopen, 0, 1, 1, 1)
        self.main_layout.addWidget(self.label_title, 1, 1, 1, 1)
        self.main_layout.addWidget(self.sentence_cnt, 2, 1, 1, 1)
        self.main_layout.addWidget(self.btn_prev, 3, 0, 1, 1)
        self.main_layout.addWidget(self.txt_edit, 3, 1, 2, 1)
        self.main_layout.addWidget(self.btn_next, 3, 2, 1, 1)
        self.main_layout.addWidget(self.qa_groupbox, 3, 3, 1, 1)
        self.main_layout.addWidget(self.btn_submit, 4, 3, 1, 1)

        self.qa_layout = QtWidgets.QVBoxLayout(self.qa_groupbox)
        self.qa_layout.setAlignment(QtCore.Qt.AlignTop)

        self.label_title.setStyleSheet("QLabel{color:black; font-size:21pt; font-weight:bold; margin: 15px 0px 15px 0px;}")
        self.sentence_cnt.setStyleSheet(("QLabel{background-color:#006633; color:white; font-size:10pt; font-weight:bold; padding: 5px 0px 5px 0px;}"))
        self.btn_fileopen.setStyleSheet("QPushButton{border:1px solid white; background-color:#3333FF; color:white; font-size:10pt;}")
        self.btn_submit.setStyleSheet("QPushButton{border:1px solid white; background-color:#3333FF; color:white; font-size:13pt; font-weight:bold}")

        # 요약텍스트
        self.make_basic_sentence()
               
        # etc widget    
        self.menubar = QtWidgets.QMenuBar(self)                     # 상단 메뉴바
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1900, 20))
        self.menubar.setObjectName("menubar")

        self.menuFiles = QtWidgets.QMenu(self.menubar)
        
        self.actionOpen = QtWidgets.QAction(self)
        self.actionOpen.setShortcut('Ctrl+O')
        self.actionExit = QtWidgets.QAction(self)
        self.actionExit.setShortcut('Ctrl+X')

        self.menuFiles.addAction(self.actionOpen)
        self.menuFiles.addAction(self.actionExit)

        self.menubar.addAction(self.menuFiles.menuAction())

        self.statusbar = QtWidgets.QStatusBar(self)                 # 하단 상태바

        self.lbl_json_label = QtWidgets.QLabel('파일명 : ')
        self.lbl_opened_json_value = QtWidgets.QLabel('')
        self.lbl_json_title = QtWidgets.QLabel('제목 : ')
        self.lbl_json_title.hide()
        self.lbl_json_title_value = QtWidgets.QLabel('')
        self.lbl_json_title_value.hide()
        self.lbl_json_index = QtWidgets.QLabel('인덱스 : ')
        self.lbl_json_index.hide()
        self.lbl_json_index_value = QtWidgets.QLabel()
        self.lbl_json_index_value.hide()

        self.statusbar.addWidget(self.lbl_json_label)
        self.statusbar.addWidget(self.lbl_opened_json_value)
        self.statusbar.addWidget(self.lbl_json_title)
        self.statusbar.addWidget(self.lbl_json_title_value)
        self.statusbar.addWidget(self.lbl_json_index)
        self.statusbar.addWidget(self.lbl_json_index_value)

        self.retranslateUi(self)

        self.menubar.setNativeMenuBar(False)
        self.setMenuBar(self.menubar)
        self.setStatusBar(self.statusbar)

        # 버튼 기능 연결
        self.btn_next.clicked.connect(self.next_open)
        self.btn_prev.clicked.connect(self.prev_open)
        
        self.actionOpen.triggered.connect(self.file_open)
        self.btn_fileopen.clicked.connect(self.file_open)

        self.actionExit.triggered.connect(QtCore.QCoreApplication.instance().quit)

        self.resize(1600, 920)


    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Labelon Visual QA Viewer"))

        self.btn_prev.setIcon(QtGui.QIcon('./icon/Angle-double-left.png'))
        self.btn_next.setIcon(QtGui.QIcon('./icon/Angle-double-right.png'))

        self.menuFiles.setTitle(_translate("MainWindow", "Files"))
        self.actionOpen.setText(_translate("MainWindow", "Open"))
        self.actionExit.setText(_translate("MainWindow", "Exit"))


    # 파일 열기
    def file_open(self):
        
        self.list_json_data = []
        self.open_folder = str(QtWidgets.QFileDialog.getExistingDirectory(self, "Select Directory", "./tool_sentence/json"))

        # 폴더 한 개만 선택 (파일 개수가 많으면 로드 할 때 시간이 오래걸림)
        file_list = os.listdir(self.open_folder)

        for file in file_list:
            file_full = os.path.join(self.open_folder, file)
            file_name, file_ext = os.path.splitext(file_full)

            if '.json' == file_ext:

                with open(file_full, 'r', encoding='utf8') as fp:
                    load_data = json.load(fp)

                    one_load_json = {'json_file': '', 'data': ''}

                    one_load_json['json_file'] = file_full
                    one_load_json['data'] = load_data
                    
                    self.list_json_data.append(one_load_json)

        if len(self.list_json_data) == 0:
            # json 데이터 없을 경우
            QtWidgets.QMessageBox.warning(self, "알림", "현재 선택한 폴더에는 json 파일이 없습니다.")
            return

        sentence_type = self.open_folder.split('/')[-1]
        label_title_text = '요약텍스트([sentence] ' + sentence_type + ')'
        self.label_title.setText(label_title_text)

        self.lbl_json_title.show()
        self.lbl_json_title_value.show()
        self.lbl_json_index.show()
        self.lbl_json_index_value.show()
        
        self.total_count = len(self.list_json_data)
        self.set_sentence(0)

    
    def next_open(self):
        # 파일이 선택 안되었을 경우
        if self.lbl_opened_json_value.text() == "":
            self.file_open()
            return

        # 데이터 저장 체크
        if not self.check_save():
            return

        next_index = self.current_index + 1

        if next_index >= self.total_count:
            next_index = 0

        self.set_sentence(next_index)

    
    def prev_open(self):
        # 파일이 선택 안되었을 경우
        if self.lbl_opened_json_value.text() == "":
            self.file_open()
            return

        # 데이터 저장 체크
        if not self.check_save():
            return

        prev_index = self.current_index - 1

        if prev_index < 0:
            prev_index = self.total_count - 1

        self.set_sentence(prev_index)


    def check_save(self):
        # 저장 필수 데이터 체크
        now_summary_one = self.qa_layout.itemAt(0).widget().layout().itemAt(1).widget().toPlainText()
        now_summary_two = self.qa_layout.itemAt(0).widget().layout().itemAt(3).widget().toPlainText()

        now_type = self.qa_layout.itemAt(0).widget().layout().itemAt(2).widget().layout().itemAt(0).widget().text()

        # 한문장 요약이 안 되었을 경우
        if now_summary_one == '':
            QtWidgets.QMessageBox.warning(self, '필수입력사항 확인', '"한 문장 요약"을 입력하세요.', QtWidgets.QMessageBox.Ok)
            return False

        # 3문장 요약 또는 원문대비 20% 요약이 안 되었을 경우
        if now_summary_two == '':
            message_type = now_type
            QtWidgets.QMessageBox.warning(self, '필수입력사항 확인', '"' + now_type + '"을 입력하세요.', QtWidgets.QMessageBox.Ok)
            return False

        # 수정 데이터 > 데이터 형태에 저장
        self.list_json_data[self.current_index]['data']['Annotation']['summary1'] = now_summary_one

        if now_type == '3문장 요약':
            self.list_json_data[self.current_index]['data']['Annotation']['summary2'] = now_summary_two
        else:
            self.list_json_data[self.current_index]['data']['Annotation']['summary3'] = now_summary_two
            self.list_json_data[self.current_index]['data']['Annotation']['summary_3_cnt'] = len(now_summary_two)

        return True
    

    def save_file(self):
        
        # 데이터 저장 체크
        if not self.check_save():
            return

        # 데이터 JSON 파일에 저장
        reply = QtWidgets.QMessageBox.question(self, 'Message', '저장하시겠습니까?', QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)

        if reply == QtWidgets.QMessageBox.Yes:

            """
            # 현재 index 만 저장!!            
            with open(self.list_json_data[self.current_index]['json_file'], 'w', encoding='UTF-8') as outfile:
                outfile.write(json.dumps(self.list_json_data[self.current_index]['data'], ensure_ascii=False, default=str, indent='\t'))
            """

            for one_json in self.list_json_data:
                with open(one_json['json_file'], 'w', encoding='UTF-8') as outfile:
                    outfile.write(json.dumps(one_json['data'], ensure_ascii=False, default=str, indent='\t'))
            
            QtWidgets.QMessageBox.warning(self, "저장 완료 확인", "저장이 완료 되었습니다.", QtWidgets.QMessageBox.Ok)

    
    def make_basic_sentence(self):
        # 요약텍스트 관련 함수
        sentence_groupbox = QtWidgets.QGroupBox()
        sentence_groupbox.setStyleSheet("QGroupBox{border:1px solid white;}")

        sentece_layout = QtWidgets.QGridLayout(sentence_groupbox)
        sentece_layout.setAlignment(QtCore.Qt.AlignTop)
        
        os_groupbox = QtWidgets.QGroupBox()
        os_groupbox.setAlignment(QtCore.Qt.AlignVCenter)
        os_groupbox.setStyleSheet("QGroupBox{background-color:#03C75A;}")
        os_groupbox.setFixedHeight(45)

        os_layout = QtWidgets.QGridLayout(os_groupbox)
        os_layout.setAlignment(QtCore.Qt.AlignVCenter)
        
        os_label_answer = QtWidgets.QLabel("한문장 요약")
        os_label_answer.setAlignment(QtCore.Qt.AlignLeft)
        os_label_answer.setStyleSheet(("QLabel{color:white; font-size:10pt; font-weight:bold;}"))
        
        os_label_count = QtWidgets.QLabel("글자수 : 0")
        os_label_count.setAlignment(QtCore.Qt.AlignRight)
        os_label_count.setStyleSheet(("QLabel{color:white; font-size:10pt; font-weight:bold;}"))

        os_layout.addWidget(os_label_answer, 0, 0, 1, 1)
        os_layout.addWidget(os_label_count, 0, 1, 1, 1)
        
        os_detail_answer = QtWidgets.QTextBrowser()
        os_detail_answer.setFixedHeight(200)
        os_detail_answer.setEnabled(True)
        os_detail_answer.setReadOnly(True)
        os_detail_answer.setStyleSheet("QTextBrowser{border:0px;}")
        os_detail_answer.textChanged.connect(self.changed_summary_one)

        ds_groupbox = QtWidgets.QGroupBox()
        ds_groupbox.setAlignment(QtCore.Qt.AlignVCenter)
        ds_groupbox.setStyleSheet("QGroupBox{background-color:#006633;}")
        ds_groupbox.setFixedHeight(45)

        ds_layout = QtWidgets.QGridLayout(ds_groupbox)
        ds_layout.setAlignment(QtCore.Qt.AlignVCenter)
        
        ds_label_answer = QtWidgets.QLabel("3문장 요약")
        ds_label_answer.setAlignment(QtCore.Qt.AlignLeft)
        ds_label_answer.setStyleSheet(("QLabel{color:white; font-size:10pt; font-weight:bold;}"))

        ds_label_count = QtWidgets.QLabel("글자수 : 0")
        ds_label_count.setAlignment(QtCore.Qt.AlignRight)
        ds_label_count.setStyleSheet(("QLabel{color:white; font-size:10pt; font-weight:bold;}"))

        ds_layout.addWidget(ds_label_answer, 0, 0, 1, 1)
        ds_layout.addWidget(ds_label_count, 0, 1, 1, 1)

        ds_detail_answer = QtWidgets.QTextBrowser()
        ds_detail_answer.setFixedHeight(420)
        ds_detail_answer.setEnabled(True)
        ds_detail_answer.setReadOnly(True)
        ds_detail_answer.setStyleSheet("QTextBrowser{border:0px;}")
        ds_detail_answer.textChanged.connect(self.changed_summary_two)

        sentece_layout.addWidget(os_groupbox, 0, 0, 1, 1)
        sentece_layout.addWidget(os_detail_answer, 1, 0, 1, 1)
        sentece_layout.addWidget(ds_groupbox, 2, 0, 1, 1)
        sentece_layout.addWidget(ds_detail_answer, 3, 0, 1, 1)

        self.qa_layout.addWidget(sentence_groupbox)
    

    def set_sentence(self, index):
        # 현재 index 기준으로 데이터 셋팅 함수
        self.txt_edit.setText("")
    
        self.current_index = index
        
        self.lbl_opened_json_value.setText(os.path.split(self.list_json_data[index]['json_file'])[1])

        self.lbl_json_title_value.setText(self.list_json_data[index]['data'].get("Meta(Acqusition)").get("doc_name"))        
        self.lbl_json_index_value.setText(str(index + 1) + " / " + str(self.total_count))

        temp_context = self.list_json_data[index]['data'].get("Meta(Refine)").get("passage")
        temp_context = temp_context.strip()
        temp_list = temp_context.split("\n")
        temp_list = [s.strip() for s in temp_list]

        self.origin_context = '\n'.join(temp_list).strip()

        json_context = '\n'.join(temp_list).strip()
        
        json_context_cnt = self.list_json_data[index]['data'].get("Meta(Refine)").get("passage_Cnt")
        
        self.sentence_cnt.setText("글자수 : " + str(json_context_cnt))

        summary_1 = self.list_json_data[index]['data'].get("Annotation").get("summary1")
        summary_1_len = len(summary_1)
        self.qa_layout.itemAt(0).widget().layout().itemAt(0).widget().layout().itemAt(1).widget().setText("글자수 : " + str(summary_1_len))
        self.qa_layout.itemAt(0).widget().layout().itemAt(1).widget().setText(summary_1)
        self.qa_layout.itemAt(0).widget().layout().itemAt(1).widget().setReadOnly(False)

        summary_3_cnt = self.list_json_data[index]['data'].get("Annotation").get("summary_3_cnt")

        list_sentence = []

        if summary_3_cnt is not None:
            summary_3 = self.list_json_data[index]['data'].get("Annotation").get("summary3")
            summary_rate = int(summary_3_cnt / json_context_cnt * 100)

            self.qa_layout.itemAt(0).widget().layout().itemAt(2).widget().layout().itemAt(0).widget().setText("원문대비 20% 요약")
            self.qa_layout.itemAt(0).widget().layout().itemAt(2).widget().layout().itemAt(1).widget().setText("원문대비 : " + str(summary_rate) + "%")
            self.qa_layout.itemAt(0).widget().layout().itemAt(3).widget().setText(summary_3)
            list_sentence = summary_3.split("  ")
        else:
            summary_2 = self.list_json_data[index]['data'].get("Annotation").get("summary2")
            summary_2_len = len(summary_2)

            self.qa_layout.itemAt(0).widget().layout().itemAt(2).widget().layout().itemAt(0).widget().setText("3문장 요약")
            self.qa_layout.itemAt(0).widget().layout().itemAt(2).widget().layout().itemAt(1).widget().setText("글자수 : " + str(summary_2_len))
            self.qa_layout.itemAt(0).widget().layout().itemAt(3).widget().setText(summary_2)
            list_sentence = summary_2.split("  ")

        self.list_index = []
        color_i = 0
        
        for idx, sentence in enumerate(list_sentence):
            set_index = []
            start_index = json_context.find(sentence)

            if start_index >= 0:
                end_index = start_index + len(sentence)

                set_index.append(start_index)
                set_index.append(end_index)
                set_index.append(sentence)
                set_index.append(idx)

                if color_i >= len(self.list_color):
                    color_i = 0
                
                set_index.append(self.list_color[color_i])
                color_i += 1
                self.list_index.append(set_index)
        
        self.make_change_text()


    def eventFilter(self, object, event):
        # 이벤트 필터        
        # 답변 생성 이벤트
        if event.type() == QtCore.QEvent.MouseButtonRelease and event.button() == QtCore.Qt.LeftButton and self.check_select == True:
            selection_text = self.txt_edit.toPlainText()[self.selection_text_start:self.selection_text_end]            
            self.list_index.sort(key=lambda x:x[3])

            check_drag = True

            # 드레그 부분 체크 (기존에 선택된 문장은 선택 안됨)
            for origin_index in self.list_index:
                if origin_index[0] <= self.selection_text_start and self.selection_text_start <= origin_index[1]:
                    check_drag = False
                    
                if origin_index[0] <= self.selection_text_end and self.selection_text_end <= origin_index[1]:
                    check_drag = False
                    
                if self.selection_text_start <= origin_index[0] and origin_index[1] <= self.selection_text_end:
                    check_drag = False

            # 이미 선택된 문장이 포함되 있는 경우는 변경 불가                   
            if check_drag == False:
                self.check_select = False                
                self.make_change_text()

                QtWidgets.QMessageBox.warning(self, "문장 요약 체크", "이미 선택된 문장입니다.", QtWidgets.QMessageBox.Ok)
                return QtWidgets.QMainWindow.eventFilter(self, object, event)

            set_new_index = []
            set_new_index.append(self.selection_text_start)
            set_new_index.append(self.selection_text_end)
            set_new_index.append(selection_text)

            new_idx = 0
            if len(self.list_index) > 0:
                new_idx = self.list_index[-1][3] + 1

            set_new_index.append(new_idx)
            
            color_i = new_idx
            if color_i >= len(self.list_color):
                color_i = 0                
            set_new_index.append(self.list_color[color_i])

            self.list_index.append(set_new_index)

            self.make_change_summary2()            
            self.make_change_text()

            self.check_select = False

        # 답변 제거 이벤트
        if event.type() == QtCore.QEvent.MouseButtonRelease and event.button() == QtCore.Qt.LeftButton and self.check_cursor == True:
            
            cursor = self.txt_edit.textCursor()
            start = cursor.selectionStart()

            end_position = len(self.origin_context)
            
            self.list_index.sort(key=lambda x:x[0], reverse=True)

            if start < end_position:
                delete_index = -1
                
                for i in range(len(self.list_index)):
                    if start >= self.list_index[i][0] and start <= self.list_index[i][1]:                        
                        # 현재 답변과 일치하면 문장 삭제
                        delete_index = i
                
                if delete_index >= 0:
                    self.list_index.remove(self.list_index[delete_index])
                
                self.make_change_summary2()
                self.make_change_text()

            self.check_cursor = False

        return QtWidgets.QMainWindow.eventFilter(self, object, event)
        
       
    def text_selection_changed(self):
        # 본문 선택된 글자수가 변경되었을 경우
        cursor = self.txt_edit.textCursor()
        select_start = cursor.selectionStart()
        select_end = cursor.selectionEnd()

        if select_start == select_end:
            return

        selection_text = self.txt_edit.toPlainText()[select_start:select_end]
        self.selection_text_start = select_start
        self.selection_text_end = select_end

        self.check_select = True


    def text_cursor_changed(self):
        # 본문 커서 위치가 변경 되었을 경우
        self.check_cursor = True


    def changed_summary_one(self):
        # 한문장 요약 내용 변경 되었을 때 글자수 카운트
        summary_one = self.qa_layout.itemAt(0).widget().layout().itemAt(1).widget().toPlainText()
        self.qa_layout.itemAt(0).widget().layout().itemAt(0).widget().layout().itemAt(1).widget().setText("글자수 : " + str(len(summary_one)))


    def changed_summary_two(self):
        # 한문장 요약 내용 변경 되었을 때 글자수 카운트
        now_type = self.qa_layout.itemAt(0).widget().layout().itemAt(2).widget().layout().itemAt(0).widget().text()
        summary_two = self.qa_layout.itemAt(0).widget().layout().itemAt(3).widget().toPlainText()
        
        if now_type == '3문장 요약':
            self.qa_layout.itemAt(0).widget().layout().itemAt(2).widget().layout().itemAt(1).widget().setText("글자수 : " + str(len(summary_two)))
        else:
            json_context_cnt = self.list_json_data[self.current_index]['data'].get("Meta(Refine)").get("passage_Cnt")

            summary_rate = int(len(summary_two) / json_context_cnt * 100)
            self.qa_layout.itemAt(0).widget().layout().itemAt(2).widget().layout().itemAt(1).widget().setText("원문대비 : " + str(summary_rate) + "%")

    
    def make_change_text(self):
        # 선택된 내용이 변경 되었을 때 메인 내용 변경 적용 함수
        text = list(self.origin_context)
        self.list_index.sort(key=lambda x:x[0], reverse=True)
        
        for one in self.list_index:
            text.insert(one[1], '</span>')
            text.insert(one[0], '<span style=background-color:' + one[4] + '; style=color:white;>')
        
        self.txt_edit.setText("")

        new_context = ''.join(text)
        list_split = new_context.split('\n')

        line_cnt = 1
        self.txt_edit.setText("")

        for one_line in list_split:
            self.txt_edit.insertHtml(one_line)
            
            if line_cnt < len(list_split):
                self.txt_edit.append("")
                line_cnt += 1

    
    def make_change_summary2(self):
        # 선택된 내용이 변경 되었을 때 2번째 요약 내용 변경 적용 함수
        self.list_index.sort(key=lambda x:x[3])
        
        new_summary_2 = ''
        
        for one in self.list_index:
            new_summary_2 = new_summary_2 + "  " + one[2]
        new_summary_2 = new_summary_2.strip()

        self.qa_layout.itemAt(0).widget().layout().itemAt(3).widget().setText(new_summary_2)
        

if __name__ == "__main__":

    print("--------------------------------------------------------------------------------------------------------------")
    print(__name__)
    print(sys.argv)
    print("--------------------------------------------------------------------------------------------------------------")
    
    app = QtWidgets.QApplication(sys.argv)

    ui = SentenceMainWindow()
    ui.show()

    app.installEventFilter(ui)
    sys.exit(app.exec_())