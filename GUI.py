import sys
import os
from glob import glob
import cv2
import numpy as np
from PIL import Image
import onnxruntime as ort
from ultralytics import YOLO

# PyQt5 관련 라이브러리
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QProgressBar, QMessageBox, QFileDialog)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# 상수 설정 (모델 경로)
YOLO_PATH = 'yolo/model.pt'
ARCFACE_PATH = "weights/arcface.onnx"

def preprocess(img):
    img = cv2.resize(img, (112, 112))
    img = (img.astype(np.float32) - 127.5) / 128.0
    return img[np.newaxis, ...]

# --- 작업 쓰레드 (비디오 처리 로직) ---
class VideoProcessThread(QThread):
    progress_signal = pyqtSignal(int)      # 진행률 전송
    finished_signal = pyqtSignal(str)      # 완료 시 메시지 전송
    error_signal = pyqtSignal(str)         # 에러 메시지

    def __init__(self, video_path, face_detector, sess, known_face_embeddings):
        super().__init__()
        self.video_path = video_path
        self.face_detector = face_detector
        self.sess = sess
        self.known_face_embeddings = known_face_embeddings
        self.running = True

    def run(self):
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.error_signal.emit("비디오 파일을 열 수 없습니다.")
                return

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            output_path = 'output_processed.mp4'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            input_name = self.sess.get_inputs()[0].name
            output_name = self.sess.get_outputs()[0].name
            
            frame_idx = 0
            
            while cap.isOpened() and self.running:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                
                # YOLO Detect
                result = self.face_detector(pil_image, verbose=False)[0]
                image_np = np.array(pil_image)

                for box in result.boxes.xyxy:
                    box = list(map(int, box.tolist()))
                    
                    y1, y2 = max(0, box[1]), min(height, box[3])
                    x1, x2 = max(0, box[0]), min(width, box[2])
                    
                    if x2 <= x1 or y2 <= y1:
                        continue

                    cropped_face = image_np[y1:y2, x1:x2]
                    
                    # Embedding 추출
                    try:
                        face_embedding = self.sess.run([output_name], {input_name: preprocess(cropped_face)})[0][0]
                        
                        is_known = False
                        # 등록된 얼굴이 하나라도 있을 때만 비교
                        if self.known_face_embeddings:
                            for known_face_embedding in self.known_face_embeddings:
                                similarity = np.dot(face_embedding, known_face_embedding.T) / (
                                    np.linalg.norm(face_embedding) * np.linalg.norm(known_face_embedding)
                                )
                                if similarity >= 0.7: 
                                    is_known = True
                                    break
                        
                        # 모르는 사람이거나, 등록된 얼굴이 아예 없으면 블러 처리
                        if not is_known:
                            k_w = max(1, (x2-x1)//5 | 1) 
                            k_h = max(1, (y2-y1)//5 | 1)
                            blurred_face = cv2.blur(cropped_face, (k_w, k_h))
                            image_np[y1:y2, x1:x2] = blurred_face
                            
                    except Exception:
                        continue

                final_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                writer.write(final_bgr)

                frame_idx += 1
                if total_frames > 0:
                    progress = int((frame_idx / total_frames) * 100)
                    self.progress_signal.emit(progress)

            cap.release()
            writer.release()
            
            if self.running:
                self.finished_signal.emit(f"완료되었습니다!\n저장 위치: {os.path.abspath(output_path)}")
            else:
                self.finished_signal.emit("작업이 취소되었습니다.")

        except Exception as e:
            self.error_signal.emit(f"처리 중 오류 발생: {str(e)}")

    def stop(self):
        self.running = False
        self.wait()

# --- Drag & Drop 위젯 ---
class DragDropLabel(QLabel):
    file_dropped = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setText("여기에 모자이크 처리 할 비디오 파일을 드래그하세요\n(또는 클릭하여 선택)")
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #aaa;
                border-radius: 10px;
                background-color: #f0f0f0;
                color: #555;
                font-size: 16px;
            }
            QLabel:hover {
                background-color: #e0e0e0;
                border-color: #555;
            }
        """)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        if files:
            self.file_dropped.emit(files[0])

    def mousePressEvent(self, event):
        file_path, _ = QFileDialog.getOpenFileName(self, "비디오 선택", "", "Video Files (*.mp4 *.avi *.mkv)")
        if file_path:
            self.file_dropped.emit(file_path)

# --- 메인 윈도우 ---
class FaceBlurApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Auto Face Blur System")
        self.resize(500, 500)
        
        self.current_video_path = None
        self.worker = None
        self.known_face_embeddings = [] # 얼굴 데이터 저장 리스트

        self.init_models()
        self.init_ui()

    def init_models(self):
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("AI 모델 로딩 중...")
        QApplication.processEvents()

        try:
            self.face_detector = YOLO(YOLO_PATH)
            self.sess = ort.InferenceSession(ARCFACE_PATH)
            self.status_bar.showMessage("모델 로딩 완료. 참조할 얼굴 폴더를 선택해주세요.")
        except Exception as e:
            QMessageBox.critical(self, "초기화 오류", f"모델 로딩 실패:\n{e}")
            sys.exit()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # 1. 얼굴 데이터 폴더 선택 영역 (수평 레이아웃)
        face_layout = QHBoxLayout()
        
        self.face_dir_btn = QPushButton("참조 얼굴 폴더 선택")
        self.face_dir_btn.setFixedHeight(40)
        self.face_dir_btn.clicked.connect(self.select_face_dir)
        
        self.face_info_label = QLabel("선택된 얼굴: 0명")
        self.face_info_label.setAlignment(Qt.AlignCenter)
        self.face_info_label.setStyleSheet("color: #333; font-weight: bold;")

        face_layout.addWidget(self.face_dir_btn)
        face_layout.addWidget(self.face_info_label)
        
        layout.addLayout(face_layout)
        
        # 구분선
        layout.addWidget(QLabel("<hr>"))

        # 2. 드래그 앤 드롭 영역
        self.drop_area = DragDropLabel()
        self.drop_area.file_dropped.connect(self.on_file_loaded)
        layout.addWidget(self.drop_area, stretch=2)

        # 3. 파일 정보 표시
        self.file_label = QLabel("선택된 비디오: 없음")
        self.file_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.file_label)

        # 4. 진행률 바
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        # 5. 실행 버튼
        self.start_btn = QPushButton("처리 시작")
        self.start_btn.setFixedHeight(50)
        self.start_btn.setEnabled(False)
        self.start_btn.clicked.connect(self.start_processing)
        layout.addWidget(self.start_btn)

    def select_face_dir(self):
        # 폴더 선택 다이얼로그
        dir_path = QFileDialog.getExistingDirectory(self, "참조할 얼굴 이미지가 있는 폴더 선택")
        
        if dir_path:
            self.load_known_faces(dir_path)

    def load_known_faces(self, dir_path):
        self.known_face_embeddings = [] # 초기화
        self.status_bar.showMessage(f"얼굴 데이터 분석 중... ({dir_path})")
        QApplication.processEvents()

        input_name = self.sess.get_inputs()[0].name
        output_name = self.sess.get_outputs()[0].name
        
        # png, jpg, jpeg 확장자 검색
        extensions = ['*.png', '*.jpg', '*.jpeg']
        sources = []
        for ext in extensions:
            sources.extend(glob(os.path.join(dir_path, ext)))

        count = 0
        for source in sources:
            try:
                image = np.array(Image.open(source).convert('RGB'))
                embedding = self.sess.run([output_name], {input_name: preprocess(image)})[0][0]
                self.known_face_embeddings.append(embedding)
                count += 1
            except Exception as e:
                print(f"Error loading {source}: {e}")

        self.face_info_label.setText(f"선택된 얼굴: {count}명")
        
        if count > 0:
            self.status_bar.showMessage(f"등록 완료: {count}명의 얼굴 데이터를 로드했습니다.")
        else:
            self.status_bar.showMessage("경고: 선택한 폴더에서 이미지를 찾을 수 없습니다.")
            QMessageBox.warning(self, "이미지 없음", "선택한 폴더에 이미지 파일(png, jpg)이 없습니다.\n모든 얼굴이 블러 처리됩니다.")

    def on_file_loaded(self, file_path):
        self.current_video_path = file_path
        self.file_label.setText(f"선택된 비디오: {os.path.basename(file_path)}")
        self.start_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.start_btn.setText("처리 시작")

    def start_processing(self):
        if not self.current_video_path:
            return

        # 얼굴 데이터가 없는 경우 경고 (선택 사항)
        if not self.known_face_embeddings:
            reply = QMessageBox.question(self, '경고', 
                                         '참조할 얼굴 데이터가 없습니다.\n영상 내의 "모든" 얼굴을 블러 처리하시겠습니까?',
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.No:
                return

        self.start_btn.setEnabled(False)
        self.drop_area.setEnabled(False)
        self.face_dir_btn.setEnabled(False) # 처리 중 폴더 변경 방지
        self.start_btn.setText("처리 중...")
        
        self.worker = VideoProcessThread(
            self.current_video_path, 
            self.face_detector, 
            self.sess, 
            self.known_face_embeddings
        )
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.finished_signal.connect(self.on_finished)
        self.worker.error_signal.connect(self.on_error)
        self.worker.start()

    def update_progress(self, val):
        self.progress_bar.setValue(val)

    def on_finished(self, msg):
        QMessageBox.information(self, "완료", msg)
        self.reset_ui()

    def on_error(self, err_msg):
        QMessageBox.critical(self, "오류", err_msg)
        self.reset_ui()

    def reset_ui(self):
        self.start_btn.setEnabled(True)
        self.drop_area.setEnabled(True)
        self.face_dir_btn.setEnabled(True)
        self.start_btn.setText("처리 시작")
        self.progress_bar.setValue(0)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = FaceBlurApp()
    window.show()
    sys.exit(app.exec_())