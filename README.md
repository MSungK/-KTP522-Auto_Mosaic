# [KTP522] 개인 프로젝트

## Installation
```bash
conda create -n auto_mosaic python=3.10 -y
conda activate auto_mosaic

git clone https://github.com/MSungK/-KTP522-Auto_Mosaic.git
cd -KTP522-Auto_Mosaic
pip install -r requirements.txt
```

## System Execution

```bash
python GUI.py
```

위 코드를 실행하면 제공되는 GUI 화면을 마주할 수 있습니다.

아래 설명에서는 GUI 화면에 표시된 각 버튼에 대한 설명을 제공합니다.

1. **‘참조 얼굴 폴더 선택’** : 모자이크 처리에 제외할 사용자 얼굴 이미지들이 남긴 디렉토리를 선택하세요.
2. **‘여기에 모자이크 처리 할 비디오 파일을 드래그하세요’** : 모자이크 처리를 적용할 비디오 파일을 선택하세요.
3. **‘처리 시작’** : 1, 2 단계에서 선택된 입력을 기반으로 모자이크 처리를 시작합니다.
4. **‘0-100%’** : 현재 모자이크 처리 진행률을 표시합니다.
5. 모자이크 처리된 결과물은 **‘output_processed.mp4’** 파일에서 확인할 수 있습니다.