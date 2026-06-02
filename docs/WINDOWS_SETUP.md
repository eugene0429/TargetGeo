# TargetGeo — Windows 환경 구축 가이드

이 문서는 Linux에서 개발된 TargetGeo 프로덕션을 **Windows 10/11 (x64)** 환경에서
구동하기 위한 설치/실행 절차를 정리한 것입니다. Linux의 `setup_env.sh` / `run_viewer.sh`에
대응하는 PowerShell 스크립트(`setup_env.ps1`, `run_viewer.ps1`)도 함께 제공됩니다.

> **요약**: 사전 요구사항 설치 → `setup_env.ps1` 실행 → HuggingFace 인증 → 탐지기 가중치 배치 → 스모크 테스트.

---

## 1. Linux와 Windows의 핵심 차이점

스크립트를 그대로 가져오면 안 되는 이유입니다. 자동화 스크립트가 이미 처리하지만, 트러블슈팅을 위해 알아두세요.

| 항목 | Linux (`setup_env.sh`) | Windows 대응 |
|---|---|---|
| 패키지 import 이름 연결 | `ln -sfn` **심볼릭 링크** (`site-packages/targetgeo → repo`) | **디렉터리 정션**(junction) `New-Item -ItemType Junction` (관리자 권한 불필요) |
| venv 파이썬 경로 | `.venv/bin/python` | `.venv\Scripts\python.exe` |
| 실행 스크립트 | bash | PowerShell (`.ps1`) |
| 탐지기 가중치 심볼릭 링크 복사 | `cp --remove-destination` | 파일을 직접 복사/배치 |
| RTSP 강제 TCP | `OPENCV_FFMPEG_CAPTURE_OPTIONS=... cmd` | `$env:OPENCV_FFMPEG_CAPTURE_OPTIONS="rtsp_transport;tcp"` |
| `pycocotools` 빌드 | 시스템 컴파일러 존재 | **MSVC C++ Build Tools** 필요할 수 있음 |
| `submitit` (학습 스택) | POSIX 네이티브 | 설치는 되나 일부 POSIX 기능 의존 — 추론 경로에서만 사용하므로 보통 무해 |

---

## 2. 사전 요구사항 (Prerequisites)

설치 **순서대로** 진행하세요.

### 2.1 NVIDIA GPU + 드라이버
- CUDA 지원 NVIDIA GPU (SAM 3.1은 ~3 GB VRAM, 전체 파이프라인은 4 GB+ 권장).
- 최신 **NVIDIA 게임/스튜디오 드라이버** 설치. (CUDA Toolkit 별도 설치 불필요 — PyTorch가 자체 CUDA 런타임을 포함합니다.)
- 확인: PowerShell에서 `nvidia-smi` 실행 → GPU와 드라이버 버전이 보여야 함.

### 2.2 Python 3.10 (64-bit)
- https://www.python.org/downloads/ 에서 **Python 3.10.x (Windows installer 64-bit)** 설치.
- 설치 시 **"Add python.exe to PATH"** 및 **"py launcher"** 체크.
- 확인:
  ```powershell
  py -3.10 --version    # Python 3.10.x 출력되어야 함
  ```
> torch CUDA 12.6 휠은 Python 3.10 기준으로 검증되었습니다. 3.11/3.12도 휠은 존재하나, 재현성을 위해 **3.10 권장**.

### 2.3 Git for Windows
- https://git-scm.com/download/win — `requirements.txt`의 `sam3 @ git+https://...` 설치에 필요.
- 확인: `git --version`

### 2.4 (조건부) Microsoft C++ Build Tools
- `pycocotools` / 일부 의존성이 사전 빌드 휠이 없을 때 컴파일이 필요합니다.
- 먼저 그냥 설치를 시도하고, `error: Microsoft Visual C++ 14.0 or greater is required` 가 뜨면 설치하세요.
- https://visualstudio.microsoft.com/visual-cpp-build-tools/ → **"Desktop development with C++"** 워크로드 선택.

### 2.5 HuggingFace 계정 + SAM 3.1 라이선스 동의
- SAM 3.1 가중치는 최초 실행 시 `huggingface.co/facebook/sam3.1` 에서 자동 다운로드됩니다. 사전에:
  1. https://huggingface.co/facebook/sam3.1 에서 **라이선스 동의 (Accept)**
  2. HuggingFace **액세스 토큰** 발급 (Settings → Access Tokens, read 권한)

---

## 3. 설치 (자동 — 권장)

리포지토리 루트(`TargetGeo\`)에서 PowerShell을 열고:

```powershell
# (최초 1회) 현재 세션에서 로컬 스크립트 실행 허용
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# 환경 구축 (.venv 새로 생성, 모든 의존성 설치, targetgeo import 연결, 스모크 체크)
.\setup_env.ps1
```

`setup_env.ps1`이 수행하는 작업 (Linux `setup_env.sh`와 1:1 대응):
1. 기존 `.venv` 제거 후 새 가상환경 생성, `pip`/`wheel`/`setuptools` 업그레이드
2. **torch + torchvision (CUDA 12.6 휠)** 설치 — `--index-url https://download.pytorch.org/whl/cu126`
3. `requirements.txt`의 나머지 의존성 설치 (`sam3` 포함)
4. `site-packages\targetgeo` → 리포지토리 루트로 **정션(junction)** 생성 → `import targetgeo` 가능
5. 탐지기 가중치(`models\target_detector.pt`) 존재 시 그대로 사용
6. 스모크 체크 (torch/cuda, cv2, numpy, sam3, targetgeo import)

다른 Python을 쓰려면:
```powershell
$env:PYTHON = "C:\Path\to\python.exe"   # 또는 "py -3.10"
.\setup_env.ps1
```

---

## 4. 설치 (수동 — 스크립트가 막힐 때)

스크립트가 어느 단계에서 실패하는지 파악하거나 세밀하게 제어할 때 사용하세요.

```powershell
# 리포 루트에서
cd C:\path\to\TargetGeo

# 1) 가상환경
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip wheel setuptools

# 2) torch (CUDA 12.6) — 반드시 index-url 사용
pip install --index-url https://download.pytorch.org/whl/cu126 torch torchvision

# 3) 나머지 의존성 (sam3 git 설치 포함)
pip install -r requirements.txt

# 4) import 이름 'targetgeo' 연결 (정션)
$site = python -c "import site; print(site.getsitepackages()[0])"
New-Item -ItemType Junction -Path (Join-Path $site 'targetgeo') -Target $PWD -Force

# 5) 탐지기 가중치 배치 (4장 참고) — models\target_detector.pt

# 6) 스모크 체크 (sam3.py 가 sam3 패키지를 가리지 않도록 중립 디렉터리에서 실행)
cd $env:TEMP
python -c "import torch,cv2,numpy as np; import sam3.model_builder; import targetgeo; from targetgeo import TargetGeoEstimator; print('torch',torch.__version__,'cuda',torch.cuda.is_available()); print('OK')"
cd C:\path\to\TargetGeo
```

> **중요 — `sam3.py` 그림자(shadowing) 문제**: 리포 루트에 `sam3.py` 파일이 있어, 현재 작업 디렉터리가 리포 루트이면 `import sam3`가 외부 `sam3` 패키지 대신 이 파일을 가리킵니다. 스모크 테스트와 import 검증은 반드시 **리포 밖(예: `$env:TEMP`)** 에서 실행하세요. 프로덕션 코드는 `targetgeo.sam3`로 명시 import하므로 영향받지 않습니다.

---

## 5. 탐지기 가중치 (YOLO rec_bbox)

가중치는 리포에 포함되지 않습니다(`.gitignore`). 배포본에서 받은 `.pt` 파일을 배치하세요:

```powershell
# 기본 경로에 복사
Copy-Item C:\path\to\target_detector.pt .\models\target_detector.pt
```

또는 코드에서 명시 경로를 전달:
```python
TargetGeoEstimator(detector_checkpoint=r"C:\weights\target_detector.pt")
```

---

## 6. HuggingFace 인증 (SAM 3.1 가중치 다운로드)

최초 SAM 호출 전에 둘 중 하나로 인증하세요.

```powershell
# 방법 A: 대화형 로그인 (토큰 붙여넣기)
.\.venv\Scripts\huggingface-cli.exe login

# 방법 B: 환경 변수 (자동화/서비스에 적합)
$env:HF_TOKEN = "hf_xxxxxxxxxxxxxxxx"          # 현재 세션
[Environment]::SetEnvironmentVariable("HF_TOKEN","hf_xxxx","User")  # 영구
```

> 사내망/프록시 환경에서 다운로드가 막히면 `HF_HOME`을 캐시 경로로 지정하고, 사전에 가중치를 받아 캐시에 배치하는 오프라인 방식을 검토하세요.

---

## 7. 실행

### 7.1 Python에서 직접 사용
`.venv` 활성화 후, **리포 루트가 아닌 곳**에서 실행하는 것이 안전합니다(§4의 shadowing 주의). 패키지로 import하므로 정상 동작합니다:

```powershell
.\.venv\Scripts\Activate.ps1
python your_app.py
```
사용 예시는 루트 [README.md](../README.md)의 *Usage* 절 참고.

### 7.2 인터랙티브 뷰어
```powershell
.\run_viewer.ps1 C:\path\to\video.mp4 --hfov-deg 60 --radius 2.5
# 또는 RTSP
.\run_viewer.ps1 rtsp://host/stream --hfov-deg 60
```

RTSP에서 UDP 디코드가 실패하면 TCP 강제:
```powershell
$env:OPENCV_FFMPEG_CAPTURE_OPTIONS = "rtsp_transport;tcp"
.\run_viewer.ps1 rtsp://host/stream --hfov-deg 60
```

> 뷰어는 GUI(Tkinter + OpenCV 창)가 필요합니다. **로컬 데스크톱 세션**에서 실행하세요. RDP/원격 세션이나 Windows Server Core에서는 디스플레이가 없어 동작하지 않을 수 있습니다.

---

## 8. 검증 (테스트)

```powershell
.\.venv\Scripts\Activate.ps1

# 빠른 테스트 (리포의 부모 디렉터리에서 실행 — README의 portability 규칙과 동일)
cd ..
python -m pytest TargetGeo\tests\ -m "not slow" -q

# E2E (실제 SAM 3.1 사용; GPU + 가중치 + HF 인증 필요)
python -m pytest TargetGeo\tests\ -m slow -q
```
> pytest는 `TargetGeo\`의 **부모 디렉터리**에서 실행하세요. 작업 디렉터리가 `sys.path`에 추가되어, 리포 내부에서 실행하면 동일 이름의 stdlib/패키지 모듈을 가립니다.

---

## 9. 트러블슈팅

| 증상 | 원인 / 해결 |
|---|---|
| `setup_env.ps1` 실행 거부 (`running scripts is disabled`) | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` 후 재실행 |
| `torch.cuda.is_available()` → `False` | NVIDIA 드라이버 미설치/구버전, 또는 CPU 휠이 설치됨. §2.1 드라이버 확인 후 `pip uninstall torch torchvision` → §4 단계 2 재설치(반드시 `--index-url ...cu126`) |
| `Microsoft Visual C++ 14.0 ... required` | §2.4 C++ Build Tools 설치 후 재실행. (`pycocotools` 등 컴파일 의존성) |
| `import sam3` 시 리포의 `sam3.py`가 잡힘 | 작업 디렉터리가 리포 루트임. 리포 밖에서 실행하거나 `targetgeo.sam3`로 import |
| `ModuleNotFoundError: targetgeo` | §4 단계 4 정션이 안 만들어짐. `site-packages` 경로 재확인 후 `New-Item -ItemType Junction ...` 재실행 |
| 정션 생성 실패(권한) | 정션은 보통 관리자 불필요. 그래도 막히면 PowerShell을 **관리자**로 실행하거나, 대안으로 `.venv\Lib\site-packages`에 `targetgeo.pth` 파일을 만들고 그 안에 리포 부모 경로를 한 줄로 기입 (단, 폴더명이 `TargetGeo`라 Windows의 대소문자 무시로 `import targetgeo`가 해석됨) |
| SAM 가중치 다운로드 실패 (401/403) | HF 라이선스 미동의 또는 토큰 누락. §2.5 / §6 재확인 |
| RTSP "no frames from stream" | §7.2 TCP 강제 옵션 적용. 방화벽/네트워크에서 RTSP 포트 확인 |
| 뷰어 창이 안 뜸 | 원격/헤드리스 세션. 로컬 데스크톱에서 실행 |
| 경로에 한글/공백 포함 시 오류 | 리포를 `C:\TargetGeo` 같은 ASCII/공백 없는 경로에 배치 권장 |

---

## 10. 빠른 시작 (Cheat Sheet)

```powershell
# 1. 사전 요구사항: Python 3.10 x64, Git, NVIDIA 드라이버, (필요시) C++ Build Tools
# 2. 리포 루트에서:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\setup_env.ps1
# 3. HF 인증
$env:HF_TOKEN = "hf_xxxx"
# 4. 탐지기 가중치 배치
Copy-Item C:\weights\target_detector.pt .\models\target_detector.pt
# 5. 실행
.\run_viewer.ps1 C:\path\to\video.mp4 --hfov-deg 60
```
