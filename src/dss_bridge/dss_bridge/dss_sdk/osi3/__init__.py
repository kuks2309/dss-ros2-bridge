__version__ = '3.5.0'

import sys
import os
import glob

# 현재 디렉토리 경로
current_dir = os.path.dirname(os.path.abspath(__file__))

# 실제로 존재하는 pb2 파일들만 찾기
pb2_files = glob.glob(os.path.join(current_dir, "osi_*.py"))
module_names = [os.path.basename(f)[:-3] for f in pb2_files]  # .py 확장자 제거

# 모듈 목록 설정
__all__ = module_names

# osi3 네임스페이스를 현재 모듈로 설정
sys.modules['osi3'] = sys.modules[__name__]

# 각 모듈을 현재 네임스페이스로 가져오기
for module_name in module_names:
    try:
        # 실제 모듈 임포트
        module = __import__(module_name, globals(), locals(), [], 1)
        # 현재 네임스페이스에 추가
        globals()[module_name] = module
    except ImportError:
        # 임포트 실패한 모듈은 __all__에서 제거
        if module_name in __all__:
            __all__.remove(module_name)