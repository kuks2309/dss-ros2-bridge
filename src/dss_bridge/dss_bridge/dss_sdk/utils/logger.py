import logging

# 커스텀 레벨 정의
SDK_LEVEL = 25
logging.addLevelName(SDK_LEVEL, 'DSSSDK_PY')

def sdk(self, message, *args, **kwargs):
    if self.isEnabledFor(SDK_LEVEL):
        self._log(SDK_LEVEL, message, args, **kwargs)

# Logger 클래스에 메서드 추가
logging.Logger.sdk = sdk

# 설정
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger("dss_sdk")
logger.setLevel(logging.INFO)
