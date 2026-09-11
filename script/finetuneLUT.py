"""WV-LUT 查找表微调入口。"""

import sys
from pathlib import Path

# 根据脚本自身位置定位项目根目录，避免依赖启动时的当前工作目录。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from script.train import main


if __name__ == "__main__":
    main(stage="finetune")
