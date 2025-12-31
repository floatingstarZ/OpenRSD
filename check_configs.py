import os
import re
from pathlib import Path

# 读取 model_ad_config.py
config_file = Path('EXP_CONFIG/CONFIGS/model_ad_config.py')
content = config_file.read_text(encoding='utf-8')

# 获取 M_configs 目录下所有实际存在的配置文件
m_configs_root = Path('M_configs')
existing_files = set()
for py_file in m_configs_root.rglob('*.py'):
    rel_path = str(py_file.relative_to(m_configs_root)).replace('\\', '/')
    existing_files.add(rel_path)

print(f"找到 {len(existing_files)} 个实际存在的配置文件")

# 提取所有 gen_dict 调用中的配置文件路径
# 匹配格式: root + '目录/文件名.py'
pattern = r"root\s*\+\s*['\"]([^'\"]+\.py)['\"]"
matches = re.findall(pattern, content)

print(f"\n在 model_ad_config.py 中找到 {len(matches)} 个配置路径")

# 检查哪些配置存在
existing_configs = []
missing_configs = []

for config_path in matches:
    if config_path in existing_files:
        existing_configs.append(config_path)
    else:
        missing_configs.append(config_path)

print(f"\n存在的配置: {len(existing_configs)}")
print(f"缺失的配置: {len(missing_configs)}")

# 输出缺失的配置（前20个）
if missing_configs:
    print("\n缺失的配置示例（前20个）:")
    for cfg in missing_configs[:20]:
        print(f"  - {cfg}")

# 输出存在的配置
print("\n存在的配置:")
for cfg in sorted(existing_configs):
    print(f"  - {cfg}")

