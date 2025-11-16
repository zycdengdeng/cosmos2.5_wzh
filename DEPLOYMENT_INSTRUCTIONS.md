# 部署到服务器的说明

## 问题诊断

训练错误显示：
```
MissingConfigException: Could not find 'conditioner/custom_multiview_vis_conditioner'
```

**根本原因**：自定义配置文件在 git 仓库中，但还没有复制到服务器训练目录。

训练脚本路径：`/mnt/zihanw/cosmos-transfer2.5/` （服务器）
Git 仓库路径：`/home/user/cosmos2.5_wzh/` （本地）

## 解决方案

有两种方法将文件部署到服务器：

### 方法 1: 使用复制脚本（推荐）

```bash
# 运行一键复制脚本
./copy_to_server.sh
```

此脚本会：
1. 检查源目录和目标目录是否存在
2. 复制所有必需的自定义配置文件
3. 验证文件已成功复制

### 方法 2: 在服务器上 Git Pull

如果 `/mnt/zihanw/cosmos-transfer2.5/` 是一个 git 仓库：

```bash
cd /mnt/zihanw/cosmos-transfer2.5/

# 获取最新更改
git fetch origin claude/review-library-posttraining-011CV3cyEQyHkLwftwajiYbV

# 切换到正确的分支
git checkout claude/review-library-posttraining-011CV3cyEQyHkLwftwajiYbV

# 拉取最新更改
git pull origin claude/review-library-posttraining-011CV3cyEQyHkLwftwajiYbV
```

### 方法 3: 手动复制文件

如果你需要手动复制，以下是需要复制的文件：

```bash
# 1. __init__.py (已修改 - 注册自定义配置)
cp /home/user/cosmos2.5_wzh/cosmos_transfer2/experiments/multiview/__init__.py \
   /mnt/zihanw/cosmos-transfer2.5/cosmos_transfer2/experiments/multiview/__init__.py

# 2. custom_conditioners.py (新文件 - 核心功能)
cp /home/user/cosmos2.5_wzh/cosmos_transfer2/experiments/multiview/custom_conditioners.py \
   /mnt/zihanw/cosmos-transfer2.5/cosmos_transfer2/experiments/multiview/custom_conditioners.py

# 3. custom_datasets.py (已存在 - 数据集加载器)
cp /home/user/cosmos2.5_wzh/cosmos_transfer2/experiments/multiview/custom_datasets.py \
   /mnt/zihanw/cosmos-transfer2.5/cosmos_transfer2/experiments/multiview/custom_datasets.py

# 4. custom_experiments.py (已修改 - 实验配置)
cp /home/user/cosmos2.5_wzh/cosmos_transfer2/experiments/multiview/custom_experiments.py \
   /mnt/zihanw/cosmos-transfer2.5/cosmos_transfer2/experiments/multiview/custom_experiments.py
```

## 验证部署

复制完成后，验证文件是否存在：

```bash
ls -la /mnt/zihanw/cosmos-transfer2.5/cosmos_transfer2/experiments/multiview/

# 应该看到：
# __init__.py
# custom_conditioners.py  (新文件)
# custom_datasets.py
# custom_experiments.py
```

验证 custom_conditioners.py 内容是否正确：

```bash
grep -n "CustomMultiViewVisConditioner\|CustomMultiViewDepthConditioner" \
  /mnt/zihanw/cosmos-transfer2.5/cosmos_transfer2/experiments/multiview/custom_conditioners.py

# 应该看到这两个 conditioner 的定义
```

验证注册函数是否被调用：

```bash
grep -n "register_custom_conditioners" \
  /mnt/zihanw/cosmos-transfer2.5/cosmos_transfer2/experiments/multiview/__init__.py

# 应该看到：
# custom_conditioners.register_custom_conditioners()
```

## 需要复制的关键文件说明

### 1. custom_conditioners.py (新文件 - 最重要)

这个文件定义了两个自定义 conditioner：

- **CustomMultiViewVisConditioner**: 将 `control_input_hdmap_bbox` 目录映射到 `vis` 控制分支
- **CustomMultiViewDepthConditioner**: 将 `control_input_hdmap_bbox` 目录映射到 `depth` 控制分支

这解决了官方 multiview 示例只支持 `hdmap_bbox` 的限制。

### 2. __init__.py (已修改)

添加了对自定义配置的导入和注册：

```python
from cosmos_transfer2.experiments.multiview import custom_conditioners, custom_datasets, custom_experiments

custom_conditioners.register_custom_conditioners()
custom_datasets.register_custom_datasets()
```

### 3. custom_experiments.py (已修改)

- 设置 `context_parallel_size=2` 以匹配 2 GPU 配置
- 引用自定义 conditioner：`custom_multiview_vis_conditioner` 和 `custom_multiview_depth_conditioner`
- 设置正确的 `hint_keys`："vis" 和 "depth"

### 4. custom_datasets.py (已存在)

- 定义 RGBCloud 和 DepthSparse 数据集加载器
- `hint_key="control_input_hdmap_bbox"` 匹配实际目录名

## 部署后启动训练

文件复制完成后，就可以开始训练了：

```bash
cd /mnt/zihanw/cosmos-transfer2.5/

# RGBCloud 训练 (vis control)
./train_rgbcloud.sh

# 或

# DepthSparse 训练 (depth control)
./train_depthsparse.sh
```

## 如果还是出现错误

如果复制后还是出现 `MissingConfigException`，请检查：

1. **Python 缓存**：删除 `__pycache__` 目录
   ```bash
   find /mnt/zihanw/cosmos-transfer2.5/cosmos_transfer2/experiments/multiview/ -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
   ```

2. **文件内容**：确保 custom_conditioners.py 没有语法错误
   ```bash
   python3 -m py_compile /mnt/zihanw/cosmos-transfer2.5/cosmos_transfer2/experiments/multiview/custom_conditioners.py
   ```

3. **导入路径**：确保 __init__.py 正确导入了 custom_conditioners
   ```bash
   cd /mnt/zihanw/cosmos-transfer2.5/
   python3 -c "from cosmos_transfer2.experiments.multiview import custom_conditioners; print('✓ Import successful')"
   ```

## 文件清单

以下是需要确保存在于服务器上的所有自定义文件：

```
/mnt/zihanw/cosmos-transfer2.5/
├── cosmos_transfer2/experiments/multiview/
│   ├── __init__.py                    (修改) - 注册函数调用
│   ├── custom_conditioners.py         (新建) - 自定义 conditioner 映射
│   ├── custom_datasets.py             (已存在) - 数据集加载器
│   └── custom_experiments.py          (修改) - 实验配置
├── build_dataset.sh                   (已存在) - RGBCloud 数据集构建
├── build_dataset_depthsparse.sh       (已存在) - DepthSparse 数据集构建
├── train_rgbcloud.sh                  (已存在) - RGBCloud 训练脚本
└── train_depthsparse.sh               (已存在) - DepthSparse 训练脚本
```

## 联系支持

如果按照上述步骤操作后仍然遇到问题，请提供：

1. `ls -la` 输出，显示服务器上的文件列表
2. 训练启动时的完整错误信息
3. Python 导入测试的结果
