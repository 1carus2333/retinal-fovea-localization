# 黄斑中央凹定位

眼底图像中央凹坐标预测基线。程序使用 ResNet18 特征提取网络、热图定位分支和可见性分支，支持五折训练及按比赛模板生成坐标提交文件。

## 实验结果

- 数据：80 张训练图像，20 张测试图像。
- 五折交叉验证 OOF RMSE：16.5216 像素（随机种子 42）。
- 测试预测文件：20 张图对应 40 行 X/Y 坐标，模板顺序及坐标边界检查通过。
- Kaggle 截图成绩：Public Score 10.70534，Private Score 21.61345。最终排名以比赛私榜规则为准。

训练数据没有中央凹不可见 `(0,0)` 样本，因此不可见分类分支缺乏负例监督。第五折后期验证误差波动明显；交叉验证分数用于参考，不保证测试榜表现。测试真值不可用，无法计算本地测试 RMSE。

## 文件

- `fovea.py`：训练与预测入口。
- `requirements.txt`：依赖版本。
- `setup.ps1`：Windows Python 3.11 环境安装脚本，配置 PyTorch CUDA 12.4。
- `environment-lock.txt`：本次运行环境实际安装版本。
- `results.json`：交叉验证、测试文件检查和 Kaggle 成绩摘要。

## 数据放置

从比赛数据压缩包中解压到项目目录的 `data/`：

```text
data/
  train/                         # 训练眼底图
  test/                          # 测试眼底图
  train_location/                # 可选 XML 标注；当前程序不读取
  fovea_localization_train_GT.csv # data,Fovea_X,Fovea_Y
  sample_submission.csv          # ImageID,value
```

图像名和标签编号应对应。程序兼容 `0001.jpg` 与 CSV 中的 `1` 这种前导零差异。坐标使用原图像素，X 向右，Y 向下；`(0,0)` 表示不可见。

## 安装

在 Windows PowerShell 中：

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

安装需要访问 PyPI 和 PyTorch 官方下载源。GPU 可用时默认使用 CUDA，否则代码支持 CPU 模式。

## 训练

```powershell
.\.venv\Scripts\python.exe .\fovea.py train --data .\data --output .\outputs --device cuda
```

默认参数为 512×512 输入、batch size 4、80 个 epoch、5 折、随机种子 42。训练生成每折最佳权重、每轮记录、OOF 预测和评估 JSON。

## 生成提交文件

```powershell
.\.venv\Scripts\python.exe .\fovea.py predict --data .\data --output .\outputs --submission .\outputs\submission.csv --device cuda
```

预测使用 `outputs/` 中的五折最佳权重，按模板保留 ImageID 和行顺序，只写入预测 value。

## 上传说明

本源码包不包含比赛图片和标签、虚拟环境、预训练权重缓存、训练权重或个人提交账户信息。训练时需自行取得比赛数据；首次训练还会下载 ResNet18 ImageNet 预训练权重。
