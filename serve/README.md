# 使用 TorchServe 提供模型服务

按照是否使用 Docker 分为两种方式，本质是一样的。学习测试，推荐不用 Docker 的，生产环境，推荐使用 Docker 的。



### 基础环境

- Ubuntu 22.04 或者 20.04
- Python 3.8

```sh
cat /etc/lsb-release

sudo apt update && sudo apt install python3.8
```
- 如果需要，[安装 Docker](https://docs.docker.com/engine/install/ubuntu/)



### 打包模型

```sh
# 将训练好的模型（如在notebook 目录中的步骤训练的best.pt）拷贝到当前目录并改名为：helmet.torchscript.pt, 现有的可作为练习用
# 注意在当前目录里还应包含以下文件：
# torchserve_handler.py
# index_to_name.json
# 运行下面的脚本：
archiver_model.sh
# 上面的脚本运行后，会创建 model_store 目录，其中包含文件 helmet_detection.mar
```



### 启动服务

#### 不用 Docker

```sh
sudo apt-get install openjdk-11-jdk

pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

torchserve --start --ncs --model-store model_store --models helmet_detection.mar
```

#### 使用 Docker

```sh
docker build . -t helmet_yolov5_torchserve:v1

docker run -p 8080:8080 -p 8081:8081 helmet_yolov5_torchserve:v1
```



### 注册模型

```sh
curl -X POST  "http://localhost:8081/models?url=./model_store/helmet_detection.mar&model_name=helmet_detection"
curl -X PUT "http://0.0.0.0:8081/models/helmet_detection?min_worker=3"
```



### 检测

```sh
python detect.py --input <输入图片文件名> --output <输出图片文件名>

# 默认的输入输出分别为: test_1.jpg 和 result_test_1.jpg
```



### 终止服务

#### 不用 Docker

```sh
Ctrl-C torchserve 命令
```

#### 使用 Docker

```sh
docker container ls # 列出运行模型服务的容器 ID

docker stop <上面命令列出的 ID>
```

