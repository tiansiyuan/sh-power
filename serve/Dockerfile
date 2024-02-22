## Build the image torchserve locally before running this, cf github torchserve:
## https://github.com/pytorch/serve/tree/master/docker
FROM pytorch/torchserve:0.5.1-cpu
USER root
RUN apt-get update
RUN apt-get install -y libgl1-mesa-glx
RUN apt-get install -y libglib2.0-0
RUN apt-get install -y python3-distutils
COPY ./ressources/ /home/model-server/ressources/
RUN chmod -R a+rw /home/model-server/
USER model-server

RUN pip3 install --upgrade pip
RUN pip install torch-model-archiver -i https://pypi.douban.com/simple/
RUN pip install opencv-python -i https://pypi.douban.com/simple/
# test opencv :
RUN python3 -c "import cv2"

RUN pip install -r /home/model-server/ressources/helmet_yolov5/requirements.txt -i https://pypi.douban.com/simple/
EXPOSE 8080 8081
ENV PYTHONPATH "${PYTHONPATH}:/home/model-server/ressources/helmet_yolov5/"
RUN python /home/model-server/ressources/helmet_yolov5/export.py --weights /home/model-server/ressources/helmet.pt --img 640 --batch 1

RUN torch-model-archiver --model-name helmet_detection \
--version 0.1 --serialized-file /home/model-server/ressources/helmet.torchscript.pt \
--handler /home/model-server/ressources/torchserve_handler.py \
--extra-files /home/model-server/ressources/index_to_name.json,/home/model-server/ressources/torchserve_handler.py

# RUN torch-model-archiver --model-name helmet_detection \
# --version 0.1 --serialized-file /home/model-server/ressources/helmet.torchscript.pt \
# --handler image_classifier \
# --extra-files /home/model-server/ressources/index_to_name.json


RUN mv helmet_detection.mar model-store/helmet_detection.mar
CMD [ "torchserve", "--start", "--model-store", "model-store", "--models", "my_model_name=helmet_detection.mar" ]