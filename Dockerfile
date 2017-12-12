FROM continuumio/miniconda3

RUN mkdir /opt/deepdream
COPY src /opt/deepdream/src
COPY requirements.txt /opt/deepdream/

WORKDIR /opt/deepdream
RUN pip install -r requirements.txt

CMD [ "jupyter", "notebook", "--ip='*'", "--port=9999", "--no-browser", "--allow-root", "src"]
# CMD [ "/bin/bash"]