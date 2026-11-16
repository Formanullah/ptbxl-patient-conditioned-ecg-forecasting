FROM python:3.13
ENV PYTHONUNBUFFERED=1

RUN pip install --upgrade pip 

RUN apt-get update && apt-get install

COPY requirements.txt . 

RUN pip install --no-cache-dir -r requirements.txt -v

WORKDIR /workspace

EXPOSE 8888



#CMD ["jupyter", "notebook", "--allow-root", "--ip=0.0.0.0", "--port=8888", "--no-browser"]

