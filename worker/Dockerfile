FROM python:3.12-slim

WORKDIR /app
ADD worker.py /app/
RUN pip install rpyc

EXPOSE 18861
CMD ["python3", "-u", "worker.py"]
