FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Keep container running
COPY start.sh .
RUN sed -i 's/\r$//' start.sh
RUN chmod +x start.sh
CMD ["./start.sh"]
