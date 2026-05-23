FROM python:3.10-slim

WORKDIR /app

# Install system dependencies required for building C++ extensions (like ChromaDB's hnswlib)
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose ports for both API and Streamlit
EXPOSE 8000
EXPOSE 8501

# The default command will be overridden by docker-compose
CMD ["bash"]
