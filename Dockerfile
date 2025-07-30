# Use a base image with Python and PyTorch
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    libhdf5-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .
COPY test_requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r test_requirements.txt

RUN pip install torch_geometric torch_scatter torch_sparse torch_cluster torch_spline_conv -f [https://data.pyg.org/whl/torch-2.0.1+cu117.html](https://data.pyg.org/whl/torch-2.0.1+cu117.html)
COPY . .

# CMD ["python", "src/pretrain.py"]