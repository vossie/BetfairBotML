# (A) Python env
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip

# (B) Core libs
pip install pyarrow polars s3fs fsspec boto3 scikit-learn xgboost lightgbm \
            matplotlib numpy pandas

# (C) (Optional deep models) PyTorch (GPU build) – pick the CUDA build matching your driver
# Example (you may need to adjust the cuXX index URL per your system):
# pip install torch --index-url https://download.pytorch.org/whl/cu121
