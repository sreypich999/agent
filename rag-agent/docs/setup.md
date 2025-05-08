# Setup Guide

1. Install MinIO:
```bash
# Windows
curl -O https://dl.min.io/server/minio/release/windows-amd64/minio.exe
mkdir C:\minio-data
minio.exe server C:\minio-data --console-address :9001

# Linux/Mac
wget https://dl.min.io/server/minio/release/linux-amd64/minio
chmod +x minio
./minio server ./minio-data --console-address :9001