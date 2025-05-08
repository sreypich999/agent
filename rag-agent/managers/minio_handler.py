# managers/minio_handler.py
import os
import time
import threading
import yaml
from minio import Minio
from minio.error import S3Error

class MinIOHandler:
    def __init__(self):
        cfg = yaml.safe_load(open("config/settings.yaml"))["minio"]
        endpoint = os.getenv("MINIO_ENDPOINT", cfg["endpoint"])
        bucket_name = os.getenv("MINIO_BUCKET_NAME", cfg["bucket_name"])
        access_key = os.getenv("MINIO_ACCESS_KEY")
        secret_key = os.getenv("MINIO_SECRET_KEY")

        if not access_key or not secret_key:
            raise RuntimeError("Missing MinIO credentials")

        self.client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=cfg.get("secure", True)
        )
        self.bucket_name = bucket_name
        self.scan_interval = cfg.get("scan_interval", 30)
        self._ensure_bucket()

    def _ensure_bucket(self):
        if not self.client.bucket_exists(self.bucket_name):
            self.client.make_bucket(self.bucket_name)

    def list_all_objects(self):
        """Return an iterable of all objects currently in the bucket."""
        return self.client.list_objects(self.bucket_name, recursive=True)

    def get_object_data(self, name):
        """Fetch and return the raw bytes of a given object."""
        return self.client.get_object(self.bucket_name, name).read()

    def watch_bucket(self, callback):
        def watcher():
            known = set()
            while True:
                try:
                    current = {obj.object_name for obj in self.client.list_objects(self.bucket_name)}
                    new = current - known
                    for name in new:
                        data = self.client.get_object(self.bucket_name, name).read()
                        callback(data, name)
                    known = current
                except S3Error as e:
                    print(f"MinIO Error: {e}")
                time.sleep(self.scan_interval)
        threading.Thread(target=watcher, daemon=True).start()
