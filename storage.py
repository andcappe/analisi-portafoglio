"""
Storage persistente per sessioni, file clienti e cache prezzi.

Se sono impostate le variabili S3_* i file vengono salvati su un bucket
S3-compatibile (Cloudflare R2, AWS S3, Backblaze B2...), quindi sopravvivono
ai deploy di Render. Altrimenti si usa una cartella locale (default: sessions/),
comodo per lo sviluppo.

Variabili d'ambiente:
  S3_BUCKET             nome del bucket (attiva lo storage remoto)
  S3_ENDPOINT_URL       es. https://<account_id>.r2.cloudflarestorage.com
  S3_ACCESS_KEY_ID      chiave di accesso
  S3_SECRET_ACCESS_KEY  chiave segreta
  S3_REGION             default "auto" (corretto per R2)
  S3_PREFIX             prefisso opzionale delle chiavi, es. "prod/"
  DATA_DIR              cartella locale usata quando S3_BUCKET non è impostata
"""
import os
from pathlib import Path


class _LocalStorage:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def describe(self):
        return f"disco locale {self.root}"

    def read(self, name):
        p = self.root / name
        return p.read_bytes() if p.exists() else None

    def write(self, name, data):
        p = self.root / name
        tmp = p.with_name(p.name + '.tmp')
        tmp.write_bytes(data)
        os.replace(tmp, p)  # scrittura atomica

    def delete(self, name):
        p = self.root / name
        if p.exists():
            p.unlink()

    def list(self, prefix=''):
        return sorted(p.name for p in self.root.glob(f'{prefix}*')
                      if p.is_file() and not p.name.endswith('.tmp'))


class _S3Storage:
    def __init__(self, bucket, endpoint_url, access_key, secret_key, region, prefix):
        import boto3
        from botocore.config import Config
        self.bucket = bucket
        self.prefix = prefix
        self.client = boto3.client(
            's3',
            endpoint_url=endpoint_url or None,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
            config=Config(retries={'max_attempts': 5, 'mode': 'standard'}),
        )

    def describe(self):
        return f"bucket S3 {self.bucket}/{self.prefix}"

    def _key(self, name):
        return f'{self.prefix}{name}'

    def read(self, name):
        try:
            obj = self.client.get_object(Bucket=self.bucket, Key=self._key(name))
        except self.client.exceptions.NoSuchKey:
            return None
        return obj['Body'].read()

    def write(self, name, data):
        self.client.put_object(Bucket=self.bucket, Key=self._key(name), Body=data)

    def delete(self, name):
        self.client.delete_object(Bucket=self.bucket, Key=self._key(name))

    def list(self, prefix=''):
        names = []
        paginator = self.client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self._key(prefix)):
            for obj in page.get('Contents', []):
                names.append(obj['Key'][len(self.prefix):])
        return sorted(names)


def _make_storage():
    bucket = os.environ.get('S3_BUCKET')
    if bucket:
        prefix = os.environ.get('S3_PREFIX', '')
        if prefix and not prefix.endswith('/'):
            prefix += '/'
        return _S3Storage(
            bucket=bucket,
            endpoint_url=os.environ.get('S3_ENDPOINT_URL'),
            access_key=os.environ.get('S3_ACCESS_KEY_ID'),
            secret_key=os.environ.get('S3_SECRET_ACCESS_KEY'),
            region=os.environ.get('S3_REGION', 'auto'),
            prefix=prefix,
        )
    root = os.environ.get('DATA_DIR') or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'sessions')
    return _LocalStorage(root)


storage = _make_storage()
print(f"✓ Storage dati: {storage.describe()}")
