import lmdb
import os
from typing import Dict, Optional, Any
import json
import hashlib
import pickle

class DiskCache():
    def __init__(self, cache_dir, max_size_mb, db_name):
        os.makedirs(cache_dir, exist_ok=True)
        db_max_size_bytes = max_size_mb * 1024 * 1024
        self.env = lmdb.open(cache_dir, map_size=db_max_size_bytes, max_dbs=1)
        self.db = self.env.open_db(db_name.encode())

    @staticmethod
    def _serialize_key(params: Dict[str, Any]) -> str:
        json_str = json.dumps(params, sort_keys=True)
        hash_suffix = hashlib.sha1(json_str.encode()).hexdigest()[:8]
        return f"{json_str}_{hash_suffix}"

    def store(self, params: Dict[str, Any], data) -> None:
        key = self._serialize_key(params).encode()
        with self.env.begin(write=True, db=self.db) as txn:
            serialized_data = pickle.dumps(data)
            txn.put(key, serialized_data)

    def lookup(self, params: Dict[str, Any]) -> Optional[bytes]:
        key_str = self._serialize_key(params)
        with self.env.begin(db=self.db) as txn:
            serialized_data = txn.get(key_str.encode())
            if serialized_data:
                return pickle.loads(serialized_data)
            return None

    def clear(self) -> None:
        """
        Drop all entries from the LMDB database (but keep the env files).
        """
        with self.env.begin(write=True) as txn:
            txn.drop(self.db, delete=False)  # delete=True would remove the DB files

    # ------------------------------------------------------------------ cleanup
    def __del__(self):
        try:
            self.env.close()
        except Exception:
            pass
