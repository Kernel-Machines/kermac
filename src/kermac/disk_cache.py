import lmdb
import os
from typing import Dict, Optional, Any, List
import json
import hashlib
import pickle

class DiskCache():
    def __init__(
            self, 
            cache_dir, 
            max_size_mb, 
            db_name, 
            current_file_src_hash,
            debug = False
        ):
        os.makedirs(cache_dir, exist_ok=True)
        db_max_size_bytes = max_size_mb * 1024 * 1024
        self.env = lmdb.open(cache_dir, map_size=db_max_size_bytes, max_dbs=1)
        self.db = self.env.open_db(db_name.encode())

        """Check if provided hash matches stored hash; clear DB and update if needed."""
        with self.env.begin(write=True, db=self.db) as txn:
            stored_hash = txn.get(b'file_source_hash')
            stored_hash = stored_hash.decode() if stored_hash else None

            if stored_hash != current_file_src_hash:
                if debug:
                    print(f"(Kermac Debug) File source hash mismatch (stored: {stored_hash}, provided: {current_file_src_hash}).")
                    print(f"(Kermac Debug) Clearing database of pre-built cubin entries")
                txn.drop(self.db, delete=False)  # Clear all entries
                txn.put(b'file_source_hash', current_file_src_hash.encode())
                if debug:
                    print(f"(Kermac Debug) Updated stored src hash to: {current_file_src_hash}")
            else:
                if debug:
                    print("(Kermac Debug) Hashes match. Keeping database pre-built cubin entries.")

    @staticmethod
    def _serialize_key(params: Dict[str, Any]) -> str:
        json_str = json.dumps(params, sort_keys=True)
        hash_suffix = hashlib.sha1(json_str.encode()).hexdigest()[:8]
        return f"{json_str}_{hash_suffix}"

    def store(self, params: Dict[str, Any], data) -> None:
        key = self._serialize_key(params).encode()
        serialized_data = pickle.dumps(data)
        data_hash = hashlib.sha256(serialized_data).digest()
        
        with self.env.begin(write=True, db=self.db) as txn:
            # Store key -> hash
            txn.put(key, data_hash)
            # Store hash -> serialized data
            txn.put(data_hash, serialized_data)

    # Store multiple functions that all point to the same cubin
    # This allows sending multiple function signatures to jit 
    # all getting embedded in to the same cubin
    def store_multiple(self, params: List[Dict[str, Any]], data) -> None:
        serialized_data = pickle.dumps(data)
        data_hash = hashlib.sha256(serialized_data).digest()
        
        with self.env.begin(write=True, db=self.db) as txn:
            # Store key -> hash for each params dict
            for param in params:
                key = self._serialize_key(param).encode()
                txn.put(key, data_hash)
            # Store hash -> serialized data
            txn.put(data_hash, serialized_data)

    def lookup_module(self, params: Dict[str, Any]) -> Optional[bytes]:
        key_str = self._serialize_key(params)
        with self.env.begin(db=self.db) as txn:
            data_hash = txn.get(key_str.encode())
            if data_hash:
                return data_hash
            return None
        
    def lookup_cubin(self, data_hash: bytes) -> Optional[bytes]:
        with self.env.begin(db=self.db) as txn:
            serialized_data = txn.get(data_hash)
            if serialized_data:
                # lookup hash -> binary
                return pickle.loads(serialized_data)
            return None

    def lookup(self, params: Dict[str, Any]) -> Optional[bytes]:
        key_str = self._serialize_key(params)
        with self.env.begin(db=self.db) as txn:
            data_hash = txn.get(key_str.encode())
            if data_hash:
                # lookup key -> hash
                serialized_data = txn.get(data_hash)
                if serialized_data:
                    # lookup hash -> binary
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
