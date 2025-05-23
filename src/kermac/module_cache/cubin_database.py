import lmdb
import os

from typing import Optional

from .function_db_key import *
from .function_db_value import *

class CubinDatabase:
    """Manages the LMDB database for cubin data and source files hash."""
    def __init__(
            self, 
            cache_dir: str, 
            max_size_mb,
            current_file_src_hash,
            debug = False
        ):  # 10MB default
        os.makedirs(cache_dir, exist_ok=True)
        db_max_size_bytes = max_size_mb * 1024 * 1024
        self.env = lmdb.open(cache_dir, map_size=db_max_size_bytes, max_dbs=2)
        self.function_map_db = self.env.open_db(b'function_map_db')  # First mapping: key -> (lowered_name, cubin_data_hash)
        self.data_db = self.env.open_db(b'data_db')    # Second mapping: data_hash -> cubin_data, plus source files hash
        self._put_source_files_hash(current_file_src_hash, debug = debug)

    def put_function_mapping(self, key: FunctionDBKey, value: FunctionDBValue) -> None:
        """Store a key-value pair in the function_map_db."""
        with self.env.begin(write=True, db=self.function_map_db) as txn:
            txn.put(key.to_bytes(), value.to_bytes())

    def get_function_mapping(self, key: FunctionDBKey) -> Optional[FunctionDBValue]:
        """Retrieve a value from the function_map_db by key."""
        with self.env.begin(db=self.function_map_db) as txn:
            data = txn.get(key.to_bytes())
            return FunctionDBValue.from_bytes(data) if data is not None else None

    def put_cubin(self, data_hash: bytes, cubin_data: bytes) -> None:
        """Store a data_hash to cubin_data mapping in the data_db."""
        with self.env.begin(write=True, db=self.data_db) as txn:
            txn.put(data_hash, cubin_data)

    def get_cubin(self, data_hash: bytes) -> Optional[bytes]:
        """Retrieve cubin_data by data_hash from the data_db."""
        with self.env.begin(db=self.data_db) as txn:
            return txn.get(data_hash)

    def _put_source_files_hash(self, source_hash: bytes, debug = False) -> bool:
        """Store the source files hash in the data_db, clearing databases if hash doesn't match."""
        with self.env.begin(write=True) as txn:
            stored_hash = txn.get(b'source_files_hash', db=self.data_db)
            if stored_hash is not None and stored_hash != source_hash:
                if debug:
                    print(f"(Kermac Debug) File source hash mismatch (stored: {stored_hash}, provided: {source_hash}).")
                    print(f"(Kermac Debug) Clearing database of pre-built cubin entries")
                # Hash mismatch: drop both databases
                txn.drop(self.function_map_db, delete=False)  # Clear function_map_db
                txn.drop(self.data_db, delete=False)   # Clear data_db (including old hash)
                # Store new hash
                txn.put(b'source_files_hash', source_hash, db=self.data_db)
                if debug:
                    print(f"(Kermac Debug) Updated stored src hash to: {source_hash}")
                return True  # Indicate databases were cleared
            else:
                # No mismatch or no stored hash: just store the new hash
                txn.put(b'source_files_hash', source_hash, db=self.data_db)
                if debug:
                    print("(Kermac Debug) Hashes match. Keeping database pre-built cubin entries.")
                return False  # Indicate no clearing was needed

    def _get_source_files_hash(self) -> Optional[bytes]:
        """Retrieve the source files hash from the data_db."""
        with self.env.begin(db=self.data_db) as txn:
            return txn.get(b'source_files_hash')

    def close(self):
        """Close the LMDB environment."""
        self.env.close()
