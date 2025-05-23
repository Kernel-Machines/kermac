import struct

class FunctionDBValue:
    """Represents the value structure (lowered_name, cubin_data_hash) for the function database."""
    def __init__(self, lowered_name: bytes, cubin_data_hash: bytes):
        self.lowered_name = lowered_name
        self.cubin_data_hash = cubin_data_hash

    def to_bytes(self) -> bytes:
        """Serialize the value to a bytes object for LMDB storage."""
        lowered_len = len(self.lowered_name)
        return struct.pack('>I', lowered_len) + self.lowered_name + self.cubin_data_hash

    @classmethod
    def from_bytes(cls, data: bytes) -> 'FunctionDBValue':
        """Deserialize bytes to a FunctionDBValue object."""
        lowered_len = struct.unpack('>I', data[:4])[0]
        lowered_name = data[4:4 + lowered_len]
        cubin_data_hash = data[4 + lowered_len:]
        return cls(lowered_name=lowered_name, cubin_data_hash=cubin_data_hash)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FunctionDBValue):
            return False
        return (
            self.lowered_name == other.lowered_name and
            self.cubin_data_hash == other.cubin_data_hash
        )

    def __repr__(self) -> str:
        return f"FunctionDBValue(lowered_name={self.lowered_name!r}, cubin_data_hash={self.cubin_data_hash!r})"
