import json

class FunctionDBKey:
    """Represents the key structure for the function database."""
    def __init__(
        self,
        package_name: str,
        package_version: str,
        cuda_version: str,
        arch: str,
        function_name: str
    ):
        self.package_name = package_name
        self.package_version = package_version
        self.cuda_version = cuda_version
        self.arch = arch
        self.function_name = function_name

    def to_bytes(self) -> bytes:
        """Serialize the key to a bytes object for LMDB storage."""
        key_dict = {
            'package_name': self.package_name,
            'package_version': self.package_version,
            'cuda_version': self.cuda_version,
            'arch': self.arch,
            'function_name': self.function_name
        }
        return json.dumps(key_dict, sort_keys=True).encode('utf-8')

    @classmethod
    def from_bytes(cls, data: bytes) -> 'FunctionDBKey':
        """Deserialize bytes to a FunctionDBKey object."""
        key_dict = json.loads(data.decode('utf-8'))
        return cls(
            package_name=key_dict['package_name'],
            package_version=key_dict['package_version'],
            cuda_version=key_dict['cuda_version'],
            arch=key_dict['arch'],
            function_name=key_dict['function_name']
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FunctionDBKey):
            return False
        return (
            self.package_name == other.package_name and
            self.package_version == other.package_version and
            self.cuda_version == other.cuda_version and
            self.arch == other.arch and
            self.function_name == other.function_name
        )

    def __repr__(self) -> str:
        return (f"FunctionDBKey(package_name={self.package_name}, "
                f"package_version={self.package_version}, "
                f"cuda_version={self.cuda_version}, "
                f"arch={self.arch}, "
                f"function_name={self.function_name})")
