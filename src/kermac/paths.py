from platformdirs import user_cache_dir
import os
import sys
import importlib.resources
from pathlib import Path
from importlib.metadata import version, distributions
import hashlib, os, uuid, json, pathlib, sys

def get_package_name():
    return "kermac"

def get_package_version():
    try:
        pkg_version = version(get_package_name())  # Replace with your package name
        return pkg_version
    except Exception as e:
        return f"Could not determine version: {e}"

def get_cache_ptx_dir() -> Path:
    # Get the user-specific cache directory for your package
    cache_dir = user_cache_dir(get_package_name())
    
    # Create a subdirectory for PTX files
    ptx_cache_dir = os.path.join(cache_dir, "ptx")
    
    # Create the directory if it doesn't exist
    os.makedirs(ptx_cache_dir, exist_ok=True)
    
    return Path(ptx_cache_dir)

def get_cache_cubin_dir() -> Path:
    # Get the user-specific cache directory for your package
    cache_dir = user_cache_dir(get_package_name())
    
    # Create a subdirectory for CUBIN files
    cubin_cache_dir = os.path.join(cache_dir, "cubin")
    
    # Create the directory if it doesn't exist
    os.makedirs(cubin_cache_dir, exist_ok=True)
    
    return Path(cubin_cache_dir)

def get_top_level_repo_dir(dir) -> Path:
    # directory *beside* the package (wheel layout)
    wheel_copy = importlib.resources.files(get_package_name()).parent / dir
    # directory *beside* src/ (editable / repo checkout)
    repo_copy  = Path(__file__).resolve().parents[2] / dir

    for path in (repo_copy, wheel_copy):
        if path.is_dir():
            return path.resolve()

    raise FileNotFoundError("thirdparty directory not found")

def get_local_cuda_src_dir() -> Path:
    return get_top_level_repo_dir('csrc')

def get_include_local_cuda_dir() -> Path:
    return get_top_level_repo_dir('include')

def get_include_dir_cutlass() -> Path:
    return get_top_level_repo_dir('thirdparty') / 'cutlass/include'

def get_include_dir_cuda() -> Path:
    """Best-effort guess of the Toolkit’s <cuda>/include directory."""
    import os, shutil
    if os.getenv("CUDA_HOME"):
        return Path(os.environ["CUDA_HOME"]) / "include"
    # fall back to the directory that owns nvcc (works for most local installs)
    nvcc = shutil.which("nvcc")
    if nvcc:
        return Path(nvcc).parent.parent / "include"
    raise RuntimeError("Cannot find CUDA include directory")

def cache_root() -> pathlib.Path:
    """
    <user_cache_dir>/<your-package>/<env-id>/
    Guaranteed unique per virtual-env *and* persistent across runs.
    """
    try:
        # 1. A stable ID for THIS Python installation / venv
        env_id = hashlib.sha256(sys.prefix.encode()).hexdigest()[:12]

        # 2. RFC 6685 cache location that respects XDG / Windows / macOS rules
        from platformdirs import user_cache_dir
        base_dir = pathlib.Path(user_cache_dir(f'{get_package_name()}')) / f'{get_package_version()}'
    

        # 3. Create sub-dir and a sentinel file on first use
        target = base_dir / env_id
        target.mkdir(parents=True, exist_ok=True)

        sentinel = target / "instance.json"
        if not sentinel.exists():
            sentinel.write_text(json.dumps({"uuid": str(uuid.uuid4())}))
        return target
    except Exception:               # Fallback: tmpdir
        import tempfile
        return pathlib.Path(tempfile.mkdtemp(prefix=f'{get_package_name()}-cache-'))