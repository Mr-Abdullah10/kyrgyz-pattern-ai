import importlib
import sys


REQUIRED_MODULES = [
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
    ("multipart", "python-multipart"),
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("timm", "timm"),
    ("open_clip", "open-clip-torch"),
    ("faiss", "faiss-cpu"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("PIL", "Pillow"),
]


def main() -> int:
    missing = []
    print("Checking API environment modules...")
    for module_name, package_name in REQUIRED_MODULES:
        try:
            importlib.import_module(module_name)
            print(f"[OK] {module_name}")
        except Exception as exc:
            missing.append((module_name, package_name, str(exc)))
            print(f"[MISSING] {module_name} (install package: {package_name})")

    print()
    if missing:
        print("Environment check failed. Missing modules:")
        for module_name, package_name, err in missing:
            print(f"- {module_name} (pip package: {package_name}) -> {err}")
        print("\nInstall with:")
        print("  py -3 -m pip install -r requirements-api.txt")
        return 1

    print("Environment check passed. API dependencies are ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
