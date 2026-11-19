from pathlib import Path


CASIA_ROOT = Path("data/CASIA2")


def show_files(folder, limit=30):
    print(f"\n{'=' * 60}")
    print(f"FOLDER: {folder}")
    print(f"{'=' * 60}")

    files = []

    for path in folder.rglob("*"):
        if path.is_file():
            files.append(path)

    print(f"Total files: {len(files)}")

    for path in files[:limit]:
        print(path)


def main():
    au_dir = CASIA_ROOT / "Au"
    tp_dir = CASIA_ROOT / "Tp"
    gt_dir = CASIA_ROOT / "CASIA 2 Groundtruth"

    show_files(au_dir)
    show_files(tp_dir)
    show_files(gt_dir)


if __name__ == "__main__":
    main()