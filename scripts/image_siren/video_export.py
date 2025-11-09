import os
from typing import List, Optional


def _collect_recon_frames(run_dir: str) -> List[str]:
    prefix = "image_siren_epoch_"
    suffix = ".png"

    if not os.path.isdir(run_dir):
        print(f"image_siren: run_dir does not exist: {run_dir}")
        return []

    names = os.listdir(run_dir)
    if not names:
        print(f"image_siren: no files found in run_dir: {run_dir}")
        return []

    frames = []
    for name in names:
        if not name.startswith(prefix):
            continue
        if not name.lower().endswith(suffix):
            continue
        core = name[len(prefix):-len(suffix)]
        try:
            epoch = int(core)
        except ValueError:
            continue
        frames.append((epoch, name))

    frames.sort(key=lambda x: x[0])
    ordered = [name for _, name in frames]

    print(
        f"image_siren: found {len(ordered)} recon frames "
        f"matching {prefix}*{suffix} in {run_dir}"
    )

    if not ordered:
        sample = sorted(names)[:10]
        print("image_siren: no matching frames; sample of files:")
        for s in sample:
            print(f"  {s}")

    return ordered


def make_recon_video(
    run_dir: str,
    fps: int = 4,
    crf: int = 23,
    basename: str = "siren_recon",
) -> Optional[str]:
    frames = _collect_recon_frames(run_dir)
    if not frames:
        return None

    try:
        try:
            import imageio.v2 as imageio
        except ImportError:
            import imageio  # type: ignore
    except ImportError:
        print("image_siren: imageio is not installed; cannot create video.")
        return None

    out_path = os.path.join(run_dir, f"{basename}.mp4")

    images = []
    for name in frames:
        path = os.path.join(run_dir, name)
        try:
            img = imageio.imread(path)
        except Exception as e:
            print(f"image_siren: failed to read frame {path}: {e}")
            continue
        images.append(img)

    if not images:
        print("image_siren: no readable frames; aborting video export.")
        return None

    try:
        imageio.mimsave(out_path, images, fps=int(fps))
    except Exception as e:
        print(f"image_siren: failed to write video {out_path}: {e}")
        return None

    if not os.path.exists(out_path):
        print("image_siren: video file was not created.")
        return None

    print(f"image_siren: reconstruction video written to {out_path}")
    return out_path
