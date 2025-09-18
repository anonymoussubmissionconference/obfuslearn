from Trainer import *
from torchvision import transforms
from img_gist_feature.utils_gist import *
import cv2
from CustomDataModule import *
from pathlib import Path
# --- Unicode-safe imread/imwrite for OpenCV ---
def cv2_imread_unicode(path, flags=cv2.IMREAD_GRAYSCALE):
    path = Path(path)
    data = np.fromfile(str(path), dtype=np.uint8)   # works with Unicode paths
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)

def iter_gray_images(root_dir, exts=(".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")):
    root = Path(root_dir)
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            img = cv2_imread_unicode(p, cv2.IMREAD_GRAYSCALE)  # 2D uint8 or None
            if img is None:
                print(f"[WARN] Failed to read {p}")
                continue
            rel = p.relative_to(root)  # keep subfolder info for later saving
            yield rel, img

def getGistFeature(image):
    gist_helper = GistUtils()
    # print(image.shape, label.shape)
    np_gist = gist_helper.get_gist_vec(image, mode="gray").flatten().reshape(16, 20)

    return np_gist

root_path = r'../data/malimg'

root_dst=r'../features/gist'
for path, arr in iter_gray_images(root_path):
    np_gist = getGistFeature(arr)
    np_gist = (np_gist - np_gist.min()) / (np_gist.max() - np_gist.min() + 1e-8)
    np_gist = (np_gist * 255).astype(np.uint8)
    print(np_gist.shape)
    dst_path = root_dst / path  # mirrors the subfolder structure
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    print(dst_path)
    try:
        img = Image.fromarray(np_gist)

        img.save(dst_path)
    except Exception as e:
        print(f"[WARN] Skipped {dst_path}: {e}")
