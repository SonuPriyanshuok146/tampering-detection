from tampering_detection.preprocessing import compute_ela, ela_to_array


def test_compute_ela_runs(tmp_path):
    from PIL import Image
    img_path = tmp_path / "sample.jpg"
    Image.new("RGB", (64, 64), color=(120, 50, 200)).save(img_path)

    ela_img = compute_ela(str(img_path))
    assert ela_img.size == (64, 64)


def test_ela_to_array_shape(tmp_path):
    from PIL import Image
    img_path = tmp_path / "sample.jpg"
    Image.new("RGB", (64, 64), color=(10, 10, 10)).save(img_path)

    ela_img = compute_ela(str(img_path))
    arr = ela_to_array(ela_img, size=(128, 128))
    assert arr.shape == (128, 128, 3)
    assert arr.min() >= 0.0 and arr.max() <= 1.0