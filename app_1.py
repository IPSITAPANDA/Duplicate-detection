# app.py
# UI POC: Upload two images -> ORB + RANSAC homography -> duplicate decision + explanation
#
# Install:
#   pip install streamlit opencv-python numpy pillow
#
# Run:
#   streamlit run app.py

import io
import numpy as np
import cv2
import streamlit as st
from PIL import Image


def read_image(uploaded_file) -> np.ndarray:
    """Read an uploaded image into a BGR OpenCV array."""
    data = uploaded_file.read()
    pil = Image.open(io.BytesIO(data)).convert("RGB")
    img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return img


def orb_ransac_analysis(
    img1_bgr: np.ndarray,
    img2_bgr: np.ndarray,
    max_features: int = 5000,
    ratio_test: float = 0.75,
    ransac_reproj_threshold: float = 3.0,
):
    """Return match stats + optional visualization."""
    gray1 = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=max_features)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    result = {
        "ok": False,
        "reason": "",
        "keypoints_1": len(kp1),
        "keypoints_2": len(kp2),
        "good_matches": 0,
        "inliers": 0,
        "inlier_ratio": 0.0,
        "homography": None,
        "match_viz_bgr": None,
    }

    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        result["reason"] = "Not enough detectable features in one or both images (too blurry/low-texture)."
        return result

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(des1, des2, k=2)

    good = []
    for pair in knn:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < ratio_test * n.distance:
            good.append(m)

    result["good_matches"] = len(good)
    if len(good) < 8:
        result["reason"] = "Not enough good matches after filtering to fit a homography."
        return result

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_reproj_threshold)
    if H is None or mask is None:
        result["reason"] = "RANSAC could not find a consistent geometric mapping between images."
        return result

    inliers = int(mask.sum())
    inlier_ratio = float(inliers) / float(len(good))

    # Build a visualization: draw only inlier matches
    inlier_matches = [m for m, keep in zip(good, mask.ravel().tolist()) if keep]
    match_viz = cv2.drawMatches(
        img1_bgr, kp1, img2_bgr, kp2, inlier_matches[:200], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    result.update(
        ok=True,
        reason="",
        inliers=inliers,
        inlier_ratio=inlier_ratio,
        homography=H,
        match_viz_bgr=match_viz,
    )
    return result


def decide_duplicate(stats: dict, min_inliers: int, min_inlier_ratio: float) -> bool:
    if not stats.get("ok"):
        return False
    return stats["inliers"] >= min_inliers and stats["inlier_ratio"] >= min_inlier_ratio


st.set_page_config(page_title="ORB + RANSAC Duplicate Detector", layout="wide")
st.title("ORB + RANSAC Duplicate Detector (POC)")

st.write(
    "Upload two images. The app detects keypoints (ORB), matches them, then uses RANSAC to find a single "
    "perspective mapping (homography). If many matches agree (high inliers + inlier ratio), the images are treated as duplicates."
)

colL, colR = st.columns(2)
with colL:
    f1 = st.file_uploader("Image 1", type=["png", "jpg", "jpeg", "webp"], key="img1")
with colR:
    f2 = st.file_uploader("Image 2", type=["png", "jpg", "jpeg", "webp"], key="img2")

st.divider()

with st.expander("Advanced settings", expanded=False):
    max_features = st.slider("ORB max features", 500, 10000, 5000, 500)
    ratio_test = st.slider("Lowe ratio test (lower = stricter)", 0.50, 0.95, 0.75, 0.01)
    ransac_thresh = st.slider("RANSAC reprojection threshold (px)", 1.0, 10.0, 3.0, 0.5)

    st.write("Duplicate decision thresholds (tune for your data):")
    min_inliers = st.slider("Minimum inliers", 20, 1000, 150, 10)
    min_inlier_ratio = st.slider("Minimum inlier ratio", 0.05, 0.90, 0.25, 0.01)

if f1 and f2:
    img1 = read_image(f1)
    img2 = read_image(f2)

    c1, c2 = st.columns(2)
    with c1:
        st.image(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB), caption="Image 1", use_column_width=True)
    with c2:
        st.image(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB), caption="Image 2", use_column_width=True)

    st.subheader("Analysis")
    stats = orb_ransac_analysis(
        img1, img2,
        max_features=max_features,
        ratio_test=ratio_test,
        ransac_reproj_threshold=ransac_thresh,
    )

    if not stats["ok"]:
        st.error("Could not make a reliable geometric comparison.")
        st.write("Reason:", stats["reason"])
        st.write(
            {
                "keypoints_image_1": stats["keypoints_1"],
                "keypoints_image_2": stats["keypoints_2"],
                "good_matches": stats["good_matches"],
            }
        )
    else:
        dup = decide_duplicate(stats, min_inliers=min_inliers, min_inlier_ratio=min_inlier_ratio)

        # Why/explanation
        explanation = []
        explanation.append(f"ORB found {stats['keypoints_1']} keypoints in Image 1 and {stats['keypoints_2']} in Image 2.")
        explanation.append(f"After matching + filtering, there were {stats['good_matches']} good matches.")
        explanation.append(f"RANSAC found a single perspective mapping supported by {stats['inliers']} inliers.")
        explanation.append(f"Inlier ratio = {stats['inlier_ratio']:.3f} (inliers / good matches).")
        explanation.append(
            "Interpretation: high inliers + high inlier ratio means many local details align under one consistent perspective transform—"
            "strong evidence of the same planar object/document."
        )

        if dup:
            st.success("Decision: DUPLICATES (same document/object under viewpoint changes)")
        else:
            st.warning("Decision: NOT DUPLICATES (insufficient geometric agreement under current thresholds)")

        st.write(
            {
                "keypoints_image_1": stats["keypoints_1"],
                "keypoints_image_2": stats["keypoints_2"],
                "good_matches": stats["good_matches"],
                "inliers": stats["inliers"],
                "inlier_ratio": round(stats["inlier_ratio"], 3),
                "thresholds": {"min_inliers": min_inliers, "min_inlier_ratio": min_inlier_ratio},
            }
        )

        st.markdown("**Why:**")
        for line in explanation:
            st.write("- " + line)

        st.subheader("Inlier matches (visual proof)")
        viz_rgb = cv2.cvtColor(stats["match_viz_bgr"], cv2.COLOR_BGR2RGB)
        st.image(viz_rgb, use_column_width=True, caption="Up to 200 inlier matches drawn (lines connect matched features).")
else:
    st.info("Upload two images to run the analysis.")
