import cv2
import numpy as np
from sklearn.cluster import KMeans
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern


# OpenCV 8-bit LAB: L=0-255, A=0-255 (centered 128), B=0-255 (centered 128)
REFERENCE_COLORS_LAB = {
    "black":  np.array([0, 128, 128]),
    "white":  np.array([255, 128, 128]),
    "gray":   np.array([85, 128, 128]),
    "brown":  np.array([84, 134, 140]),
    "orange": np.array([153, 155, 185]),
    "cream":  np.array([217, 130, 145]),
    "beige":  np.array([130, 132, 140]),
    "ginger": np.array([128, 160, 175]),
}

ACHROMATIC_COLORS = {"black", "white", "gray"}
WARM_SOLID_COLORS = {"cream", "beige", "brown", "ginger", "orange"}
ACHROMATIC_THRESHOLD = 5  # max distance from neutral (128) in a/b channels

# Gray clusters with L below this are lighting highlights on black fur
DARK_GRAY_L_THRESHOLD = 80
# Gray clusters with L above this are shadows on white fur
LIGHT_GRAY_L_THRESHOLD = 200
# Secondary achromatic colors below this proportion are lighting artifacts
MINOR_ACHROMATIC_PROPORTION = 0.20
# Pixels with a or b this far from 128 are clearly chromatic — skip achromatic refs
CLEAR_CHROMATIC_THRESHOLD = 12


def map_to_named_color(lab_value: np.ndarray) -> str:
    a_dist = abs(float(lab_value[1]) - 128)
    b_dist = abs(float(lab_value[2]) - 128)

    is_achromatic = (
        a_dist < ACHROMATIC_THRESHOLD and b_dist < ACHROMATIC_THRESHOLD
    )
    is_clearly_chromatic = (
        a_dist > CLEAR_CHROMATIC_THRESHOLD or b_dist > CLEAR_CHROMATIC_THRESHOLD
    )

    min_dist = float("inf")
    best = "unknown"
    for name, ref in REFERENCE_COLORS_LAB.items():
        if is_achromatic and name not in ACHROMATIC_COLORS:
            continue
        if is_clearly_chromatic and name in ACHROMATIC_COLORS:
            continue
        dist = np.linalg.norm(lab_value - ref)
        if dist < min_dist:
            min_dist = dist
            best = name
    return best


def extract_colors(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    n_clusters: int = 4,
    min_proportion: float = 0.10,
) -> list[dict]:
    pet_pixels = image_bgr[mask > 0]
    if len(pet_pixels) == 0:
        return []

    pet_pixels_lab = cv2.cvtColor(
        pet_pixels.reshape(1, -1, 3), cv2.COLOR_BGR2LAB
    ).reshape(-1, 3)

    n_clusters = min(n_clusters, len(pet_pixels_lab))
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(pet_pixels_lab)

    counts = np.bincount(labels)
    proportions = counts / counts.sum()

    raw = []
    for i in np.argsort(-proportions):
        if proportions[i] < min_proportion:
            continue
        center_lab = kmeans.cluster_centers_[i]
        raw.append({
            "color": map_to_named_color(center_lab),
            "proportion": float(proportions[i]),
            "lab_value": center_lab,
        })

    # First pass: merge clusters that mapped to the same named color
    merged = _merge_by_color(raw)

    # Second pass: re-map from merged averages (individual clusters may have
    # different achromatic status than their weighted average)
    remapped = [
        {**entry, "color": map_to_named_color(np.array(entry["lab_value"]))}
        for entry in merged
    ]

    final = _merge_by_color(remapped)

    # Third pass: merge gray into black/white when it's a lighting artifact
    return _merge_achromatic_lighting(final)


def _merge_by_color(clusters: list[dict]) -> list[dict]:
    grouped: dict[str, dict] = {}
    for c in clusters:
        name = c["color"]
        if name not in grouped:
            grouped[name] = {"proportion": 0.0, "lab_values": [], "weights": []}
        grouped[name]["proportion"] += c["proportion"]
        grouped[name]["lab_values"].append(
            np.asarray(c["lab_value"])
        )
        grouped[name]["weights"].append(c["proportion"])

    results = []
    for name in sorted(grouped, key=lambda n: -grouped[n]["proportion"]):
        m = grouped[name]
        avg_lab = np.average(m["lab_values"], axis=0, weights=m["weights"])
        results.append({
            "color": name,
            "proportion": round(m["proportion"], 4),
            "lab_value": [round(float(v), 2) for v in avg_lab],
        })
    return results


def _merge_achromatic_lighting(colors: list[dict]) -> list[dict]:
    """Merge achromatic lighting artifacts into the dominant achromatic color."""
    color_map = {c["color"]: c for c in colors}
    achromatic_present = {name for name in color_map if name in ACHROMATIC_COLORS}

    if len(achromatic_present) < 2:
        return colors

    # Case 1: dark gray is a lighting highlight on black fur
    if "gray" in color_map and "black" in color_map:
        gray_L = color_map["gray"]["lab_value"][0]
        gray_is_dominant = all(
            color_map["gray"]["proportion"] >= c["proportion"] for c in colors
        )
        if gray_L < DARK_GRAY_L_THRESHOLD and not gray_is_dominant:
            renamed = [
                {**c, "color": "black"} if c["color"] == "gray" else c
                for c in colors
            ]
            return _merge_by_color(renamed)

    # Case 2: light gray is a shadow on white fur
    if "gray" in color_map and "white" in color_map:
        gray_L = color_map["gray"]["lab_value"][0]
        gray_is_dominant = all(
            color_map["gray"]["proportion"] >= c["proportion"] for c in colors
        )
        if gray_L > LIGHT_GRAY_L_THRESHOLD and not gray_is_dominant:
            renamed = [
                {**c, "color": "white"} if c["color"] == "gray" else c
                for c in colors
            ]
            return _merge_by_color(renamed)

    # Case 3: all achromatic with one clear dominant — minor ones are lighting
    all_achromatic = all(c["color"] in ACHROMATIC_COLORS for c in colors)
    if all_achromatic:
        dominant = max(colors, key=lambda c: c["proportion"])
        if dominant["proportion"] > 0.50:
            renamed = [
                {**c, "color": dominant["color"]}
                if c["color"] != dominant["color"]
                and c["proportion"] < MINOR_ACHROMATIC_PROPORTION
                else c
                for c in colors
            ]
            if any(r["color"] != o["color"] for r, o in zip(renamed, colors)):
                return _merge_by_color(renamed)

    return colors


def extract_glcm_features(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    distances: list[int] | None = None,
    angles: list[float] | None = None,
) -> dict:
    if distances is None:
        distances = [1, 3, 5]
    if angles is None:
        angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    gray_masked = gray.copy()
    gray_masked[mask == 0] = 0
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        return {}
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    roi = gray_masked[y_min : y_max + 1, x_min : x_max + 1]

    levels = 64
    roi_quantized = (roi / 256 * levels).astype(np.uint8)

    glcm = graycomatrix(
        roi_quantized,
        distances=distances,
        angles=angles,
        levels=levels,
        symmetric=True,
        normed=True,
    )

    properties = ["contrast", "homogeneity", "energy", "correlation", "dissimilarity"]
    features = {}
    for prop in properties:
        values = graycoprops(glcm, prop)
        features[f"glcm_{prop}_mean"] = float(values.mean())
        features[f"glcm_{prop}_std"] = float(values.std())

    return features


def extract_lbp_features(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    radius: int = 2,
    n_points: int = 16,
) -> dict:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")

    lbp_masked = lbp[mask > 0]
    if len(lbp_masked) == 0:
        return {}

    n_bins = n_points + 2
    hist, _ = np.histogram(lbp_masked, bins=n_bins, range=(0, n_bins), density=True)

    features = {f"lbp_bin_{i}": float(v) for i, v in enumerate(hist)}
    features["lbp_entropy"] = float(
        -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
    )

    return features


def classify_pattern(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    species: str = "cat",
    colors: list[dict] | None = None,
    glcm: dict | None = None,
    lbp: dict | None = None,
) -> str:
    species = species.lower()
    if colors is None:
        colors = extract_colors(image_bgr, mask)
    if glcm is None:
        glcm = extract_glcm_features(image_bgr, mask)
    if lbp is None:
        lbp = extract_lbp_features(image_bgr, mask)

    if not colors or not glcm or not lbp:
        return "unknown"

    n_dominant_colors = len(colors)

    def _chroma(c: dict) -> float:
        return np.sqrt(
            (c["lab_value"][1] - 128) ** 2 + (c["lab_value"][2] - 128) ** 2
        )

    func_white_prop = sum(
        c["proportion"] for c in colors
        if c["color"] == "white"
        or (c["lab_value"][0] > 200 and _chroma(c) < 12)
    )
    func_dark_prop = sum(
        c["proportion"] for c in colors
        if c["color"] == "black" or c["lab_value"][0] < 60
    )

    # ═══ 1. Solid: one dominant color, low texture ═══
    if n_dominant_colors == 1 and glcm["glcm_contrast_mean"] < 30:
        return "solid"

    # ═══ 2. Solid override: overwhelmingly dominant color ═══
    dominant = max(colors, key=lambda c: c["proportion"])
    secondary = sorted(
        [c for c in colors if c is not dominant],
        key=lambda c: -c["proportion"],
    )
    hidden_pattern = (
        species == "dog"
        and lbp.get("lbp_entropy", 0) >= 3.5
        and (glcm.get("glcm_contrast_mean", 0) < 30 or n_dominant_colors == 1)
    )
    if dominant["proportion"] >= 0.70 and (
        not secondary or secondary[0]["proportion"] < 0.13
    ) and not hidden_pattern:
        if not (func_white_prop >= 0.15 and func_dark_prop >= 0.15):
            return "solid"

    # ═══ 3. Solid override 2: near-achromatic, close in lightness ═══
    # Skip if significant texture is present (pattern within achromatic coat)
    if n_dominant_colors >= 2:
        chromas = [_chroma(c) for c in colors]
        L_values = [c["lab_value"][0] for c in colors]
        L_spread = max(L_values) - min(L_values)
        has_texture = (
            glcm["glcm_contrast_mean"] > 30
            and lbp.get("lbp_entropy", 0) > 3.0
        )
        if all(ch < 12 for ch in chromas) and L_spread < 80 and not has_texture:
            return "solid"

    # ═══ 3b. Dark solid: all colors are dark (L < 80) ═══
    if all(c["lab_value"][0] < 80 for c in colors):
        return "solid"

    # ═══ 3c. All-achromatic solid: all achromatic without significant white ═══
    # Threshold lowered from 0.15 to 0.10 so that gray+black+white dogs
    # with white ~13% can escape to the achromatic bicolor rule instead.
    if (all(c["color"] in ACHROMATIC_COLORS for c in colors)
            and func_white_prop < 0.10):
        return "solid"

    # ═══ 4. Tortoiseshell (cats): dark + warm, little/no white ═══
    # Must fire before tabby/bicolor rules — tortie texture looks tabby-like
    # (contrast > 12, entropy > 3.0) and tortie func_dark + warm can trigger
    # rule 9's bicolor override. Tortie signature is biologically specific:
    # dark + warm + no white. Feature means from validation:
    #   tortie → tabby leaks: func_white 0.000, func_dark 0.235, warm 0.738
    #   correct tabbies:      func_white 0.128, func_dark 0.076, warm 0.673
    # func_dark alone separates them; gate at 0.15 is safe.
    if species == "cat":
        tortie_dark_prop = sum(
            c["proportion"] for c in colors if c["lab_value"][0] < 60
        )
        # Chroma >= 15 filter rejects low-chroma beige/brown interpolation
        # clusters (bicolor → tortie leak: high_chroma_warm 0.078 vs true
        # tortie 0.250). Warm_prop gate drops from 0.25 to 0.20 since the
        # stricter chroma filter removes the inflation from K-means noise.
        tortie_warm = [
            c for c in colors
            if c["color"] not in ACHROMATIC_COLORS
            and c["lab_value"][2] > 130
            and _chroma(c) >= 15
        ]
        tortie_warm_prop = sum(c["proportion"] for c in tortie_warm)
        # Contrast < 80 rejects high-contrast brown tabbies (tabby → tortie
        # leak: contrast 89.7 vs true tortie 51.7). Tabby stripes drive
        # contrast up; tortie patches stay moderate.
        if (tortie_dark_prop >= 0.15 and tortie_warm_prop >= 0.10
                and func_white_prop < 0.15
                and glcm["glcm_contrast_mean"] < 80):
            return "tortoiseshell"

    # ═══ 5. Warm-solid override: all warm colors, low entropy, low contrast ═══
    if (all(c["color"] in WARM_SOLID_COLORS for c in colors)
            and lbp.get("lbp_entropy", 0) < 3.55
            and glcm["glcm_contrast_mean"] < 35):
        return "solid"

    # ═══ 5b. Warm solid, few clusters: shade variation, not a pattern ═══
    # A 2-cluster warm coat with low entropy is just lighting/shade variation
    # (e.g., Abyssinian cat: ginger + orange). High entropy excluded to
    # preserve tabbies like brown 69% + beige 22% with entropy ~3.9.
    if (all(c["color"] in WARM_SOLID_COLORS for c in colors)
            and n_dominant_colors <= 2
            and dominant["proportion"] >= 0.55
            and lbp.get("lbp_entropy", 0) < 3.5):
        return "solid"

    # ═══ 5c. Light-dominant warm dog: K-means splits a warm coat into shades ═══
    # Yellow/golden labs produce cream + beige + brown clusters from a single
    # warm coat. The light dominant color (L > 150) is the base coat color.
    if (species == "dog"
            and all(c["color"] in WARM_SOLID_COLORS for c in colors)
            and dominant["proportion"] >= 0.50
            and dominant["lab_value"][0] > 150):
        return "solid"

    # ═══ 6. Early dog bicolor: top-2 colors dominate, different families ═══
    # Catches bicolor dogs before tricolor absorb them. Requires the
    # top 2 colors to be from different families (one achromatic + one not).
    # L_spread alternative removed — it caught gray+white merle/spotted dogs.
    # Tricolor escape: skip bicolor when remaining colors show a warm accent
    # alongside dark+light, indicating a tricolor rather than bicolor.
    if species == "dog" and len(colors) >= 2:
        top2 = sorted(colors, key=lambda c: -c["proportion"])[:2]
        top2_total = top2[0]["proportion"] + top2[1]["proportion"]
        both_warm = all(c["color"] in WARM_SOLID_COLORS for c in top2)
        top2_different_family = (
            (top2[0]["color"] in ACHROMATIC_COLORS)
            != (top2[1]["color"] in ACHROMATIC_COLORS)
        )
        if top2_total >= 0.60 and not both_warm and top2_different_family:
            remaining = [
                c for c in colors
                if c is not top2[0] and c is not top2[1]
            ]
            has_warm_accent_remaining = any(
                c["color"] in WARM_SOLID_COLORS
                and c["proportion"] >= 0.11
                and (c["lab_value"][0] < 150 or c["lab_value"][2] > 140)
                for c in remaining
            )
            all_light_prop = sum(
                c["proportion"] for c in colors if c["lab_value"][0] > 150
            )
            # Also detect tricolor when the warm color is in the top-2
            # and the remaining has a very light (L>200) "white" patch.
            # E.g., beagle: beige(warm) + black + cream(L=219).
            one_top2_warm = any(
                c["color"] in WARM_SOLID_COLORS for c in top2
            )
            has_light_remaining = any(
                c["lab_value"][0] > 200 and c["proportion"] >= 0.15
                for c in remaining
            )
            is_likely_tricolor = (
                (has_warm_accent_remaining
                 and func_dark_prop >= 0.10
                 and all_light_prop >= 0.15)
                or (one_top2_warm
                    and has_light_remaining
                    and func_dark_prop >= 0.10)
            )
            if not is_likely_tricolor:
                return "bicolor"

    # ═══ 6a. Calico (cats): dark + DISTINCT white patches + warm ═══
    # Must fire before rule 6b (cat bicolor extreme pair) and rule 9 (bicolor
    # override) — both were absorbing calicos: pair_prop >= 0.60 cats with
    # warm intermediates hit rule 6b, and func_white + func_dark >= 0.55 cats
    # failed rule 9's tabby escape. Biology: calico needs distinct white
    # patches (not belly highlights). func_white >= 0.15 forms the partition
    # with tortoiseshell; warm_patch_prop >= 0.15 distinguishes from plain
    # black+white bicolors.
    if species == "cat" and n_dominant_colors >= 3:
        calico_achromatic_set = {
            id(c) for c in colors
            if c["color"] in ACHROMATIC_COLORS
            or (c["lab_value"][0] > 200 and _chroma(c) < 12)
            or c["lab_value"][0] < 60
        }
        broader_white_prop = sum(
            c["proportion"] for c in colors
            if c["color"] == "white" or c["lab_value"][0] > 200
        )
        # Chroma >= 15 filter rejects K-means interpolation clusters
        # (bicolor → calico leak: high_chroma_warm 0.000 vs true calico 0.250).
        # Beige at b=140, a=132 has chroma ~12.6 — just barely chromatic,
        # often a lightness-gradient artifact between black and white patches.
        warm_patches = [
            c for c in colors
            if c["lab_value"][0] <= 200
            and id(c) not in calico_achromatic_set
            and (c["color"] in {"orange", "ginger"}
                 or (c["color"] not in ACHROMATIC_COLORS
                     and c["lab_value"][2] > 137))
            and _chroma(c) >= 15
        ]
        warm_patch_prop = sum(c["proportion"] for c in warm_patches)
        # Homogeneity couldn't separate calico from tabby — both sit at
        # 0.540 in validation. The calico/tabby split is purely color-based:
        # true calicos have combined extreme color (white + dark patches)
        # at 0.610; tabby→calico leaks sit at 0.430, tabby→tabby at 0.203.
        # 0.45 gate recovers calicos lost to tabby and rejects tabby leaks.
        if (func_white_prop >= 0.15
                and broader_white_prop >= 0.20
                and func_dark_prop >= 0.10
                and warm_patch_prop >= 0.10
                and func_white_prop + func_dark_prop >= 0.45):
            return "calico"

    # ═══ 6b. Cat bicolor: dark+light extreme pair ═══
    # Catches cat bicolors where K-means finds one very dark (L<60) and one
    # very light (L>200) color with combined proportion >= 60%, and all other
    # clusters are low-chroma intermediates (chroma < 15). In true calico,
    # the remaining warm patches have high chroma (orange/ginger >= 15-40).
    # Must fire BEFORE calico to prevent false calico on bicolors.
    if species == "cat" and n_dominant_colors >= 2:
        cat_very_light = [c for c in colors if c["lab_value"][0] > 200]
        cat_very_dark = [c for c in colors if c["lab_value"][0] < 60]
        if cat_very_light and cat_very_dark:
            best_light = max(cat_very_light, key=lambda c: c["proportion"])
            best_dark = max(cat_very_dark, key=lambda c: c["proportion"])
            pair_prop = best_light["proportion"] + best_dark["proportion"]
            pair_remaining = [
                c for c in colors
                if c is not best_light and c is not best_dark
            ]
            pair_remaining_low_chroma = (
                all(_chroma(c) < 15 for c in pair_remaining)
                if pair_remaining else True
            )
            if pair_prop >= 0.60 and pair_remaining_low_chroma:
                return "bicolor"

    # ═══ 7. Calico (cats) / Tricolor (dogs) ═══
    func_achromatic = [
        c for c in colors
        if c["color"] in ACHROMATIC_COLORS
        or (c["lab_value"][0] > 200 and _chroma(c) < 12)
        or c["lab_value"][0] < 60
    ]
    func_achromatic_set = set(id(c) for c in func_achromatic)

    # Dog tricolor: dark + light + warm accent (medium-tone warm color)
    # Uses WARM_SOLID_COLORS with L < 175 to catch tan/brown accents
    # including lighter beige patches (L ~165-170) seen in Bernese and
    # border collie tricolors. Chroma >= 11 filters K-means interpolation
    # artifacts — when black+white dogs produce low-chroma brown/beige
    # clusters that are just lightness gradients, not true warm patches.
    if species == "dog" and n_dominant_colors >= 3:
        warm_accent = [
            c for c in colors
            if c["color"] in WARM_SOLID_COLORS
            and id(c) not in func_achromatic_set
            and c["lab_value"][0] < 175
            and _chroma(c) >= 11
        ]
        warm_accent_prop = sum(c["proportion"] for c in warm_accent)
        light_prop = sum(
            c["proportion"] for c in colors if c["lab_value"][0] > 150
        )
        if (func_dark_prop >= 0.10 and light_prop >= 0.15
                and warm_accent_prop >= 0.10):
            return "tricolor"

    # ═══ 9. Bicolor override: distinct light + dark regions ═══
    # Cat tabby escape: if extreme colors (func_white + func_dark) account
    # for less than 55% of the coat and the cat shows tabby texture, the
    # intermediate tones (gray/beige) indicate a patterned coat, not a
    # clean bicolor. True bicolors have extreme colors dominating (60%+).
    if func_white_prop >= 0.15 and func_dark_prop >= 0.20:
        if (species == "cat"
                and func_white_prop + func_dark_prop < 0.55
                and glcm["glcm_contrast_mean"] > 12
                and lbp["lbp_entropy"] > 3.0):
            pass  # skip bicolor, fall through to tabby
        else:
            return "bicolor"

    # ═══ 9b. All-achromatic dog bicolor ═══
    # Catches gray+black+white dogs where white is 10-15% (below rule 9's
    # func_white >= 0.15 threshold). These dogs escape rule 3c with the
    # lowered threshold and need a dedicated bicolor path.
    if (species == "dog"
            and all(c["color"] in ACHROMATIC_COLORS for c in colors)
            and func_white_prop >= 0.10
            and func_dark_prop >= 0.20):
        return "bicolor"

    # ═══ 10. Cat bicolor: dark + light with low-chroma intermediates ═══
    # Catches cat bicolors where the coat has a clear dark region (L < 60)
    # and a light region (high L) but K-means produces intermediate
    # clusters (beige, cream) that are low-chroma — just lightness
    # interpolation between the extremes, not distinct colored patches.
    # Distinguished from tortie/calico by requiring ALL intermediates
    # to have chroma < 12 (true warm patches have higher chroma).
    if species == "cat" and n_dominant_colors >= 2:
        darkest = min(colors, key=lambda c: c["lab_value"][0])
        lightest = max(colors, key=lambda c: c["lab_value"][0])
        cat_L_spread = lightest["lab_value"][0] - darkest["lab_value"][0]
        intermediates = [
            c for c in colors if c is not darkest and c is not lightest
        ]
        intermediates_low_chroma = (
            all(_chroma(c) < 12 for c in intermediates)
            if intermediates else True
        )
        if cat_L_spread >= 130 and func_dark_prop >= 0.35 and intermediates_low_chroma:
            return "bicolor"

    # ═══ 11. Tabby (cats): merged tabby + spotted ═══
    # Contrast threshold lowered from 15 to 12 to catch low-contrast tabbies
    # (e.g., beige + cream Bengal with contrast ~12.6).
    if species == "cat":
        if (glcm["glcm_contrast_mean"] > 12 and lbp["lbp_entropy"] > 3.0):
            return "tabby"
        # Warm tabby: all warm colors + high contrast (e.g., ocicat)
        if (all(c["color"] in WARM_SOLID_COLORS for c in colors)
                and glcm["glcm_contrast_mean"] > 40):
            return "tabby"

    # ═══ 13. Bicolor fallback (cats only, relaxed L_spread) ═══
    # Scoped to cats: dogs that reach this point failed the strict bicolor
    # rules above, so they're probably mottled/ticked/merle/brindle and
    # should land in "irregular" rather than be force-fit to bicolor.
    if species == "cat":
        primary_colors = [c for c in colors if c["proportion"] >= 0.20]
        if len(primary_colors) >= 2:
            p_L = [c["lab_value"][0] for c in primary_colors]
            if max(p_L) - min(p_L) >= 50:
                return "bicolor"

    # ═══ 13b. Bicolor fallback (cats only, all-colors L_spread) ═══
    if species == "cat" and len(colors) >= 2:
        all_L = [c["lab_value"][0] for c in colors]
        if max(all_L) - min(all_L) >= 100:
            return "bicolor"

    # ═══ 13c. Dog bicolor fallback (texture-gated L_spread) ═══
    # Catches warm+warm and intermediate-shade bicolors that strict rules
    # miss: brown+cream, brown+beige, chocolate+tan, red+white-as-cream,
    # plus fluffy/shaggy coats whose fur texture lowers GLCM energy.
    #
    # Validated against 400 hand-labeled bicolor dogs across two passes:
    # the missed ones sit at entropy ~3.55, energy ~0.29. True irregular
    # coats (ticked/roan/merle) sit at entropy ≥ 3.85 AND energy < 0.22
    # *simultaneously* — the gate rejects only that joint extreme, so
    # fluffy bicolors with moderate energy OR moderate entropy still pass.
    #
    # Requires ≥ 2 colors at ≥ 15% each so K-means noise clusters can't
    # qualify a solid coat. L_spread uses all clusters (each already
    # ≥ 10% by extract_colors's min_proportion) to catch bicolors whose
    # light patch is a minority region (white legs/chest, cream accents).
    if species == "dog" and len(colors) >= 2:
        primary_count = sum(1 for c in colors if c["proportion"] >= 0.15)
        if primary_count >= 2:
            all_L = [c["lab_value"][0] for c in colors]
            is_irregular_texture = (
                lbp.get("lbp_entropy", 0) >= 3.85
                and glcm["glcm_energy_mean"] < 0.22
            )
            if max(all_L) - min(all_L) >= 50 and not is_irregular_texture:
                return "bicolor"

    # Dogs get "irregular" — anything not clearly solid/bicolor/tricolor
    # (merle, dapple, ticked, roan, brindle, noisy/ambiguous coats).
    # Cats fall to "unknown" since their irregular patterns (tortie, tabby,
    # calico) already have dedicated rules.
    return "irregular" if species == "dog" else "unknown"
