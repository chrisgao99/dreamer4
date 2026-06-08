"""Build a lightweight HTML gallery for OOI visual samples."""

from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path
from typing import Dict, List


def _read_manifest(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _rel(path: str, base: Path) -> str:
    return Path(path).resolve().relative_to(base.resolve()).as_posix()


def _mpl_paths(row: Dict[str, str]) -> tuple[str | None, str | None]:
    mp4 = Path(row["mp4_path"])
    preview = Path(row["preview_png"])
    mpl_mp4 = mp4.with_name(f"{mp4.stem}_mpl.mp4")
    mpl_preview = preview.with_name(f"{mp4.stem}_mpl_preview.png")
    return (str(mpl_mp4) if mpl_mp4.exists() else None, str(mpl_preview) if mpl_preview.exists() else None)


def build_gallery(manifest: Path, output: Path) -> None:
    rows = _read_manifest(manifest)
    base = output.parent
    grouped: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row.get("sample_label", "unknown"), []).append(row)

    parts = [
        "<!doctype html>",
        "<html>",
        "<head>",
        '<meta charset="utf-8">',
        "<title>Waymo OOI Visual Samples</title>",
        "<style>",
        "body { font-family: system-ui, sans-serif; margin: 24px; background: #111; color: #eee; }",
        "h1 { margin-bottom: 4px; }",
        "h2 { margin-top: 32px; border-top: 1px solid #444; padding-top: 20px; }",
        ".grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap: 18px; }",
        ".card { background: #1b1b1b; border: 1px solid #333; border-radius: 8px; padding: 12px; }",
        ".card img { width: 100%; height: auto; border-radius: 4px; border: 1px solid #333; }",
        ".thumbs { display: grid; grid-template-columns: 1fr; gap: 8px; }",
        ".tag { display: inline-block; color: #111; background: #ddd; border-radius: 4px; padding: 1px 5px; font-size: 12px; margin-bottom: 4px; }",
        ".meta { color: #bbb; font-size: 13px; line-height: 1.45; margin-top: 8px; }",
        "a { color: #8ecbff; }",
        "code { color: #ddd; }",
        "</style>",
        "</head>",
        "<body>",
        "<h1>Waymo OOI Visual Samples</h1>",
        f"<p>{len(rows)} samples from <code>{html.escape(str(manifest))}</code>.</p>",
        "<p>Orange ring = OOI, white ring = tracks_to_predict. Open MP4 links in VSCode preview or browser.</p>",
    ]

    for label, items in sorted(grouped.items()):
        parts.append(f"<h2>{html.escape(label)} ({len(items)})</h2>")
        parts.append('<div class="grid">')
        for row in items:
            preview = _rel(row["preview_png"], base)
            mp4 = _rel(row["mp4_path"], base)
            mpl_mp4_abs, mpl_preview_abs = _mpl_paths(row)
            scenario = row.get("scenario_id", "")
            labels = row.get("interaction_labels", "")
            ooi_ids = row.get("ooi_track_ids", "")
            focus = row.get("focus_track_id", "")
            mpl_block = ""
            mpl_link = ""
            if mpl_mp4_abs and mpl_preview_abs:
                mpl_mp4 = _rel(mpl_mp4_abs, base)
                mpl_preview = _rel(mpl_preview_abs, base)
                mpl_block = (
                    f'<div><span class="tag">matplotlib</span>'
                    f'<a href="{html.escape(mpl_mp4)}"><img src="{html.escape(mpl_preview)}" alt="{html.escape(scenario)} matplotlib"></a></div>'
                )
                mpl_link = f' | <a href="{html.escape(mpl_mp4)}">Open MPL MP4</a> | <a href="{html.escape(mpl_preview)}">Open MPL PNG</a>'
            parts.extend(
                [
                    '<div class="card">',
                    '<div class="thumbs">',
                    f'<div><span class="tag">opencv</span><a href="{html.escape(mp4)}"><img src="{html.escape(preview)}" alt="{html.escape(scenario)}"></a></div>',
                    mpl_block,
                    "</div>",
                    '<div class="meta">',
                    f"<div><b>scenario</b>: <code>{html.escape(scenario)}</code></div>",
                    f"<div><b>OOI ids</b>: <code>{html.escape(ooi_ids)}</code></div>",
                    f"<div><b>focus</b>: <code>{html.escape(focus)}</code></div>",
                    f"<div><b>labels</b>: {html.escape(labels)}</div>",
                    f'<div><a href="{html.escape(mp4)}">Open OpenCV MP4</a> | <a href="{html.escape(preview)}">Open OpenCV PNG</a>{mpl_link}</div>',
                    "</div>",
                    "</div>",
                ]
            )
        parts.append("</div>")

    parts.extend(["</body>", "</html>"])
    output.write_text("\n".join(parts) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description="Build HTML gallery for OOI visual samples.")
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path("/p/yufeng/tri30/dreamer4/waymo/reports/ooi_visual_samples/selected_samples.csv"),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("/p/yufeng/tri30/dreamer4/waymo/reports/ooi_visual_samples/index.html"),
    )
    args = p.parse_args()
    build_gallery(args.manifest, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
