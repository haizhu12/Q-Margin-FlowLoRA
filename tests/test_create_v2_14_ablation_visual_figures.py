from PIL import ImageFont

import scripts.create_v2_14_ablation_visual_figures as figures


def test_sampling_headers_follow_grouped_image_column_positions(monkeypatch, tmp_path):
    case_ids = [
        "sop_small_valid_004_sop_19647",
        "sop_small_valid_018_sop_12224",
        "sop_small_valid_032_sop_15805",
        "sop_small_valid_035_sop_16188",
    ]
    eval_rows = {
        case_id: {
            "multi_reference_images": ["ref1.png", "ref2.png"],
            "target_image": "target.png",
        }
        for case_id in case_ids
    }
    header_boxes = []

    monkeypatch.setattr(figures, "_read_jsonl", lambda _path: eval_rows)
    monkeypatch.setattr(figures, "_font", lambda _size: object())
    monkeypatch.setattr(
        figures,
        "_fit_resize_pad",
        lambda _path, size: figures.Image.new("RGB", (size, size), "white"),
    )
    monkeypatch.setattr(
        figures,
        "_draw_centered_text",
        lambda _draw, box, _text, _font: header_boxes.append(box),
    )
    monkeypatch.setattr(figures, "_save_figure", lambda _image, _stem, _figures_dir: {})

    figures.create_sampling_copy_risk_figure(
        project_root=tmp_path,
        figures_dir=tmp_path / "figures",
    )

    expected_lefts = figures._column_positions(6, 420, 42, 32, {1, 4})
    assert [box[0] for box in header_boxes] == expected_lefts


def test_font_fallback_preserves_requested_size(monkeypatch):
    requested_sizes = []

    monkeypatch.setattr(
        ImageFont,
        "truetype",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("font unavailable")),
    )
    monkeypatch.setattr(
        ImageFont,
        "load_default",
        lambda *, size: requested_sizes.append(size) or object(),
    )

    figures._font(37)

    assert requested_sizes == [37]
