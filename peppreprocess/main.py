import argparse
import csv
import cv2
import numpy as np
import traceback
import json
import os
import tempfile

from pathlib import Path
from .step_01.global_deskew import global_deskew
from .step_02.crop_images import crop
from .step_03.preprocess import preprocess


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


# ============================================================
# PARAMETERS
# ============================================================

parameters = {
    # Step 01: Global deskew
    "threshold_deskew": 170,
    "kernel_size": 15,
    "connectivity": 8,
    "min_angle": 0.3,

    # Step 02: Crop
    "max_side": 1500,
    "mask_background": False,
    "perspective_rectified": False,

    # Step 03: Main preprocess
    "kernel_size_median_blur": 3,
    "minimum": 15,
    "maximum": 71,
    "low_percentile_stretch_illum": 0.7,
    "high_percentile_stretch_illum": 99.3,
    "scale": 180.0,
    "low_percentile_stretch_tophat": 0.5,
    "high_percentile_stretch_tophat": 99.5,
    "flip": True,
}


# ============================================================
# CSV CONFIGURATION
# ============================================================

CSV_FIELDS = [
    # Existing tracking fields
    "STT", "Ảnh", "góc", "status 1", "status 2", "status 3", "status 4",
    "Ảnh đầu ra", "Lỗi bước", "Chi tiết lỗi",
    # Schema / source and output dimensions
    "reconstruction_version", "input_width", "input_height", "input_channels", "input_dtype",
    "deskew_width", "deskew_height", "crop_x0", "crop_y0", "crop_x1", "crop_y1",
    "crop_width_before_perspective", "crop_height_before_perspective",
    "perspective_rectified", "perspective_width", "perspective_height",
    "flip_applied", "output_width", "output_height", "output_channels", "output_dtype",
    # Exact geometry and full configuration stored as JSON
    "affine_matrix_json", "perspective_matrix_json", "perspective_source_points_json",
    "perspective_destination_points_json", "geometry_json", "parameters_json",
    # Crop diagnostics useful for auditing localization
    "crop_method", "crop_confidence", "downscale_factor", "bbox_low_resolution_json",
]

RECONSTRUCTION_VERSION = "4.0"


def json_cell(value):
    """Serialize metadata to compact, UTF-8-friendly JSON for a CSV cell."""
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def ensure_csv_schema(csv_path: Path) -> None:
    """Migrate an older CSV header to the current schema, preserving old rows."""
    if not csv_path.is_file() or csv_path.stat().st_size == 0:
        return

    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        old_fields = reader.fieldnames or []
        old_rows = list(reader)

    if old_fields == CSV_FIELDS:
        return

    # Rewrite to a temporary file first so an interrupted migration does not
    # leave the original CSV half-written.
    fd, temp_name = tempfile.mkstemp(prefix="processing_results_", suffix=".csv", dir=str(csv_path.parent))
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        with temp_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for old_row in old_rows:
                migrated = {field: old_row.get(field, "") for field in CSV_FIELDS}
                writer.writerow(migrated)
        os.replace(temp_path, csv_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()



def load_csv_results(csv_path: Path) -> dict:
    """
    Đọc CSV một lần lúc khởi động.
    Nếu một ảnh có nhiều bản ghi, sử dụng bản ghi cuối cùng.
    """
    rows = {}

    if not csv_path.is_file() or csv_path.stat().st_size == 0:
        return rows

    with csv_path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        reader = csv.DictReader(csv_file)

        if not reader.fieldnames or "Ảnh" not in reader.fieldnames:
            raise ValueError(
                f"CSV không đúng cấu trúc hoặc thiếu cột 'Ảnh': "
                f"{csv_path}"
            )

        for row in reader:
            key = row.get("Ảnh", "")

            if key:
                rows[key] = row

    return rows


def append_csv_row(writer, csv_file, row: dict) -> None:
    """
    Thêm một dòng vào CSV đang mở.
    Flush sau mỗi ảnh để hạn chế mất dữ liệu khi chương trình dừng.
    """
    writer.writerow(row)
    csv_file.flush()


def is_ok(value) -> bool:
    try:
        return int(value) == 1
    except (TypeError, ValueError):
        return False



def read_image(image_path: Path):
    """
    Đọc ảnh có đường dẫn chứa ký tự Unicode/tiếng Việt.
    Trả về None nếu không đọc hoặc giải mã được ảnh.
    """
    try:
        image_bytes = np.fromfile(
            str(image_path),
            dtype=np.uint8,
        )

        if image_bytes.size == 0:
            return None

        return cv2.imdecode(
            image_bytes,
            cv2.IMREAD_COLOR,
        )

    except (OSError, ValueError, cv2.error):
        return None


def write_image(image_path: Path, image: np.ndarray) -> bool:
    """
    Ghi ảnh hỗ trợ đường dẫn chứa ký tự Unicode.
    """
    try:
        image_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        extension = image_path.suffix.lower() or ".jpg"

        success, encoded = cv2.imencode(
            extension,
            image,
        )

        if not success:
            return False

        encoded.tofile(str(image_path))
        return True

    except (OSError, ValueError, cv2.error):
        return False



def process_image(
    input_path: Path,
    output_path: Path,
    parameters: dict,
    step=(True, True, True),
    main_preprocess_mode: str = "full",
) -> dict:

    if len(step) != 3 or not all(isinstance(x, (bool, np.bool_)) for x in step):
        raise ValueError("step phải gồm đúng 3 giá trị Boolean.")

    if main_preprocess_mode not in {"full", "gp_only", "cde_only", "none"}:
        raise ValueError(
            "main_preprocess_mode phải là full, gp_only, cde_only hoặc none."
        )

    step = list(step)

    run_parameters = dict(parameters)

    if not step[0]:
        for key in [
            "threshold_deskew",
            "kernel_size",
            "connectivity",
            "min_angle",
        ]:
            run_parameters[key] = None

    if not step[1]:
        for key in [
            "max_side",
            "mask_background",
            "perspective_rectified",
        ]:
            run_parameters[key] = None

    if not step[2]:
        for key in [
            "kernel_size_median_blur",
            "minimum",
            "maximum",
            "low_percentile_stretch_illum",
            "high_percentile_stretch_illum",
            "scale",
            "low_percentile_stretch_tophat",
            "high_percentile_stretch_tophat",
            "flip",
        ]:
            run_parameters[key] = None

    run_parameters["steps"] = step
    run_parameters["main_preprocess_mode"] = (
        main_preprocess_mode if step[2] else None
    )

    result = {
        "success": False,
        "angle": None,
        "status 1": None,
        "status 2": None,
        "status 3": None,
        "status 4": -1,
        "error_step": "",
        "error": "",
        "geometry": {},
        "input_shape": None,
        "output_shape": None,
        "parameters": run_parameters,
    }

    # --------------------------------------------------------
    # Read input
    # --------------------------------------------------------
    image = read_image(input_path)

    if image is None:
        result.update({
            "error_step": "Read input",
            "error": f"Cannot read image: {input_path}",
        })
        return result

    input_shape = list(image.shape)
    result["input_shape"] = input_shape

    current_image = image.copy()

    # --------------------------------------------------------
    # Step 01: Global deskew
    # --------------------------------------------------------
    if step[0]:
        try:
            output_step_01, angle, affine_matrix = global_deskew(
                current_image,
                parameters["threshold_deskew"],
                parameters["kernel_size"],
                parameters["connectivity"],
                parameters["min_angle"],
            )

            current_image = output_step_01
            result["angle"] = angle

            result["geometry"].update({
                "coordinate_convention":
                    "pixel coordinates; crop bbox uses x0,y0 inclusive and x1,y1 exclusive",
                "input_shape": input_shape,
                "deskew_shape": list(current_image.shape),
                "deskew_angle_estimated_degrees": float(angle),
                "deskew_angle_applied_degrees": (
                    float(angle)
                    if abs(float(angle)) >= parameters["min_angle"]
                    else 0.0
                ),
                "affine_matrix_original_to_deskew": affine_matrix.tolist(),
            })

            result["status 1"] = 1

        except Exception as e:
            result.update({
                "status 1": 0,
                "error_step": "Step 01",
                "error": str(e),
            })
            return result

    # --------------------------------------------------------
    # Step 02: Crop
    # --------------------------------------------------------
    if step[1]:
        try:
            output_step_02, _, crop_meta = crop(
                current_image,
                max_side=parameters["max_side"],
                mask_background=parameters["mask_background"],
                perspective_rectified=parameters["perspective_rectified"],
            )

            current_image = output_step_02

            crop_bbox = crop_meta["crop_bbox_original"]
            crop_before_shape = crop_meta["crop_shape_before_perspective"]
            perspective_shape = crop_meta["perspective_shape"]

            result["geometry"].update({
                "crop_bbox_on_deskew_image_xyxy": crop_bbox,
                "crop_shape_before_perspective_hw": crop_before_shape,
                "perspective_rectified": bool(
                    crop_meta["perspective_rectified"]
                ),
                "perspective_matrix_crop_to_rectified":
                    crop_meta["perspective_matrix"],
                "perspective_source_points_xy":
                    crop_meta["perspective_src_points"],
                "perspective_destination_points_xy":
                    crop_meta["perspective_dst_points"],
                "perspective_output_shape_hw": perspective_shape,
                "crop_method": crop_meta["crop_method"],
                "crop_confidence": crop_meta["crop_confidence"],
                "downscale_factor_for_localization":
                    crop_meta["downscale_factor"],
                "bbox_low_resolution_xyxy":
                    crop_meta["bbox_low_resolution"],
            })

            result["status 2"] = 1

        except Exception as e:
            result.update({
                "status 2": 0,
                "error_step": "Step 02",
                "error": str(e),
            })
            return result

    # --------------------------------------------------------
    # Step 03: Main preprocess
    # --------------------------------------------------------
    if step[2]:
        try:
            step03_input_shape = list(current_image.shape)

            output_step_03 = preprocess(
                current_image,
                kernel_size_median_blur=parameters["kernel_size_median_blur"],
                minimum=parameters["minimum"],
                maximum=parameters["maximum"],
                low_percentile_stretch_illum=parameters["low_percentile_stretch_illum"],
                high_percentile_stretch_illum=parameters["high_percentile_stretch_illum"],
                scale=parameters["scale"],
                low_percentile_stretch_tophat=parameters["low_percentile_stretch_tophat"],
                high_percentile_stretch_tophat=parameters["high_percentile_stretch_tophat"],
                flip=parameters["flip"],
                mode=main_preprocess_mode,
            )

            current_image = output_step_03

            result["geometry"].update({
                "step03_input_shape": step03_input_shape,
                "step03_output_shape": list(current_image.shape),
                "horizontal_flip_applied": bool(parameters["flip"]),
                "main_preprocess_mode": main_preprocess_mode,
            })

            result["status 3"] = 1

        except Exception as e:
            result.update({
                "status 3": 0,
                "error_step": "Step 03",
                "error": str(e),
            })
            return result

    # --------------------------------------------------------
    # Save final output
    # --------------------------------------------------------
    try:
        if not write_image(output_path, current_image):
            raise IOError(f"Cannot save image: {output_path}")

    except Exception as e:
        result.update({
            "error_step": "Save output",
            "error": str(e),
        })
        return result

    result["output_shape"] = list(current_image.shape)
    result["success"] = True

    return result



def main(
    main_preprocess_mode: str,
    input_dir: Path,
    output_dir: Path,
    csv_path: Path,
    step: tuple,
    parameters=parameters,
):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    csv_path = Path(csv_path)
    step = tuple(bool(value) for value in step)

    if len(step) != 3 or not all(
        isinstance(x, (bool, np.bool_)) for x in step
    ):
        raise ValueError(
            "step phải có dạng [True/False, True/False, True/False]"
        )

    step = list(step)

    if main_preprocess_mode not in {"full", "gp_only", "cde_only", "none"}:
        raise ValueError(
            "main_preprocess_mode phải là full, gp_only, cde_only hoặc none."
        )

    # --------------------------------------------------------
    # Validate input directory
    # --------------------------------------------------------
    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input directory does not exist: {input_dir}"
        )

    if not input_dir.is_dir():
        raise NotADirectoryError(
            f"Input path is not a directory: {input_dir}"
        )

    # --------------------------------------------------------
    # Find all images recursively
    # --------------------------------------------------------
    image_paths = sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    total_images = len(image_paths)

    if total_images == 0:
        print(f"No images found in: {input_dir}")
        return

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Upgrade the CSV header if it was created by an older version.
    # Old rows are preserved; rows without geometry metadata will be reprocessed.
    ensure_csv_schema(csv_path)

    # Load previous results once.
    rows = load_csv_results(csv_path)

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------
    processed_count = 0
    skipped_count = 0
    failed_count = 0

    print("=" * 70)
    print("BATCH PREPROCESSING")
    print("=" * 70)
    print(f"Main preprocess mode: {main_preprocess_mode}")
    print(f"Input  : {input_dir}")
    print(f"Output : {output_dir}")
    print(f"CSV    : {csv_path}")
    print(f"Images : {total_images}")
    print(f"Step configuration: {step}")
    print(
        "Pipeline: " +
        " -> ".join(
            name
            for enabled, name in zip(
                step,
                ["Step 01", "Step 02", "Step 03"]
            )
            if enabled
        )
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Open CSV ONCE and keep it open during the whole batch.
    # Do not open the CSV in Excel while this script runs.
    # --------------------------------------------------------
    csv_exists = (
        csv_path.is_file()
        and csv_path.stat().st_size > 0
    )

    try:
        with csv_path.open(
            "a",
            newline="",
            encoding="utf-8-sig",
        ) as csv_file:

            writer = csv.DictWriter(
                csv_file,
                fieldnames=CSV_FIELDS,
                extrasaction="raise",
            )

            if not csv_exists:
                writer.writeheader()
                csv_file.flush()

            # ------------------------------------------------
            # Process each image
            # ------------------------------------------------
            for index, input_path in enumerate(
                image_paths,
                start=1,
            ):

                relative_path = input_path.relative_to(
                    input_dir
                )

                relative_key = relative_path.as_posix()

                output_path = output_dir / relative_path

                previous = rows.get(relative_key, {})

                print(
                    f"\rProgress: {index}/{total_images}",
                    end="",
                    flush=True,
                )

                # --------------------------------------------
                # Skip previously successful images
                # --------------------------------------------
                output_is_valid = False

                if output_path.is_file():
                    output_is_valid = (
                        read_image(output_path) is not None
                    )

                # Kiểm tra cấu hình step trước đó
                try:
                    previous_parameters = json.loads(
                        previous.get("parameters_json") or "{}"
                    )
                    previous_step = previous_parameters.get("steps")
                    previous_mode = previous_parameters.get("main_preprocess_mode")
                except (json.JSONDecodeError, TypeError):
                    previous_step = None
                    previous_mode = None

                # Các bước được bật phải có status = 1.
                # Các bước bị tắt phải có status rỗng/None.
                status_matches = all(
                    (
                        is_ok(previous.get(f"status {i + 1}"))
                        if enabled
                        else not previous.get(f"status {i + 1}")
                    )
                    for i, enabled in enumerate(step)
                )

                already_complete = (
                    output_is_valid
                    and status_matches
                    and previous_step == step
                    and previous.get("reconstruction_version") == RECONSTRUCTION_VERSION
                    and bool(previous.get("geometry_json"))
                    and (
                        not step[2]
                        or previous_mode == main_preprocess_mode
                    )
                )

                if already_complete:
                    skipped_count += 1
                    continue

                # --------------------------------------------
                # Process image
                # --------------------------------------------
                result = process_image(
                    input_path=input_path,
                    output_path=output_path,
                    parameters=parameters,
                    step=step,
                    main_preprocess_mode=main_preprocess_mode,
                )

                geometry = result.get("geometry", {})
                input_shape = result.get("input_shape") or []
                output_shape = result.get("output_shape") or []
                deskew_shape = geometry.get("deskew_shape") or []
                crop_bbox = geometry.get("crop_bbox_on_deskew_image_xyxy") or []
                crop_pre_shape = geometry.get("crop_shape_before_perspective_hw") or []
                perspective_shape = geometry.get("perspective_output_shape_hw") or []

                row = {
                    "STT": str(index),
                    "Ảnh": relative_key,
                    "góc": "" if result["angle"] is None else result["angle"],
                    "status 1": result["status 1"],
                    "status 2": result["status 2"],
                    "status 3": result["status 3"],
                    "status 4": result["status 4"],
                    "Ảnh đầu ra": relative_path.as_posix(),
                    "Lỗi bước": result["error_step"],
                    "Chi tiết lỗi": result["error"],
                    "reconstruction_version": RECONSTRUCTION_VERSION if result["success"] else "",
                    "input_width": input_shape[1] if len(input_shape) >= 2 else "",
                    "input_height": input_shape[0] if len(input_shape) >= 2 else "",
                    "input_channels": input_shape[2] if len(input_shape) >= 3 else 1 if len(input_shape) == 2 else "",
                    "input_dtype": "uint8" if input_shape else "",
                    "deskew_width": deskew_shape[1] if len(deskew_shape) >= 2 else "",
                    "deskew_height": deskew_shape[0] if len(deskew_shape) >= 2 else "",
                    "crop_x0": crop_bbox[0] if len(crop_bbox) == 4 else "",
                    "crop_y0": crop_bbox[1] if len(crop_bbox) == 4 else "",
                    "crop_x1": crop_bbox[2] if len(crop_bbox) == 4 else "",
                    "crop_y1": crop_bbox[3] if len(crop_bbox) == 4 else "",
                    "crop_width_before_perspective": crop_pre_shape[1] if len(crop_pre_shape) >= 2 else "",
                    "crop_height_before_perspective": crop_pre_shape[0] if len(crop_pre_shape) >= 2 else "",
                    "perspective_rectified": geometry.get("perspective_rectified", ""),
                    "perspective_width": perspective_shape[1] if len(perspective_shape) >= 2 else "",
                    "perspective_height": perspective_shape[0] if len(perspective_shape) >= 2 else "",
                    "flip_applied": geometry.get("horizontal_flip_applied", ""),
                    "output_width": output_shape[1] if len(output_shape) >= 2 else "",
                    "output_height": output_shape[0] if len(output_shape) >= 2 else "",
                    "output_channels": output_shape[2] if len(output_shape) >= 3 else 1 if len(output_shape) == 2 else "",
                    "output_dtype": "uint8" if output_shape else "",
                    "affine_matrix_json": json_cell(geometry.get("affine_matrix_original_to_deskew")),
                    "perspective_matrix_json": json_cell(geometry.get("perspective_matrix_crop_to_rectified")),
                    "perspective_source_points_json": json_cell(geometry.get("perspective_source_points_xy")),
                    "perspective_destination_points_json": json_cell(geometry.get("perspective_destination_points_xy")),
                    "geometry_json": json_cell(geometry),
                    "parameters_json": json_cell(result.get("parameters", parameters)),
                    "crop_method": geometry.get("crop_method", ""),
                    "crop_confidence": geometry.get("crop_confidence", ""),
                    "downscale_factor": geometry.get("downscale_factor_for_localization", ""),
                    "bbox_low_resolution_json": json_cell(geometry.get("bbox_low_resolution_xyxy")),
                }

                # Append one record without reopening the CSV.
                append_csv_row(
                    writer,
                    csv_file,
                    row,
                )

                # Keep latest record in memory for this run.
                rows[relative_key] = row
                processed_count += 1

                failed_steps = [
                    f"status {i + 1}"
                    for i, enabled in enumerate(step)
                    if enabled and not is_ok(row[f"status {i + 1}"])
                ]

                if failed_steps:
                    failed_count += 1

                    print(
                        f"\nFAILED: {relative_key}"
                        f" | {', '.join(failed_steps)}"
                        f" | {row['Lỗi bước']}"
                        f" | {row['Chi tiết lỗi']}"
                    )

    except PermissionError:
        print(
            "\nCannot open/write the CSV. "
            "Close Excel or other applications using this file, "
            "then run the script again."
        )
        raise

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------
    print("\n")
    print("=" * 70)
    print("BATCH SUMMARY")
    print("=" * 70)
    print(f"Total images      : {total_images}")
    print(f"Processed         : {processed_count}")
    print(f"Skipped           : {skipped_count}")
    print(f"Failed this run   : {failed_count}")
    print(f"CSV results       : {csv_path}")
    print("=" * 70)

