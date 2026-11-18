from pathlib import Path
import argparse
from peppreprocess.main import main




def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline")

    # --------------------------------------------------------
    # Input / Output
    # --------------------------------------------------------
    parser.add_argument(
        "--input", type=Path, default="datasets",help="Thư mục chứa ảnh đầu vào.")
    parser.add_argument(
        "--output", type=Path,default="results", help="Thư mục lưu ảnh đã xử lý.")
    parser.add_argument(
        "--csv", type=Path, default="results/processing_results.csv", help="Đường dẫn CSV kết quả. Mặc định: output/processing_results.csv")

    # --------------------------------------------------------
    # Pipeline configuration
    # --------------------------------------------------------
    parser.add_argument(
        "--mode", choices=["full", "gp_only", "cde_only", "none"], default="full", help="Chế độ Step 03.")
    parser.add_argument(
        "--step", type=int, nargs=3, choices=[0, 1], default=[1, 1, 1], metavar=("STEP1", "STEP2", "STEP3"),
        help="Bật/tắt 3 bước: 1 = bật, 0 = tắt.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    main(
        main_preprocess_mode=args.mode,
        input_dir=args.input,
        output_dir=args.output,
        csv_path=(
            args.csv
            if args.csv is not None
            else args.output / "processing_results.csv"
        ),
        step=args.step,
    )