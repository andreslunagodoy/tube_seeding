#!/usr/bin/env python3

import argparse
import math
import sys
from pathlib import Path


DATA_PATH = Path("save/resorted_seed_op_list.txt")
DEFAULT_TARGET_MARKER = (6, 6)
DEFAULT_DELAY_MS = 100
PYFEYNGYM_SRC = (
    Path(__file__).resolve().parents[2] / "FeynGym" / "pyfeyngym" / "src"
)


def load_visualize_seeds():
    try:
        import pyfeyngym
    except ModuleNotFoundError as exc:
        if exc.name != "pyfeyngym":
            raise RuntimeError(
                "Failed to import pyfeyngym because one of its dependencies is missing. "
                f"Original error: {exc}"
            ) from exc

        if not PYFEYNGYM_SRC.exists():
            raise RuntimeError(
                "Could not import pyfeyngym and could not find the local source tree at "
                f"{PYFEYNGYM_SRC}."
            ) from exc

        sys.path.insert(0, str(PYFEYNGYM_SRC))
        try:
            import pyfeyngym
        except ModuleNotFoundError as inner_exc:
            raise RuntimeError(
                "Found the local pyfeyngym source tree, but importing it still failed. "
                f"Original error: {inner_exc}"
            ) from inner_exc

    return pyfeyngym.visualize_seeds


def load_seeds(path: Path):
    seeds = []

    with path.open() as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 3:
                raise ValueError(
                    f"Expected 3 integers on line {line_number}, got: {raw_line.rstrip()!r}"
                )

            x, y, seed_type = map(int, parts)
            if seed_type not in (1, 2):
                raise ValueError(
                    f"Expected type 1 or 2 on line {line_number}, got: {seed_type}"
                )

            seeds.append((x, y, seed_type))

    return seeds


def parse_marker(value: str):
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            "target marker must be in the form x,y (for example: 6,6)"
        )

    try:
        x, y = map(float, parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "target marker coordinates must be numeric"
        ) from exc

    return x, y


def compute_board_size(seeds, target_integral):
    if not seeds:
        raise ValueError("No points found in the input data file")

    max_x = max(x for x, _, _ in seeds)
    max_y = max(y for _, y, _ in seeds)
    max_x = max(max_x, math.ceil(target_integral[0]))
    max_y = max(max_y, math.ceil(target_integral[1]))
    max_x += 1
    max_y += 1

    # `visualize_seeds` interprets `board_size` as board dimensions, with the
    # largest visible seed coordinate being `board_size - 2`.
    return (max_x + 2, max_y + 2)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize seeds from save/resorted_seed_op_list.txt using pyfeyngym."
    )
    parser.add_argument(
        "--data-file",
        type=Path,
        default=DATA_PATH,
        help=f"Path to the input data file. Default: {DATA_PATH}",
    )
    parser.add_argument(
        "--target-marker",
        type=parse_marker,
        default=DEFAULT_TARGET_MARKER,
        metavar="X,Y",
        help="Target integral to highlight and always include in the visualization. "
        f"Default: {DEFAULT_TARGET_MARKER[0]},{DEFAULT_TARGET_MARKER[1]}",
    )
    parser.add_argument(
        "--no-animation",
        action="store_true",
        help="Disable animation playback and render the final static plot directly. "
        "Useful for batch runs.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--show",
        action="store_true",
        help="Show the plot interactively.",
    )
    mode.add_argument(
        "--save",
        type=Path,
        metavar="PNG_PATH",
        help="Save the plot to a PNG file.",
    )
    return parser.parse_args()


def suppress_animation_delete_warning(anim):
    if anim is not None:
        anim._draw_was_started = True


def save_static_figure(fig, output_path, anim):
    suppress_animation_delete_warning(anim)
    fig.savefig(output_path, dpi=200)


def backend_supports_preview(plt):
    backend = plt.get_backend().lower()
    noninteractive_backends = {"agg", "cairo", "pdf", "pgf", "ps", "svg", "template"}
    return backend not in noninteractive_backends


def main():
    args = parse_args()
    seeds = load_seeds(args.data_file)
    target_integral = args.target_marker
    board_size = compute_board_size(seeds, target_integral)
    visualize_seeds = load_visualize_seeds()

    fig, ax, anim = visualize_seeds(
        seeds=seeds,
        target_integral=target_integral,
        board_size=board_size,
        delay_ms=DEFAULT_DELAY_MS,
    )
    # ax.set_title("IBP equations used")
    fig.tight_layout()

    import matplotlib.pyplot as plt

    if args.save is not None:
        if args.no_animation or anim is None or not backend_supports_preview(plt):
            save_static_figure(fig, args.save, anim)
            plt.close(fig)
            return

        fig._pyfeyngym_anim = anim
        plt.show(block=False)
        preview_seconds = len(seeds) * DEFAULT_DELAY_MS / 1000.0 + 0.5
        plt.pause(preview_seconds)
        save_static_figure(fig, args.save, anim)
        plt.close(fig)
    elif args.show:
        if args.no_animation:
            suppress_animation_delete_warning(anim)
        else:
            fig._pyfeyngym_anim = anim
        plt.show()


if __name__ == "__main__":
    main()
