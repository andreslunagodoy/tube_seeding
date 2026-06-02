import juliapkg
from pathlib import Path
script_path = Path(__file__)
root_dir = script_path.parent.parent
# Add Julia package
juliapkg.add("FeynGym", "5b049d17-076e-46a2-a291-65c7ac6d0766", path=str(root_dir/"FeynGym.jl"), dev=True)
juliapkg.add("SparseSolveExact", "cbcf3da8-18ff-47ec-83ec-bcc90e4636f1", path=str(root_dir/"SparseSolveExact.jl"), dev=True)
juliapkg.resolve()
