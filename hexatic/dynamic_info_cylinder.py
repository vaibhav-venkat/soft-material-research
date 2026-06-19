from pathlib import Path
import sys

if __package__:
    from hexatic.cylinder_dynamics import *  # noqa: F403
    from hexatic.cylinder_dynamics import main
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from hexatic.cylinder_dynamics import *  # noqa: F403
    from hexatic.cylinder_dynamics import main


if __name__ == "__main__":
    main()
