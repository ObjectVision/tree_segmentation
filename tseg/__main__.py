"""Allow ``python -m tseg`` without installing the console script."""

from tseg.cli import main

if __name__ == "__main__":
    main()
