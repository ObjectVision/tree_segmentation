# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Object Vision B.V. and tseg contributors
"""Allow ``python -m tseg`` without installing the console script."""

from tseg.cli import main

if __name__ == "__main__":
    main()
