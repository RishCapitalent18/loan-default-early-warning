"""Generate the messy loan book, then rebuild it and train the early-warning model."""
from src.ews import generate_data  # noqa: F401  (runs on import)
from src.ews.pipeline import main

if __name__ == "__main__":
    main()
