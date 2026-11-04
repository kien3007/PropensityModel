"""Run a no-data installation smoke check."""
from tecd_propensity import __version__


def main() -> None:
    """Print the package version without accessing external data."""
    print(f"tecd-propensity {__version__}")


if __name__ == "__main__":
    main()
