"""Module providing a Command Line Parser."""
import argparse


def get_default_parser() -> argparse.ArgumentParser:
    """creates a cli parser object

    Returns:
        argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser()

    # The command line args should be added here

    return parser
