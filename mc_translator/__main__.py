import sys


def main() -> None:
    if len(sys.argv) > 1:
        from mc_translator.cli import run_cli

        sys.exit(run_cli())
    else:
        from mc_translator.gui.app import run

        run()


if __name__ == "__main__":
    main()
