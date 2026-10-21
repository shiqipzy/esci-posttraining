"""Train the official base -> S -> CE_CONT / REF_KD / FUSION_KD."""
import argparse
import sys

sys.dont_write_bytecode = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('sft', 'fusion'), help='training stage')
    parser.add_argument('args', nargs=argparse.REMAINDER, help='stage options; use STAGE --help')
    args = parser.parse_args()
    sys.argv = [f'{parser.prog} {args.stage}', *args.args]
    if args.stage == 'sft':
        from sft import main as run
    else:
        from fusion import main as run
    run()


if __name__ == '__main__':
    main()
