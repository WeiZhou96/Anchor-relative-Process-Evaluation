"""K-b-compatible APE CLI entry point for old and new cards. No metrics run on import."""
from systems.r2_identity import install
install()
from ape.cli import main
if __name__=='__main__': raise SystemExit(main())
