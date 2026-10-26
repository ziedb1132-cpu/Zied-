#!/usr/bin/env python3
"""Run Zied Quant with conservative self-evolution enabled."""
import sys
import zied_quant
import evolution_engine

evolution_engine.install(zied_quant)
zied_quant.main(["zied_quant.py", *sys.argv[1:]])
