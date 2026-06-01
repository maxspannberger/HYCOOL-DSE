from pathlib import Path
from hycool_pipe.cases.lh2  import run_analysis as lh2_sizing,  LH2Config
from hycool_pipe.cases.cch2 import run_analysis as cch2_sizing, CcH2Config
from hycool_pipe.cli import _get_march_for_plot, _plot_march, _save_json

# Always write outputs into the TMS folder, regardless of working directory
TMS = Path(__file__).resolve().parent.parent

# ── LH2 ──────────────────────────────────────────────────────────────────────
lh2_json = str(TMS / "lh2_results.json")
lh2_cfg  = LH2Config()
lh2_res  = lh2_sizing(lh2_cfg)
_save_json(lh2_res, lh2_json)
lh2_march = _get_march_for_plot(lh2_res, lh2_cfg, "lh2")
_plot_march(lh2_march, lh2_res, "lh2", lh2_json)   # saves lh2_results.png next to JSON

# ── CcH2 ─────────────────────────────────────────────────────────────────────
cch2_json = str(TMS / "cch2_results.json")
cch2_cfg  = CcH2Config()
cch2_res  = cch2_sizing(cch2_cfg)
_save_json(cch2_res, cch2_json)
cch2_march = _get_march_for_plot(cch2_res, cch2_cfg, "cch2")
_plot_march(cch2_march, cch2_res, "cch2", cch2_json)  # saves cch2_results.png next to JSON
