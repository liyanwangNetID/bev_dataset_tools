from pathlib import Path


def test_finalize_declares_static_enhanced_v03_contract():
    text = (Path(__file__).resolve().parents[2] / "step6" / "finalize.py").read_text(encoding="utf-8")
    assert 'FINALIZER_VERSION = "0.4.0"' in text
    assert '"contract_version": "0.3"' in text
    assert 'step6_dataset_contract_v0.3.json' in text
    assert 'static_occupancy/columns.npz' in text
    assert '"static_occupancy_semantic_id": 0' in text
    assert '"visibility_version": "0.3.3"' in text
    assert 'until occupancy-grid exit' in text
    assert 'space behind remains 255' in text
