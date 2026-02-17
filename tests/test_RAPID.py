import pandas as pd
import pytest
import yaml
from pathlib import Path

from rapid.RAPID import main

@pytest.fixture
def data_path():
    return Path(__file__).parent.parent / "data"


@pytest.fixture
def make_rapid_config(tmp_path, data_path):
    """Factory fixture for creating RAPID configs with custom overrides."""

    def _make(**overrides):
        config = {
            "q_imgs": str(data_path / "demo/example_query"),
            "db_imgs": str(data_path / "demo/example_database"),
            "nr_kps": 150,
            "topN_ids_match": 10,
            "conf_score_limit": 500,
            "conf_threshold": 0.5,
            "extend_db_while_proc": False,
            "db_annoy_index_path": str(data_path / "demo/saved_RAPID/db_index.ann"),  # path to ANN index (optional)
            "idx_map_path": str(data_path / "demo/saved_RAPID/db_idx_map.pkl"),  # path to index map (optional)
            "closed_correct_ratios_path": str(data_path / "demo/saved_RAPID/closed_set_correct_ratios.npy"),  # path to closed correct ratios (optional)
            "closed_false_ratios_path": str(data_path / "demo/saved_RAPID/closed_set_false_ratios.npy"),  # path to closed false ratios (optional)
            "open_ratios_path": str(data_path / "demo/saved_RAPID/open_set_ratios.npy")  # path to open ratios (optional)
        }
        config.update(overrides)

        (tmp_path / "query").mkdir(exist_ok=True)
        (tmp_path / "database").mkdir(exist_ok=True)

        config_path = tmp_path / "config_RAPID.yaml"
        config_path.write_text(yaml.dump(config, default_flow_style=False))
        return config_path

    return _make


@pytest.fixture
def make_rapid_config_build_db(tmp_path, data_path):
    """Factory fixture for creating RAPID configs with custom overrides."""

    def _make(**overrides):
        config = {
            "q_imgs": str(data_path / "demo/example_query"),
            "db_imgs": str(data_path / "demo/example_database"),
            "nr_kps": 150,
            "topN_ids_match": 10,
            "conf_score_limit": 500,
            "conf_threshold": 0.5,
            "extend_db_while_proc": False,
            "db_annoy_index_path": "",  # path to ANN index (optional)
            "idx_map_path": "",  # path to index map (optional)
            "closed_correct_ratios_path": "",  # path to closed correct ratios (optional)
            "closed_false_ratios_path": "",  # path to closed false ratios (optional)
            "open_ratios_path": ""  # path to open ratios (optional)
        }
        config.update(overrides)

        (tmp_path / "query").mkdir(exist_ok=True)
        (tmp_path / "database").mkdir(exist_ok=True)

        config_path = tmp_path / "config_RAPID.yaml"
        config_path.write_text(yaml.dump(config, default_flow_style=False))
        return config_path

    return _make


class TestRapid:
    @pytest.mark.skip(reason="Test will fail if data is not there.")
    def test_RAPID_config_parses_valid_yaml(self, make_rapid_config, data_path):
        config_path = make_rapid_config()

        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        assert isinstance(config_path, Path)

        main(config_path=config_path)

        df_rapid_result = pd.read_csv(data_path / "demo/saved_RAPID/prediction_results_98%_84query.csv")

        assert len(df_rapid_result) == 84, "84 Query images"
        assert len(df_rapid_result.columns) == 16, "There should be exactly 16 columns"



    def test_RAPID_build_db(self, make_rapid_config_build_db, data_path):
        config_path = make_rapid_config_build_db()

        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        assert isinstance(config_path, Path)

        main(config_path=config_path)

        df_rapid_result = pd.read_csv(data_path / "demo/saved_RAPID/prediction_results_98%_84query.csv")

        assert len(df_rapid_result) == 84, "84 Query images"
        assert len(df_rapid_result.columns) == 16, "There should be exactly 16 columns"

