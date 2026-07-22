"""Tests for dataset.py: loading instance definitions from data/ into the
Dict shape run_baseline/run_kg_augmented expect.
"""

from kg_test_generation.dataset import load_all_instances, load_instance


class TestLoadInstance:
    def test_loads_expected_fields(self):
        instance = load_instance("send_2012")

        assert instance["name"] == "send_2012"
        assert instance["repo"] == "psf/requests"
        assert instance["code_file"] == "requests/sessions.py"
        assert len(instance["base_commit"]) == 40  # full git SHA
        assert "diff --git" in instance["patch"]

    def test_patch_is_read_from_the_referenced_file(self):
        instance = load_instance("raise_for_status_2011")
        assert "raise_for_status" in instance["patch"]


class TestLoadAllInstances:
    def test_loads_all_dataset_instances(self):
        instances = load_all_instances()
        assert len(instances) == 22

    def test_sorted_by_name_for_determinism(self):
        instances = load_all_instances()
        names = [i["name"] for i in instances]
        assert names == sorted(names)

    def test_every_instance_has_the_shape_pipeline_expects(self):
        required_keys = {"name", "repo", "base_commit", "patch", "code_file", "test_file"}
        for instance in load_all_instances():
            assert required_keys <= instance.keys()
