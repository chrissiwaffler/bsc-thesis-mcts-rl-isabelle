import pytest

from mcts.data_minif2f import load_dataset


class TestLoadDataset:
    """Test cases for miniF2F dataset loading"""

    def test_load_valid_split(self):
        """Test loading the valid split"""
        path_to_minif2f = "./miniF2F-facebook/"
        theorems = load_dataset(path_to_minif2f, "valid")

        assert len(theorems) > 0, "Should load at least one theorem"
        assert all(
            isinstance(t["name"], str) for t in theorems
        ), "All theorems should have string names"
        assert all(
            isinstance(t["content"], str) for t in theorems
        ), "All theorems should have string content"
        assert all(
            "theorem" in t["content"] for t in theorems
        ), "All theorem contents should contain 'theorem'"

    def test_load_test_split(self):
        """Test loading the test split"""
        path_to_minif2f = "./miniF2F-facebook/"
        theorems = load_dataset(path_to_minif2f, "test")

        assert len(theorems) > 0, "Should load at least one theorem"
        assert all(
            isinstance(t["name"], str) for t in theorems
        ), "All theorems should have string names"
        assert all(
            isinstance(t["content"], str) for t in theorems
        ), "All theorems should have string content"
        assert all(
            "theorem" in t["content"] for t in theorems
        ), "All theorem contents should contain 'theorem'"

    def test_invalid_split_raises_error(self):
        """Test that invalid split raises ValueError"""
        path_to_minif2f = "./miniF2F-facebook/"

        with pytest.raises(ValueError, match="split must be either 'test' or 'valid'"):
            load_dataset(path_to_minif2f, "invalid")

    def test_nonexistent_directory_raises_error(self):
        """Test that nonexistent directory raises FileNotFoundError"""
        with pytest.raises(FileNotFoundError):
            load_dataset("./nonexistent/", "valid")

    def test_theorem_files_exist(self):
        """Test that theorem files are actually loaded from disk"""
        path_to_minif2f = "./miniF2F-facebook/"
        theorems = load_dataset(path_to_minif2f, "valid")

        # check that at least some expected theorems are present
        theorem_names = {t["name"] for t in theorems}
        expected_theorems = {
            "mathd_numbertheory_12",
            "mathd_algebra_59",
            "mathd_numbertheory_335",
        }

        # at least one of the expected theorems should be present
        assert (
            len(theorem_names & expected_theorems) > 0
        ), f"None of {expected_theorems} found in {theorem_names}"

    def test_theorem_content_structure(self):
        """Test that theorem content has expected Isabelle structure"""
        path_to_minif2f = "./miniF2F-facebook/"
        theorems = load_dataset(path_to_minif2f, "valid")

        # take first theorem for detailed checks
        theorem = theorems[0]
        content = theorem["content"]

        # should contain basic isabelle theory structure
        assert "theory" in content, "Content should contain 'theory'"
        assert "begin" in content, "Content should contain 'begin'"
        assert "end" in content, "Content should contain 'end'"
        assert "theorem" in content, "Content should contain 'theorem'"

        # should contain sorry (since these are unsolved theorems)
        assert "sorry" in content, "Content should contain 'sorry'"

    def test_theorem_name_matches_filename(self):
        """Test that theorem names match their filenames"""
        path_to_minif2f = "./miniF2F-facebook/"
        theorems = load_dataset(path_to_minif2f, "valid")

        # check a few theorems to ensure names match content
        for theorem in theorems[:5]:  # Check first 5
            name = theorem["name"]
            content = theorem["content"]

            # the theorem name should appear in the content
            assert (
                name in content
            ), f"Theorem name '{name}' should appear in its content"

    def test_dataset_consistency(self):
        """Test that loading the same dataset multiple times gives consistent results"""
        path_to_minif2f = "./miniF2F-facebook/"

        theorems1 = load_dataset(path_to_minif2f, "valid")
        theorems2 = load_dataset(path_to_minif2f, "valid")

        assert len(theorems1) == len(theorems2), "Dataset sizes should be consistent"

        # check that theorem names are the same (order might differ)
        names1 = {t["name"] for t in theorems1}
        names2 = {t["name"] for t in theorems2}
        assert names1 == names2, "Theorem names should be consistent"


class TestDatasetIntegration:
    """Integration tests for dataset loading with theorem extraction"""

    def test_dataset_with_theorem_extraction(self):
        """Test that loaded theorems can be processed by theorem extraction"""
        from mcts_accelerate.utils import extract_theorem_statement

        path_to_minif2f = "./miniF2F-facebook/"
        theorems = load_dataset(path_to_minif2f, "valid")[:3]  # test first 3

        for theorem in theorems:
            content = theorem["content"]
            name = theorem["name"]

            # Should be able to extract theorem statement without error
            try:
                extracted = extract_theorem_statement(content)
                assert isinstance(
                    extracted, str
                ), f"Extraction should return string for {name}"
                assert (
                    len(extracted) > 0
                ), f"Extracted theorem should not be empty for {name}"
                assert extracted.startswith(
                    "theorem"
                ), f"Extracted theorem should start with 'theorem' for {name}"
            except Exception as e:
                pytest.fail(f"Theorem extraction failed for {name}: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

