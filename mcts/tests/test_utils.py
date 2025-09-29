import pytest

from mcts.utils import extract_imports, extract_theorem_statement


class TestExtractTheoremStatement:
    """Test cases for theorem statement extraction from Isabelle theory files"""

    def test_extract_named_theorem_multiline(self):
        """Test extraction of named theorem with multiline statement ending with sorry"""
        content = r"""(*
  Authors: Wenda Li
*)

theory mathd_numbertheory_12 imports
  Complex_Main
  "HOL-Computational_Algebra.Computational_Algebra"
begin

theorem mathd_numbertheory_12 :
  "card {x::nat. 20 dvd x \<and> 15 \<le> x \<and> x < 86} =4"
  sorry

end
"""
        result = extract_theorem_statement(content)
        expected = r'''theorem mathd_numbertheory_12 :
  "card {x::nat. 20 dvd x \<and> 15 \<le> x \<and> x < 86} =4"'''
        assert result == expected

    def test_extract_theorem_ending_with_by(self):
        """Test extraction of theorem ending with 'by'"""
        content = r"""theory test imports Main begin
theorem test_thm: "1 + 1 = 2" by simp
end"""
        result = extract_theorem_statement(content)
        expected = r'theorem test_thm: "1 + 1 = 2"'
        assert result == expected

    def test_extract_theorem_ending_with_proof(self):
        """Test extraction of theorem ending with 'proof'"""
        content = r"""theory test imports Main begin
theorem test_thm: "1 + 1 = 2"
proof
  show ?thesis by simp
qed
end"""
        result = extract_theorem_statement(content)
        expected = r'theorem test_thm: "1 + 1 = 2"'
        assert result == expected

    def test_extract_unnamed_theorem(self):
        """Test extraction of unnamed theorem"""
        content = r"""theory test imports Main begin
theorem "1 + 1 = 2" sorry
end"""
        result = extract_theorem_statement(content)
        expected = r'theorem "1 + 1 = 2"'
        assert result == expected

    def test_extract_theorem_with_using(self):
        """Test extraction of theorem ending with 'using'"""
        content = r"""theory test imports Main begin
theorem test_thm: "P \<and> Q \<longrightarrow> P" using assms by blast
end"""
        result = extract_theorem_statement(content)
        expected = r'theorem test_thm: "P \<and> Q \<longrightarrow> P"'
        assert result == expected

    def test_extract_theorem_complex_statement(self):
        """Test extraction of theorem with complex statement and fixes"""
        content = r"""theory mathd_numbertheory_709 imports
  HOL-Computational_Algebra.Computational_Algebra
begin

theorem mathd_numbertheory_709:
  fixes n :: nat
  assumes "n>0"
    and "card ({k. k dvd (2*n)}) = 2*n"
  shows "n=1"
  sorry

end"""
        result = extract_theorem_statement(content)
        expected = r'''theorem mathd_numbertheory_709:
  fixes n :: nat
  assumes "n>0"
    and "card ({k. k dvd (2*n)}) = 2*n"
  shows "n=1"'''
        assert result == expected

    def test_extract_theorem_minimal(self):
        """Test extraction of minimal theorem"""
        content = r"""theory test imports Main begin
theorem simple: "True" sorry
end"""
        result = extract_theorem_statement(content)
        expected = r'theorem simple: "True"'
        assert result == expected

    def test_extract_theorem_fallback(self):
        """Test fallback extraction when no ending keyword found"""
        content = r"""theory test imports Main begin
theorem fallback_test: "P \<or> \<not>P"
end"""
        result = extract_theorem_statement(content)
        expected = r'theorem fallback_test: "P \<or> \<not>P"'
        assert result == expected

    def test_extract_theorem_no_theorem_raises_error(self):
        """Test that function raises AssertionError when no theorem found"""
        content = """theory test imports Main begin
(* No theorem here *)
end"""
        with pytest.raises(AssertionError, match="No theorem found in content"):
            extract_theorem_statement(content)


class TestExtractImports:
    """Test cases for import extraction from Isabelle theory files"""

    def test_extract_single_import(self):
        """Test extraction of single import"""
        content = """theory test imports Main begin
theorem test: "True" sorry
end"""
        result = extract_imports(content)
        assert result == ["Main"]

    def test_extract_multiple_imports(self):
        """Test extraction of multiple imports"""
        content = """theory mathd_numbertheory_12 imports
  Complex_Main
  "HOL-Computational_Algebra.Computational_Algebra"
begin
theorem test: "True" sorry
end"""
        result = extract_imports(content)
        expected = ["Complex_Main", "HOL-Computational_Algebra.Computational_Algebra"]
        assert result == expected

    def test_extract_imports_with_quotes(self):
        """Test extraction of imports with quotes"""
        content = """theory test imports
  Main
  "HOL-Analysis.Analysis"
begin
theorem test: "True" sorry
end"""
        result = extract_imports(content)
        expected = ["Main", "HOL-Analysis.Analysis"]
        assert result == expected

    def test_extract_imports_multiline(self):
        """Test extraction of multiline imports"""
        content = """theory test imports
  HOL-Algebra.Ring
  HOL-Algebra.Group
  HOL-Algebra.Field
begin
theorem test: "True" sorry
end"""
        result = extract_imports(content)
        expected = ["HOL-Algebra.Ring", "HOL-Algebra.Group", "HOL-Algebra.Field"]
        assert result == expected

    def test_extract_imports_no_imports(self):
        """Test extraction when no imports found returns Main"""
        content = """theory test begin
theorem test: "True" sorry
end"""
        result = extract_imports(content)
        assert result == ["Main"]

    def test_extract_imports_empty_imports(self):
        """Test extraction with empty imports section"""
        content = """theory test imports
begin
theorem test: "True" sorry
end"""
        result = extract_imports(content)
        assert result == ["Main"]


class TestUtilsIntegration:
    """Integration tests for utils functions working together"""

    def test_extract_theorem_and_imports_integration(self):
        """Test that theorem and imports extraction work together on real content"""
        content = r"""(*
  Authors: Wenda Li
*)

theory mathd_numbertheory_12 imports
  Complex_Main
  "HOL-Computational_Algebra.Computational_Algebra"
begin

theorem mathd_numbertheory_12 :
  "card {x::nat. 20 dvd x \<and> 15 \<le> x \<and> x < 86} =4"
  sorry

end
"""
        theorem = extract_theorem_statement(content)
        imports = extract_imports(content)

        assert (
            theorem
            == r'''theorem mathd_numbertheory_12 :
  "card {x::nat. 20 dvd x \<and> 15 \<le> x \<and> x < 86} =4"'''
        )
        assert imports == [
            "Complex_Main",
            "HOL-Computational_Algebra.Computational_Algebra",
        ]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

