"""
Test suite for Visual 5: Łukasiewicz Truth Surface Visualization

Tests the LTN logic operations and 3D surface visualizations.

Author: Project JANUS Team
License: MIT
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Import the visual module
sys.path.insert(0, str(Path(__file__).parent.parent / "project_janus" / "examples"))
import visual_5_ltn_truth_surface as v5


class TestLukasiewiczOperations:
    """Test the mathematical correctness of Łukasiewicz logic operations."""

    def test_lukasiewicz_and_boundary_cases(self):
        """Test AND operation at boundary values."""
        # Test (0, 0) -> 0
        assert v5.lukasiewicz_and(0.0, 0.0) == 0.0

        # Test (1, 1) -> 1
        assert v5.lukasiewicz_and(1.0, 1.0) == 1.0

        # Test (0, 1) -> 0
        assert v5.lukasiewicz_and(0.0, 1.0) == 0.0

        # Test (1, 0) -> 0
        assert v5.lukasiewicz_and(1.0, 0.0) == 0.0

    def test_lukasiewicz_and_intermediate(self):
        """Test AND operation at intermediate values."""
        # Test (0.5, 0.5) -> 0
        assert v5.lukasiewicz_and(0.5, 0.5) == 0.0

        # Test (0.7, 0.8) -> 0.5
        result = v5.lukasiewicz_and(0.7, 0.8)
        assert np.isclose(result, 0.5)

    def test_lukasiewicz_and_commutativity(self):
        """Test that AND is commutative: p ∧ q = q ∧ p."""
        p, q = 0.6, 0.7
        assert v5.lukasiewicz_and(p, q) == v5.lukasiewicz_and(q, p)

    def test_lukasiewicz_and_array_input(self):
        """Test AND with NumPy array inputs."""
        p = np.array([0.0, 0.5, 1.0])
        q = np.array([0.5, 0.5, 0.5])
        result = v5.lukasiewicz_and(p, q)

        expected = np.array([0.0, 0.0, 0.5])
        np.testing.assert_array_almost_equal(result, expected)

    def test_lukasiewicz_or_boundary_cases(self):
        """Test OR operation at boundary values."""
        # Test (0, 0) -> 0
        assert v5.lukasiewicz_or(0.0, 0.0) == 0.0

        # Test (1, 1) -> 1
        assert v5.lukasiewicz_or(1.0, 1.0) == 1.0

        # Test (0, 1) -> 1
        assert v5.lukasiewicz_or(0.0, 1.0) == 1.0

        # Test (1, 0) -> 1
        assert v5.lukasiewicz_or(1.0, 0.0) == 1.0

    def test_lukasiewicz_or_intermediate(self):
        """Test OR operation at intermediate values."""
        # Test (0.5, 0.5) -> 1
        assert v5.lukasiewicz_or(0.5, 0.5) == 1.0

        # Test (0.3, 0.4) -> 0.7
        result = v5.lukasiewicz_or(0.3, 0.4)
        assert np.isclose(result, 0.7)

    def test_lukasiewicz_or_commutativity(self):
        """Test that OR is commutative: p ∨ q = q ∨ p."""
        p, q = 0.6, 0.7
        assert v5.lukasiewicz_or(p, q) == v5.lukasiewicz_or(q, p)

    def test_lukasiewicz_implies_boundary_cases(self):
        """Test IMPLIES operation at boundary values."""
        # Test (0, 0) -> 1 (false implies false is true)
        assert v5.lukasiewicz_implies(0.0, 0.0) == 1.0

        # Test (1, 1) -> 1 (true implies true is true)
        assert v5.lukasiewicz_implies(1.0, 1.0) == 1.0

        # Test (1, 0) -> 0 (true implies false is false)
        assert v5.lukasiewicz_implies(1.0, 0.0) == 0.0

        # Test (0, 1) -> 1 (false implies anything is true)
        assert v5.lukasiewicz_implies(0.0, 1.0) == 1.0

    def test_lukasiewicz_not(self):
        """Test NOT operation."""
        # Test ¬0 -> 1
        assert v5.lukasiewicz_not(0.0) == 1.0

        # Test ¬1 -> 0
        assert v5.lukasiewicz_not(1.0) == 0.0

        # Test ¬0.5 -> 0.5
        assert v5.lukasiewicz_not(0.5) == 0.5

        # Test double negation: ¬¬p = p
        p = 0.7
        assert v5.lukasiewicz_not(v5.lukasiewicz_not(p)) == p

    def test_lukasiewicz_monotonicity(self):
        """Test monotonicity: if p ≤ p' and q ≤ q', then p ∧ q ≤ p' ∧ q'."""
        p, p_prime = 0.3, 0.5
        q, q_prime = 0.4, 0.6

        result1 = v5.lukasiewicz_and(p, q)
        result2 = v5.lukasiewicz_and(p_prime, q_prime)

        assert result1 <= result2


class TestBooleanOperations:
    """Test Boolean logic operations for comparison."""

    def test_boolean_and_values(self):
        """Test Boolean AND operation."""
        assert v5.boolean_and(1.0, 1.0) == 1.0
        assert v5.boolean_and(1.0, 0.0) == 0.0
        assert v5.boolean_and(0.5, 0.5) == 0.25

    def test_boolean_or_values(self):
        """Test Boolean OR operation."""
        assert v5.boolean_or(0.0, 0.0) == 0.0
        assert v5.boolean_or(1.0, 1.0) == 1.0
        assert v5.boolean_or(1.0, 0.0) == 1.0


class TestGradientComputation:
    """Test gradient computation for Łukasiewicz AND."""

    def test_gradient_at_center(self):
        """Test gradient at center point (0.5, 0.5)."""
        p, q = 0.5, 0.5
        grad_p, grad_q = v5.compute_gradient_lukasiewicz_and(p, q)

        # At this point, both gradients should be 1.0
        # because max(0, p + q - 1) is active and has slope 1 in both directions
        assert np.isclose(grad_p, 1.0, atol=0.1)
        assert np.isclose(grad_q, 1.0, atol=0.1)

    def test_gradient_symmetry(self):
        """Test that gradients are symmetric when p = q."""
        p = q = 0.7
        grad_p, grad_q = v5.compute_gradient_lukasiewicz_and(p, q)

        # Should be equal due to symmetry
        assert np.isclose(grad_p, grad_q, atol=0.01)

    def test_gradient_at_zero_region(self):
        """Test gradient in region where output is always 0."""
        p, q = 0.2, 0.3
        grad_p, grad_q = v5.compute_gradient_lukasiewicz_and(p, q)

        # In region where p + q < 1, gradient should be 0
        assert np.isclose(grad_p, 0.0, atol=0.1)
        assert np.isclose(grad_q, 0.0, atol=0.1)


class TestVisualizationGeneration:
    """Test the visualization generation functions."""

    def test_visualize_truth_surface_and(self, output_dir):
        """Test AND operation visualization generation."""
        output_file = output_dir / "truth_and.png"

        fig = v5.visualize_truth_surface(
            operation="and",
            save_path=str(output_file),
            show_gradients=True,
            dpi=100,  # Lower DPI for faster testing
        )

        assert output_file.exists()
        assert output_file.stat().st_size > 10000  # At least 10KB

        # Check that figure was created
        assert fig is not None

    def test_visualize_truth_surface_or(self, output_dir):
        """Test OR operation visualization generation."""
        output_file = output_dir / "truth_or.png"

        fig = v5.visualize_truth_surface(
            operation="or",
            save_path=str(output_file),
            show_gradients=False,
            dpi=100,
        )

        assert output_file.exists()
        assert output_file.stat().st_size > 10000

    def test_visualize_truth_surface_implies(self, output_dir):
        """Test IMPLIES operation visualization generation."""
        output_file = output_dir / "truth_implies.png"

        fig = v5.visualize_truth_surface(
            operation="implies",
            save_path=str(output_file),
            show_gradients=False,
            dpi=100,
        )

        assert output_file.exists()
        assert output_file.stat().st_size > 10000

    def test_visualize_invalid_operation(self):
        """Test that invalid operation raises ValueError."""
        with pytest.raises(ValueError, match="Unknown operation"):
            v5.visualize_truth_surface(operation="invalid")

    @pytest.mark.slow
    def test_visualize_all_operations(self, output_dir):
        """Test generating all operations at once."""
        v5.visualize_all_operations(output_dir=str(output_dir), show_gradients=True, dpi=100)

        # Check that all three files were created
        assert (output_dir / "visual_5_ltn_and.png").exists()
        assert (output_dir / "visual_5_ltn_or.png").exists()
        assert (output_dir / "visual_5_ltn_implies.png").exists()


class TestReproducibility:
    """Test that visualizations are reproducible."""

    def test_deterministic_output(self, output_dir):
        """Test that same inputs produce identical outputs."""
        # Generate first visualization
        output1 = output_dir / "truth_1.png"
        v5.visualize_truth_surface(
            operation="and", save_path=str(output1), show_gradients=False, dpi=100
        )

        # Generate second visualization with same parameters
        output2 = output_dir / "truth_2.png"
        v5.visualize_truth_surface(
            operation="and", save_path=str(output2), show_gradients=False, dpi=100
        )

        # Files should exist and have same size
        assert output1.exists() and output2.exists()
        assert output1.stat().st_size == output2.stat().st_size


class TestCommandLineInterface:
    """Test the CLI argument parsing (if running as script)."""

    def test_module_has_main(self):
        """Test that module has a main function."""
        assert hasattr(v5, "main")
        assert callable(v5.main)


# Performance benchmarks
@pytest.mark.slow
class TestPerformance:
    """Performance tests for visualization generation."""

    def test_and_visualization_performance(self, output_dir, timer):
        """Test that AND visualization completes within time budget."""
        output_file = output_dir / "perf_and.png"

        with timer() as t:
            v5.visualize_truth_surface(operation="and", save_path=str(output_file), dpi=100)

        # Should complete in < 5 seconds (Tier 4 budget)
        assert t.elapsed < 5.0, f"Took {t.elapsed:.2f}s, expected < 5s"

    def test_all_operations_performance(self, output_dir, timer):
        """Test that all operations complete within budget."""
        with timer() as t:
            v5.visualize_all_operations(output_dir=str(output_dir), dpi=100)

        # Should complete in < 15 seconds for all three
        assert t.elapsed < 15.0, f"Took {t.elapsed:.2f}s, expected < 15s"
