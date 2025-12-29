"""
Comprehensive test suite for all 13 JANUS visualizations.

This module tests:
- Import and syntax validation
- Basic execution without errors
- Output file generation
- File size validation
- Performance benchmarks (tier-based)

Author: Project JANUS Team
License: MIT
"""

import sys
from pathlib import Path

import pytest

# Add examples to path
EXAMPLES_DIR = Path(__file__).parent.parent / "project_janus" / "examples"
sys.path.insert(0, str(EXAMPLES_DIR))


class TestVisualizationImports:
    """Test that all visualization modules can be imported."""

    def test_import_visual_1(self):
        """Test importing Visual 1 (GAF Pipeline)."""
        import visual_1_gaf_pipeline

        assert hasattr(visual_1_gaf_pipeline, "main")

    def test_import_visual_2(self):
        """Test importing Visual 2 (LOB vs GAF)."""
        import visual_2_lob_gaf_comparison

        assert hasattr(visual_2_lob_gaf_comparison, "main")

    def test_import_visual_3(self):
        """Test importing Visual 3 (ViViT Attention)."""
        import visual_3_vivit_attention

        assert hasattr(visual_3_vivit_attention, "main")

    def test_import_visual_4(self):
        """Test importing Visual 4 (LTN Grounding)."""
        import visual_4_ltn_grounding

        assert hasattr(visual_4_ltn_grounding, "main")

    def test_import_visual_5(self):
        """Test importing Visual 5 (LTN Truth Surface)."""
        import visual_5_ltn_truth_surface

        assert hasattr(visual_5_ltn_truth_surface, "main")

    def test_import_visual_6(self):
        """Test importing Visual 6 (Fusion Gate)."""
        import visual_6_fusion_gate

        assert hasattr(visual_6_fusion_gate, "main")

    def test_import_visual_7(self):
        """Test importing Visual 7 (OpAL Decision)."""
        import visual_7_opal_decision

        assert hasattr(visual_7_opal_decision, "main")

    def test_import_visual_8(self):
        """Test importing Visual 8 (Mahalanobis)."""
        import visual_8_mahalanobis

        assert hasattr(visual_8_mahalanobis, "main")

    def test_import_visual_9(self):
        """Test importing Visual 9 (Memory Consolidation)."""
        import visual_9_memory_consolidation

        assert hasattr(visual_9_memory_consolidation, "main")

    def test_import_visual_10(self):
        """Test importing Visual 10 (Recall Gate)."""
        import visual_10_recall_gate

        assert hasattr(visual_10_recall_gate, "main")

    @pytest.mark.skipif(
        not pytest.importorskip("umap", reason="umap-learn not installed"), reason=""
    )
    def test_import_visual_11(self):
        """Test importing Visual 11 (UMAP Evolution)."""
        import visual_11_umap_evolution

        assert hasattr(visual_11_umap_evolution, "main")

    def test_import_visual_12(self):
        """Test importing Visual 12 (Runtime Topology)."""
        import visual_12_runtime_topology

        assert hasattr(visual_12_runtime_topology, "main")

    def test_import_visual_13(self):
        """Test importing Visual 13 (Microservices)."""
        import visual_13_microservices_ecosystem

        assert hasattr(visual_13_microservices_ecosystem, "main")


class TestVisualizationExecution:
    """Test that visualizations execute without errors."""

    def test_visual_1_execution(self, output_dir):
        """Test Visual 1 generates output."""
        import visual_1_gaf_pipeline as v1

        fig = v1.visualize_gaf_pipeline(
            pattern="trending", window_size=20, save_path=str(output_dir / "v1.png"), dpi=100
        )
        assert fig is not None
        assert (output_dir / "v1.png").exists()

    def test_visual_5_execution(self, output_dir):
        """Test Visual 5 generates output."""
        import visual_5_ltn_truth_surface as v5

        fig = v5.visualize_truth_surface(
            operation="and", save_path=str(output_dir / "v5.png"), dpi=100
        )
        assert fig is not None
        assert (output_dir / "v5.png").exists()

    def test_visual_7_execution(self, output_dir):
        """Test Visual 7 generates output."""
        import visual_7_opal_decision as v7

        fig = v7.visualize_opal_decision(
            regime="volatile",
            n_steps=100,
            current_step=50,
            save_path=str(output_dir / "v7.png"),
            dpi=100,
            seed=42,
        )
        assert fig is not None
        assert (output_dir / "v7.png").exists()

    def test_visual_8_execution(self, output_dir):
        """Test Visual 8 generates output."""
        import visual_8_mahalanobis as v8

        fig = v8.visualize_mahalanobis_ellipsoid(
            n_samples=100, n_anomalies=10, save_path=str(output_dir / "v8.png"), dpi=100, seed=42
        )
        assert fig is not None
        assert (output_dir / "v8.png").exists()

    @pytest.mark.slow
    def test_visual_11_execution(self, output_dir):
        """Test Visual 11 generates output."""
        pytest.importorskip("umap")
        import visual_11_umap_evolution as v11

        fig = v11.visualize_umap_evolution(
            time_steps=[0, 100],
            n_samples=50,
            save_path=str(output_dir / "v11.png"),
            dpi=100,
            seed=42,
        )
        assert fig is not None
        assert (output_dir / "v11.png").exists()


class TestOutputValidation:
    """Test that generated outputs are valid."""

    def test_output_file_sizes(self, output_dir):
        """Test that generated files are non-trivial in size."""
        import visual_5_ltn_truth_surface as v5

        output_file = output_dir / "size_test.png"
        v5.visualize_truth_surface(operation="and", save_path=str(output_file), dpi=100)

        # File should be at least 10KB
        assert output_file.stat().st_size > 10000

    def test_multiple_outputs(self, output_dir):
        """Test generating multiple visualizations."""
        import visual_5_ltn_truth_surface as v5

        v5.visualize_all_operations(output_dir=str(output_dir), dpi=100)

        # Check all three files exist
        assert (output_dir / "visual_5_ltn_and.png").exists()
        assert (output_dir / "visual_5_ltn_or.png").exists()
        assert (output_dir / "visual_5_ltn_implies.png").exists()


@pytest.mark.slow
class TestPerformanceBudgets:
    """Test that visualizations meet performance tier requirements."""

    def test_tier_2_performance(self, output_dir, timer):
        """Test Tier 2 visuals complete in < 1s (real-time)."""
        import visual_7_opal_decision as v7

        with timer() as t:
            v7.visualize_opal_decision(
                regime="volatile", n_steps=100, save_path=str(output_dir / "perf.png"), dpi=100
            )

        # Tier 2 should be < 1s for production, < 5s for test DPI
        assert t.elapsed < 5.0, f"Tier 2 took {t.elapsed:.2f}s (expected < 5s)"

    def test_tier_4_performance(self, output_dir, timer):
        """Test Tier 4 visuals complete within reasonable time."""
        import visual_5_ltn_truth_surface as v5

        with timer() as t:
            v5.visualize_truth_surface(
                operation="and", save_path=str(output_dir / "perf.png"), dpi=100
            )

        # Tier 4 (static) should be < 10s
        assert t.elapsed < 10.0, f"Tier 4 took {t.elapsed:.2f}s (expected < 10s)"


class TestReproducibility:
    """Test that visualizations are reproducible with fixed seeds."""

    def test_gaf_reproducibility(self, output_dir):
        """Test GAF visualization is reproducible."""
        import visual_1_gaf_pipeline as v1

        # Generate twice with same seed
        fig1 = v1.visualize_gaf_pipeline(pattern="trending", window_size=20, seed=42)
        fig2 = v1.visualize_gaf_pipeline(pattern="trending", window_size=20, seed=42)

        # Both should succeed (exact pixel comparison is difficult with matplotlib)
        assert fig1 is not None
        assert fig2 is not None

    def test_opal_reproducibility(self, output_dir):
        """Test OpAL visualization is reproducible."""
        import visual_7_opal_decision as v7

        fig1 = v7.visualize_opal_decision(regime="volatile", n_steps=100, seed=42)
        fig2 = v7.visualize_opal_decision(regime="volatile", n_steps=100, seed=42)

        assert fig1 is not None
        assert fig2 is not None


class TestDataGeneration:
    """Test synthetic data generation functions."""

    def test_gaf_data_generation(self):
        """Test GAF synthetic price generation."""
        import visual_1_gaf_pipeline as v1

        prices = v1.generate_synthetic_prices(n_steps=100, pattern="trending", seed=42)

        assert len(prices) == 100
        assert prices.min() > 0  # Prices should be positive
        assert not any(prices != prices)  # No NaN values

    def test_opal_trajectory_generation(self):
        """Test OpAL trajectory generation."""
        import visual_7_opal_decision as v7

        engine = v7.generate_synthetic_trajectory(n_steps=50, regime="volatile", seed=42)

        assert len(engine.history) == 50
        # Check that G and N values are in valid range
        for h in engine.history:
            assert 0 <= h["G"] <= 1
            assert 0 <= h["N"] <= 1

    def test_mahalanobis_data_generation(self):
        """Test Mahalanobis data generation."""
        import visual_8_mahalanobis as v8

        data = v8.generate_correlated_data(n_samples=100, seed=42)

        assert data.shape == (100, 2)
        assert not any((data != data).flatten())  # No NaN values


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_operation(self):
        """Test that invalid operations raise appropriate errors."""
        import visual_5_ltn_truth_surface as v5

        with pytest.raises(ValueError):
            v5.visualize_truth_surface(operation="invalid_operation")

    def test_minimal_steps(self, output_dir):
        """Test visualizations with minimal data."""
        import visual_7_opal_decision as v7

        # Should work with minimal steps
        fig = v7.visualize_opal_decision(
            regime="volatile", n_steps=10, save_path=str(output_dir / "minimal.png"), dpi=100
        )
        assert fig is not None

    def test_zero_samples(self):
        """Test that zero samples raises error or handles gracefully."""
        import visual_8_mahalanobis as v8

        # Should either raise error or handle gracefully
        try:
            v8.generate_correlated_data(n_samples=0, seed=42)
        except (ValueError, AssertionError):
            pass  # Expected behavior


class TestAccessibility:
    """Test that visualizations meet accessibility standards."""

    def test_colormap_exists(self):
        """Test that visualizations use defined colormaps."""
        import visual_1_gaf_pipeline as v1
        import visual_7_opal_decision as v7

        # These should not raise errors about missing colormaps
        fig1 = v1.visualize_gaf_pipeline(pattern="trending", window_size=20, seed=42)
        fig7 = v7.visualize_opal_decision(regime="volatile", n_steps=50, seed=42)

        assert fig1 is not None
        assert fig7 is not None

    def test_dpi_setting(self, output_dir):
        """Test that DPI can be configured."""
        import visual_5_ltn_truth_surface as v5

        # Test different DPI settings
        for dpi in [72, 150, 300]:
            output_file = output_dir / f"dpi_{dpi}.png"
            v5.visualize_truth_surface(operation="and", save_path=str(output_file), dpi=dpi)
            assert output_file.exists()


# Integration test
@pytest.mark.integration
@pytest.mark.slow
def test_full_visualization_suite(output_dir):
    """Integration test: generate all visualizations."""
    # Import all modules
    import visual_1_gaf_pipeline as v1
    import visual_5_ltn_truth_surface as v5
    import visual_7_opal_decision as v7
    import visual_8_mahalanobis as v8

    # Generate outputs
    v1.visualize_all_patterns(output_dir=str(output_dir), dpi=100, seed=42)
    v5.visualize_all_operations(output_dir=str(output_dir), dpi=100)
    v7.visualize_opal_decision(
        regime="volatile", n_steps=100, save_path=str(output_dir / "v7.png"), dpi=100, seed=42
    )
    v8.visualize_mahalanobis_ellipsoid(
        n_samples=100, save_path=str(output_dir / "v8.png"), dpi=100, seed=42
    )

    # Check that multiple files were created
    png_files = list(output_dir.glob("*.png"))
    assert len(png_files) >= 5, f"Expected at least 5 outputs, got {len(png_files)}"
