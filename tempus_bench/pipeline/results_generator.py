"""Script to create pivot tables from evaluations CSV and compute aggregations."""

import pandas as pd
import argparse
import inspect
from pathlib import Path
from typing import List, Dict, Optional

import tempus_bench.aggregators as aggregators_module
from tempus_bench.aggregators.base_aggregator import BaseAggregator


class ResultsGenerator:
    """
    Class to generate pivot tables from evaluations CSV and compute aggregations.

    Automatically detects all subclasses of BaseAggregator and initializes them
    for computing aggregation scores.
    """

    def __init__(
        self,
        pivot_tables: Optional[Dict[str, pd.DataFrame]] = None,
        baseline_model: str = "seasonal_naive",
    ):
        """
        Initialize the ResultsGenerator.

        Args:
            pivot_tables: Optional dictionary mapping metric names to pivot tables.
                         Each pivot table has models as index, tasks as columns, scores as values.
                         If None, must be set later using set_pivot_tables.
            baseline_model: Name of the baseline model for aggregators that need it (default: seasonal_naive)
        """
        self.baseline_model = baseline_model
        self.aggregator_classes: List[type] = []
        self.pivot_tables: Dict[str, pd.DataFrame] = {}

        # Detect aggregator classes
        self.aggregator_classes = self._get_aggregator_subclasses()

        if pivot_tables is not None:
            self.set_pivot_tables(pivot_tables)

    def set_pivot_tables(self, pivot_tables: Dict[str, pd.DataFrame]):
        """
        Set the pivot tables dictionary.

        Args:
            pivot_tables: Dictionary mapping metric names to pivot tables
        """
        self.pivot_tables = pivot_tables

    @staticmethod
    def _get_aggregator_subclasses() -> List[type]:
        """
        Detect all subclasses of BaseAggregator in the aggregators module.

        Returns:
            List of aggregator class types (excluding BaseAggregator itself)
        """
        aggregator_classes = []

        # Get all members of the aggregators module
        for name, obj in inspect.getmembers(aggregators_module):
            if (
                inspect.isclass(obj)
                and issubclass(obj, BaseAggregator)
                and obj is not BaseAggregator
            ):
                aggregator_classes.append(obj)

        return aggregator_classes

    def _create_aggregator_instances(
        self, pivot_table: pd.DataFrame
    ) -> List[BaseAggregator]:
        """
        Create aggregator instances for a given pivot table.

        Args:
            pivot_table: DataFrame with models as index, tasks as columns, scores as values

        Returns:
            List of aggregator instances
        """
        aggregators = []

        for agg_class in self.aggregator_classes:
            try:
                # Check if the aggregator needs baseline_model
                sig = inspect.signature(agg_class.__init__)
                params = sig.parameters

                # Check if baseline_model is in the parameters
                if "baseline_model" in params:
                    # Initialize with baseline_model
                    aggregator = agg_class(
                        pivot_table, baseline_model=self.baseline_model
                    )
                else:
                    # Initialize without baseline_model
                    aggregator = agg_class(pivot_table)

                aggregators.append(aggregator)
            except (ValueError, Exception) as e:
                # Skip aggregators that fail to initialize (e.g., baseline model not found)
                # This is expected for metrics where the baseline model doesn't have results
                if "not found in pivot table" in str(e):
                    # Silently skip - this is expected behavior
                    pass
                else:
                    # Print warning for unexpected errors
                    print(f"Warning: Failed to initialize {agg_class.__name__}: {e}")
                continue

        return aggregators

    @staticmethod
    def create_pivot_tables(csv_path: str) -> Dict[str, pd.DataFrame]:
        """
        Read evaluations CSV, drop best_params column, and create pivot tables
        for each metric column.

        Args:
            csv_path: Path to the evaluations CSV file

        Returns:
            Dictionary mapping metric names to pivot tables
        """
        # Read the CSV
        df = pd.read_csv(csv_path)

        # Drop the best_params column if it exists
        if "best_params" in df.columns:
            df = df.drop(columns=["best_params"])

        # Get all columns except model_name and task_name
        metric_columns = [
            col for col in df.columns if col not in ["model_name", "task_name"]
        ]

        # Create pivot tables for each metric
        pivot_tables = {}

        for metric in metric_columns:
            # Create pivot table with model_name as index and task_name as columns
            pivot_table = df.pivot_table(
                index="model_name", columns="task_name", values=metric
            )
            pivot_tables[metric] = pivot_table

        return pivot_tables

    def compute_aggregations(
        self, metric_name: Optional[str] = None
    ) -> Dict[str, Dict[str, pd.Series]]:
        """
        Compute all aggregation scores for pivot tables.

        Args:
            metric_name: Optional metric name to compute aggregations for.
                        If None, computes for all pivot tables.

        Returns:
            Dictionary mapping metric names to dictionaries of aggregator results.
            Format: {metric_name: {aggregator_name: Series}}
        """
        if not self.pivot_tables:
            raise ValueError(
                "No pivot tables set. Use set_pivot_tables or initialize with pivot_tables."
            )

        # Determine which metrics to process
        metrics_to_process = (
            [metric_name] if metric_name is not None else list(self.pivot_tables.keys())
        )

        results = {}
        for metric in metrics_to_process:
            if metric not in self.pivot_tables:
                print(f"Warning: Metric '{metric}' not found in pivot tables")
                continue

            pivot_table = self.pivot_tables[metric]
            aggregators = self._create_aggregator_instances(pivot_table)

            metric_results = {}
            for aggregator in aggregators:
                aggregator_name = aggregator.__class__.__name__
                try:
                    # Call the aggregator's __call__ method
                    if callable(aggregator):
                        metric_results[aggregator_name] = aggregator()
                    else:
                        print(f"Warning: {aggregator_name} is not callable")
                        continue
                except Exception as e:
                    print(f"Warning: Failed to compute {aggregator_name}: {e}")
                    continue

            results[metric] = metric_results

        return results

    def save_pivot_tables(self, output_dir: str):
        """
        Save all pivot tables to CSV files.

        Args:
            output_dir: Directory to save pivot tables
        """
        if not self.pivot_tables:
            raise ValueError("No pivot tables to save. Set pivot_tables first.")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for metric_name, pivot_table in self.pivot_tables.items():
            file_path = output_path / f"{metric_name}_pivot.csv"
            pivot_table.to_csv(file_path)
            print(f"Saved pivot table for {metric_name} to {file_path}")

    def save_aggregations(self, output_dir: str):
        """
        Compute aggregations for all pivot tables and save results to CSV files.

        Args:
            output_dir: Directory to save aggregation results
        """
        if not self.aggregator_classes:
            raise ValueError("No aggregator classes detected.")

        # Compute aggregations for all metrics
        aggregations = self.compute_aggregations()

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save each aggregator's results for each metric
        for metric_name, metric_aggregations in aggregations.items():
            for aggregator_name, series in metric_aggregations.items():
                file_path = output_path / f"{metric_name}_{aggregator_name.lower()}.csv"
                # Convert Series to DataFrame for easier saving
                df = series.to_frame()
                df.index.name = "model_name"
                df.to_csv(file_path)
                print(
                    f"Saved {aggregator_name} results for {metric_name} to {file_path}"
                )


def main():
    """Main function to run the pivot table generator from command line."""
    parser = argparse.ArgumentParser(
        description="Create pivot tables from evaluations CSV and compute aggregations"
    )
    parser.add_argument("csv_path", type=str, help="Path to the evaluations CSV file")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results_summary",
        help="Directory to save pivot tables and aggregations. Default is 'results_summary'.",
    )

    args = parser.parse_args()

    # Create pivot tables from CSV
    pivot_tables = ResultsGenerator.create_pivot_tables(args.csv_path)

    # Create ResultsGenerator with the pivot tables
    generator = ResultsGenerator(pivot_tables=pivot_tables)

    # Save pivot tables
    generator.save_pivot_tables(args.output_dir)

    # Save aggregations
    generator.save_aggregations(args.output_dir)


if __name__ == "__main__":
    main()
