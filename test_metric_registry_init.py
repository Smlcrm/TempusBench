"""
Test script to initialize MetricRegistry and print its class variables.
"""

from tempus_bench.pipeline.metric_registry import MetricRegistry


def main():
    """Initialize MetricRegistry and print its attributes."""
    print("=" * 80)
    print("Initializing MetricRegistry...")
    print("=" * 80)
    
    # Initialize MetricRegistry
    registry = MetricRegistry()
    
    print("\n" + "=" * 80)
    print("MetricRegistry Class Variables:")
    print("=" * 80)
    
    # Print metric_registry dictionary
    print(f"\n1. metric_registry (Dict[str, BaseMetric]):")
    print(f"   Number of metrics registered: {len(registry.metric_registry)}")
    print(f"   Metric names: {list(registry.metric_registry.keys())}")
    
    print("\n   Details of each metric:")
    for metric_name, metric_instance in registry.metric_registry.items():
        print(f"     - {metric_name}:")
        print(f"         Type: {type(metric_instance).__name__}")
        print(f"         Metric type: {metric_instance.metric_type}")
        print(f"         Class: {type(metric_instance)}")
    
    # Print stochastic_metrics list
    print(f"\n2. stochastic_metrics (List[str]):")
    print(f"   Number of stochastic metrics: {len(registry.stochastic_metrics)}")
    print(f"   Stochastic metric names: {registry.stochastic_metrics}")
    
    # Print deterministic_metrics list
    print(f"\n3. deterministic_metrics (List[str]):")
    print(f"   Number of deterministic metrics: {len(registry.deterministic_metrics)}")
    print(f"   Deterministic metric names: {registry.deterministic_metrics}")
    
    print("\n" + "=" * 80)
    print("Summary:")
    print("=" * 80)
    print(f"Total metrics registered: {len(registry.metric_registry)}")
    print(f"Stochastic metrics: {len(registry.stochastic_metrics)}")
    print(f"Deterministic metrics: {len(registry.deterministic_metrics)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
