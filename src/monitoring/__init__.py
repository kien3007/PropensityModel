"""
Monitoring, drift detection, ground-truth feedback and observability.

Modules are imported directly (e.g. src.monitoring.metrics) so that the serving API does not
pull in the training stack just to expose metrics.
"""
