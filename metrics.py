from prometheus_client import Counter, Histogram, Gauge, Info, start_http_server

# --- 1. ROUTING DECISIONS & OVERRIDES ---
ROUTING_DECISIONS_TOTAL = Counter(
    "router_decisions_total",
    "Total routing decisions made by the router policy",
    ["selected_engine", "override_reason", "status"]
)

# --- 2. POLICY CONFIDENCE DISTRIBUTION ---
ROUTER_POLICY_CONFIDENCE = Histogram(
    "router_policy_confidence_score",
    "Probability score assigned by the policy network to the chosen engine",
    ["selected_engine"],
    buckets=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99, 1.0]
)

# --- 3. END-TO-END INFERENCE LATENCY ---
ROUTER_LATENCY_SECONDS = Histogram(
    "router_inference_latency_seconds",
    "Latencies incurred across engines and policy evaluation",
    ["stage", "selected_engine"],  # stage: policy_eval, target_inference
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

# --- 4. FINANCIAL COST ACCUMULATION ---
ROUTER_ESTIMATED_COST_USD = Counter(
    "router_estimated_cost_usd_total",
    "Cumulative estimated financial cost accrued by executing target engines",
    ["selected_engine"]
)

# --- 5. BACKEND TELEMETRY & CIRCUIT BREAKER HEALTH ---
BACKEND_HEALTH_GAUGE = Gauge(
    "router_backend_engine_health",
    "Health status of engine backends (1 = Healthy, 0 = Offline / Circuit Broken)",
    ["engine"]
)

BACKEND_LATENCY_GAUGE = Gauge(
    "router_backend_reported_latency_ms",
    "Real-time latency reported by backend health telemetry",
    ["engine"]
)



def start_metrics_server(port=8000):
    """Binds the Prometheus metrics exporter to the specified port."""
    start_http_server(port)

def record_routing_decision(engine, confidence, override_reason, policy_latency_sec, target_latency_sec, cost_usd):
    """Records telemetry for a single routing execution lifecycle."""
    # 1. Track Routing Throughput & Overrides
    ROUTING_DECISIONS_TOTAL.labels(
        selected_engine=engine, 
        override_reason=override_reason, 
        status="success"
    ).inc()
    
    # 2. Track Neural Policy Confidence (Ignore for deterministic overrides)
    if override_reason == "none":
        ROUTER_POLICY_CONFIDENCE.labels(selected_engine=engine).observe(confidence)
    
    # 3. Track Sub-Stage Latency
    ROUTER_LATENCY_SECONDS.labels(stage="policy_eval", selected_engine=engine).observe(policy_latency_sec)
    ROUTER_LATENCY_SECONDS.labels(stage="target_inference", selected_engine=engine).observe(target_latency_sec)
    
    # 4. Accumulate Financial Cost
    ROUTER_ESTIMATED_COST_USD.labels(selected_engine=engine).inc(cost_usd)