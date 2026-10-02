import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, Tuple

# Define target model engines
ENGINES = ["llama_core", "grok_core", "gemini_core"]

@dataclass
class QueryContext:
    prompt_embedding: torch.Tensor  # e.g., 768-dim semantic representation
    has_image: bool                 # Multimodal indicator for Gemini
    has_audio: bool                 # Multimodal indicator for Gemini
    requires_realtime_web: bool     # Freshness indicator for Grok
    max_acceptable_latency_ms: float
    client_budget_usd_per_1k: float

@dataclass
class BackendMetrics:
    latency_ms: Dict[str, float]
    availability: Dict[str, float]  # 1.0 = operational, 0.0 = offline / circuit broken
    cost_per_1k_tokens: Dict[str, float]

class RouterPolicyNetwork(nn.Module):
    """
    Heterogeneous Multi-Head Policy Network that routes requests to Llama, Grok, 
    or Gemini based on semantic embeddings, context flags, and backend state.
    """
    def __init__(self, embed_dim: int = 768, hidden_dim: int = 256, num_engines: int = 3):
        super(RouterPolicyNetwork, self).__init__()
        
        # 1. Semantic Query Encoder
        self.query_encoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 2. Context & Metric Features MLP
        # Features: [has_image, has_audio, requires_realtime_web, max_latency, client_budget,
        #            llama_lat, grok_lat, gemini_lat, llama_cost, grok_cost, gemini_cost,
        #            llama_avail, grok_avail, gemini_avail] -> 14 features
        self.context_encoder = nn.Sequential(
            nn.Linear(14, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # 3. Policy Head (Engine Softmax Probabilities)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim + 64, 128),
            nn.ReLU(),
            nn.Linear(128, num_engines)
        )

    def forward(self, query_embed: torch.Tensor, context_features: torch.Tensor) -> torch.Tensor:
        h_query = self.query_encoder(query_embed)
        h_ctx = self.context_encoder(context_features)
        
        # Fuse semantic representations with context/metric telemetry
        fused = torch.cat([h_query, h_ctx], dim=-1)
        logits = self.policy_head(fused)
        
        return F.softmax(logits, dim=-1)


class EngineRouter:
    """
    Execution router wrapping the policy network with hard rules, circuit breakers,
    and confidence fallback logic.
    """
    def __init__(self, policy_net: RouterPolicyNetwork):
        self.policy_net = policy_net
        self.policy_net.eval()

    def _build_context_vector(self, ctx: QueryContext, metrics: BackendMetrics) -> torch.Tensor:
        vector = [
            float(ctx.has_image),
            float(ctx.has_audio),
            float(ctx.requires_realtime_web),
            ctx.max_acceptable_latency_ms / 1000.0,  # Normalize
            ctx.client_budget_usd_per_1k,
            metrics.latency_ms["llama_core"] / 1000.0,
            metrics.latency_ms["grok_core"] / 1000.0,
            metrics.latency_ms["gemini_core"] / 1000.0,
            metrics.cost_per_1k_tokens["llama_core"],
            metrics.cost_per_1k_tokens["grok_core"],
            metrics.cost_per_1k_tokens["gemini_core"],
            metrics.availability["llama_core"],
            metrics.availability["grok_core"],
            metrics.availability["gemini_core"],
        ]
        return torch.tensor(vector, dtype=torch.float32).unsqueeze(0)

    def route_query(self, ctx: QueryContext, metrics: BackendMetrics) -> Tuple[str, Dict[str, float]]:
        # --- Rule-Based Circuit Breaker Overrides ---
        # 1. Hard Rule: Multimodal context forces Gemini
        if ctx.has_image or ctx.has_audio:
            if metrics.availability["gemini_core"] > 0:
                return "gemini_core", {"override": "multimodal_requirement", "confidence": 1.0} # pyright: ignore[reportReturnType]
        
        # 2. Hard Rule: Explicit live search forces Grok
        if ctx.requires_realtime_web:
            if metrics.availability["grok_core"] > 0:
                return "grok_core", {"override": "realtime_web_search", "confidence": 1.0} # type: ignore

        # --- Neural Policy Routing ---
        ctx_vector = self._build_context_vector(ctx, metrics)
        
        with torch.no_grad():
            probs = self.policy_net(ctx.prompt_embedding.unsqueeze(0), ctx_vector).squeeze(0)
        
        prob_dict = {ENGINES[i]: probs[i].item() for i in range(len(ENGINES))}
        
        # Mask out unavailable engines
        for engine in ENGINES:
            if metrics.availability[engine] <= 0:
                prob_dict[engine] = 0.0

        # Renormalize or fallback to Llama Core
        total_prob = sum(prob_dict.values())
        if total_prob == 0:
            selected_engine = "llama_core"  # Default on-prem safety fallback
        else:
            selected_engine = max(prob_dict, key=prob_dict.get) # pyright: ignore[reportArgumentType, reportCallIssue]

        return selected_engine, prob_dict


# --- Verification & Simulation Run ---
if __name__ == "__main__":
    # Initialize Policy Model
    policy_nn = RouterPolicyNetwork(embed_dim=768)
    router = EngineRouter(policy_nn)

    # Mock Backend Telemetry Metrics
    current_metrics = BackendMetrics(
        latency_ms={"llama_core": 80.0, "grok_core": 180.0, "gemini_core": 150.0},
        availability={"llama_core": 1.0, "grok_core": 1.0, "gemini_core": 1.0},
        cost_per_1k_tokens={"llama_core": 0.0002, "grok_core": 0.002, "gemini_core": 0.0025}
    )

    # Example 1: High-Speed Text Logic (Routes via Neural Policy / Llama Core)
    query_1 = QueryContext(
        prompt_embedding=torch.randn(768),
        has_image=False,
        has_audio=False,
        requires_realtime_web=False,
        max_acceptable_latency_ms=200.0,
        client_budget_usd_per_1k=0.001
    )
    engine_1, probs_1 = router.route_query(query_1, current_metrics)
    print(f"Query 1 Target Engine -> {engine_1} | Probabilities: {probs_1}")

    # Example 2: Real-time News / Twitter Stream (Triggers Rule Override -> Grok Core)
    query_2 = QueryContext(
        prompt_embedding=torch.randn(768),
        has_image=False,
        has_audio=False,
        requires_realtime_web=True,
        max_acceptable_latency_ms=500.0,
        client_budget_usd_per_1k=0.005
    )
    engine_2, probs_2 = router.route_query(query_2, current_metrics)
    print(f"Query 2 Target Engine -> {engine_2} | Decision: {probs_2}")