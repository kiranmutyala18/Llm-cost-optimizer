def _create_client(client, name="acme", rate_limit=1000):
    resp = client.post("/clients", json={"name": name, "rate_limit_per_minute": rate_limit})
    assert resp.status_code == 201, resp.text
    return resp.json()


def test_create_and_list_clients(client):
    created = _create_client(client, "acme-corp")
    assert created["name"] == "acme-corp"
    assert created["api_key"].startswith("sk-llmopt-")

    resp = client.get("/clients")
    assert resp.status_code == 200
    names = [c["name"] for c in resp.json()]
    assert "acme-corp" in names


def test_auth_rejects_missing_or_bad_key(client):
    resp = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 422  # missing required header

    resp = client.post(
        "/v1/chat/completions",
        headers={"X-API-Key": "not-a-real-key"},
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 401


def test_first_request_is_cache_miss_then_exact_hit(client):
    c = _create_client(client)
    headers = {"X-API-Key": c["api_key"]}
    body = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "What is the capital of France?"}]}

    r1 = client.post("/v1/chat/completions", headers=headers, json=body)
    assert r1.status_code == 200
    assert r1.json()["cache_status"] == "miss"
    first_content = r1.json()["content"]

    # Identical request again -> exact hit, same content, no new LLM call needed
    r2 = client.post("/v1/chat/completions", headers=headers, json=body)
    assert r2.status_code == 200
    assert r2.json()["cache_status"] == "exact_hit"
    assert r2.json()["content"] == first_content
    assert r2.json()["similarity_score"] == 1.0


def test_semantically_similar_prompt_is_a_semantic_hit(client):
    c = _create_client(client)
    headers = {"X-API-Key": c["api_key"]}

    body1 = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "explain how photosynthesis works"}]}
    body2 = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "explain how photosynthesis works please"}]}

    r1 = client.post("/v1/chat/completions", headers=headers, json=body1)
    assert r1.json()["cache_status"] == "miss"

    r2 = client.post("/v1/chat/completions", headers=headers, json=body2)
    assert r2.json()["cache_status"] in ("semantic_hit", "exact_hit")
    assert r2.json()["content"] == r1.json()["content"]


def test_different_prompts_are_both_misses(client):
    c = _create_client(client)
    headers = {"X-API-Key": c["api_key"]}

    r1 = client.post(
        "/v1/chat/completions", headers=headers,
        json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "write a haiku about the ocean"}]},
    )
    r2 = client.post(
        "/v1/chat/completions", headers=headers,
        json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "write python code to sort a list"}]},
    )
    assert r1.json()["cache_status"] == "miss"
    assert r2.json()["cache_status"] == "miss"
    assert r1.json()["content"] != r2.json()["content"]


def test_rate_limit_enforced(client):
    c = _create_client(client, "tight-limit", rate_limit=2)
    headers = {"X-API-Key": c["api_key"]}
    body = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hello"}]}

    r1 = client.post("/v1/chat/completions", headers=headers, json=body)
    r2 = client.post("/v1/chat/completions", headers=headers, json={
        "model": "gpt-4o-mini", "messages": [{"role": "user", "content": "a different unique prompt xyz"}]
    })
    r3 = client.post("/v1/chat/completions", headers=headers, json={
        "model": "gpt-4o-mini", "messages": [{"role": "user", "content": "yet another unique prompt abc"}]
    })

    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r3.status_code == 429


def test_analytics_reflects_hits_and_misses(client):
    c = _create_client(client, "analytics-co")
    headers = {"X-API-Key": c["api_key"]}
    body = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "tell me a joke"}]}

    client.post("/v1/chat/completions", headers=headers, json=body)  # miss
    client.post("/v1/chat/completions", headers=headers, json=body)  # exact hit
    client.post("/v1/chat/completions", headers=headers, json=body)  # exact hit

    resp = client.get(f"/analytics/usage/{c['id']}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_requests"] == 3
    assert data["exact_hits"] == 2
    assert data["misses"] == 1
    assert data["cache_hit_rate"] == round(2 / 3, 4)
    assert data["estimated_cost_saved_usd"] >= 0


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
