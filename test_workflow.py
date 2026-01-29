import requests
import time
import json
import sys

BASE_URL = "http://localhost:8000"

def wait_for_job(endpoint, job_id, description):
    print(f"Waiting for {description} (Job ID: {job_id})...")
    while True:
        response = requests.get(f"{BASE_URL}{endpoint}/status/{job_id}")
        data = response.json()
        status = data.get("status")
        progress = data.get("progress_percent", 0)
        
        print(f"\rStatus: {status} ({progress}%)", end="")
        
        if status == "completed":
            print(f"\n{description} completed!")
            return data
        elif status == "failed":
            print(f"\n{description} failed: {data.get('error')}")
            sys.exit(1)
            
        time.sleep(1)

def test_workflow():
    print("=== Starting System Test ===\n")

    # 1. Ingest Data (Local)
    print("1. Ingesting data from local directory...")
    try:
        resp = requests.post(f"{BASE_URL}/api/v1/ingest/local")
        resp.raise_for_status()
        ingest_data = resp.json()
        ingest_job_id = ingest_data["job_id"]
        wait_for_job("/api/v1/ingest", ingest_job_id, "Ingestion")
    except Exception as e:
        print(f"Ingestion failed: {e}")
        return

    # 2. Build Graph
    print("\n2. Building Knowledge Graph...")
    graph_payload = {
        "ingestion_job_id": ingest_job_id,
        "enable_structural_links": True,
        "enable_semantic_links": True,
        "enable_reference_links": True,
        "semantic_threshold": 0.75
    }
    try:
        resp = requests.post(f"{BASE_URL}/api/v1/graph/build", json=graph_payload)
        resp.raise_for_status()
        graph_data = resp.json()
        graph_job_id = graph_data["job_id"]
        wait_for_job("/api/v1/graph", graph_job_id, "Graph Building")
    except Exception as e:
        print(f"Graph build failed: {e}")
        return

    # 3. Retrieval Query
    print("\n3. Testing Retrieval...")
    component_profile = {
        "name": "Automotive LED Module",
        "type": "LED Light Source",
        "application": "Headlamp",
        "variants": ["Standard"],
        "test_level": "Component",
        "applicable_standards": ["ISO 16750-4", "IEC 60068"],
        "test_categories": ["thermal", "environmental"],
        "quantity_per_test": {"Standard": 5}
    }
    
    query_payload = {
        "component_profile": component_profile,
        "retrieval_method": "hybrid",
        "max_results": 5,
        "min_confidence": 0.1
    }
    
    try:
        resp = requests.post(f"{BASE_URL}/api/v1/retrieval/query", json=query_payload)
        resp.raise_for_status()
        retrieval_data = resp.json()
        results = retrieval_data.get("results", [])
        print(f"Retrieved {len(results)} requirements.")
        if not results:
            print("No results found. Stopping.")
            return
        print(f"Top result: {results[0].get('text')[:100]}...")
    except Exception as e:
        print(f"Retrieval failed: {e}")
        return

    # 4. LLM Generation
    print("\n4. Generating Test Procedures (LLM)...")
    llm_payload = {
        "retrieved_context": results,
        "component_profile": component_profile,
        "generation_mode": "detailed",
        "include_traceability": True
    }
    
    try:
        resp = requests.post(f"{BASE_URL}/api/v1/llm/generate", json=llm_payload)
        resp.raise_for_status()
        llm_data = resp.json()
        llm_job_id = llm_data["job_id"]
        llm_result = wait_for_job("/api/v1/llm", llm_job_id, "LLM Generation")
        
        test_cases = llm_result.get("result", {}).get("test_procedures", [])
        print(f"Generated {len(test_cases)} test cases.")
        
        if not test_cases:
            print("\n[WARN] LLM generated 0 test cases (likely due to missing LLM server).")
            print("Using mock test cases to continue testing DVP generation...")
            test_cases = [{
                "test_id": "B1",
                "test_standard": "ISO 16750-4",
                "test_description": "Operation at Low Temperature",
                "test_procedure": "1. Soak at -40C for 24h. 2. Operate.",
                "acceptance_criteria": "Must operate normally.",
                "test_responsibility": "Supplier",
                "test_stage": "DVP",
                "quantity": "5",
                "estimated_days": 2,
                "pcb_or_lamp": "System",
                "traceability": {
                    "requirement_id": "mock_req_1",
                    "source_clause": "5.1",
                    "source_standard": "ISO 16750-4"
                }
            }]

    except Exception as e:
        print(f"LLM generation failed: {e}")
        # Continue with mock data if there was an exception
        test_cases = [{
            "test_id": "B1",
            "test_standard": "ISO 16750-4",
            "test_description": "Operation at Low Temperature (Mock)",
            "test_procedure": "Mock procedure",
            "acceptance_criteria": "Mock criteria",
            "test_responsibility": "Supplier",
            "test_stage": "DVP",
            "quantity": "5",
            "estimated_days": 2,
            "pcb_or_lamp": "System",
            "traceability": {}
        }]

    # 5. DVP Generation
    print("\n5. Generating DVP Document...")
    dvp_payload = {
        "component_profile": component_profile,
        "test_cases": test_cases,
        "output_format": "docx",
        "include_traceability_sheet": True
    }
    
    try:
        resp = requests.post(f"{BASE_URL}/api/v1/dvp/generate", json=dvp_payload)
        resp.raise_for_status()
        dvp_data = resp.json()
        dvp_job_id = dvp_data["job_id"]
        dvp_result = wait_for_job("/api/v1/dvp", dvp_job_id, "DVP Generation")
        
        final_result = dvp_result.get("result", {})
        print(f"\nDVP Generated successfully!")
        print(f"File path: {final_result.get('file_path')}")
        print(f"Download URL: /api/v1/dvp/download/{final_result.get('dvp_id')}")
    except Exception as e:
        print(f"DVP generation failed: {e}")

if __name__ == "__main__":
    test_workflow()
