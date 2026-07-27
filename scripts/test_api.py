import requests
import uuid

BASE_URL = "http://127.0.0.1:8000"

def test_api():
    # Use unique username and email to prevent duplication conflicts
    random_str = str(uuid.uuid4())[:8]
    username = f"testuser_{random_str}"
    email = f"test_{random_str}@example.com"
    password = "password123"

    print("1. Registering test user...")
    reg_payload = {
        "username": username,
        "email": email,
        "password": password
    }
    reg_resp = requests.post(f"{BASE_URL}/auth/register", json=reg_payload)
    print("Registration response:", reg_resp.status_code, reg_resp.json())
    assert reg_resp.status_code == 201

    print("\n2. Logging in...")
    login_payload = {
        "username": username,
        "password": password
    }
    login_resp = requests.post(f"{BASE_URL}/auth/login", json=login_payload)
    print("Login response:", login_resp.status_code)
    assert login_resp.status_code == 200
    
    token = login_resp.json()["access_token"]
    headers = {
        "Authorization": f"Bearer {token}"
    }

    print("\n3. Calling GET /players...")
    players_resp = requests.get(f"{BASE_URL}/players/", headers=headers)
    print("Players response:", players_resp.status_code)
    players_data = players_resp.json()
    print("Total players:", players_data.get("total_players"))
    print("Total indexed:", players_data.get("total_indexed"))
    # Print first few players stats
    print("Players list snippet:")
    for p in players_data.get("players", [])[:3]:
        print(f" - {p['player']}: indexed={p['indexed']}, chunks_indexed={p['chunks_indexed']}")

    print("\n4. Calling GET /players/MS_Dhoni/chunks...")
    chunks_resp = requests.get(f"{BASE_URL}/players/MS_Dhoni/chunks?limit=2", headers=headers)
    print("Chunks response:", chunks_resp.status_code)
    chunks_data = chunks_resp.json()
    print("Player name:", chunks_data.get("player"))
    print("Total chunks:", chunks_data.get("total_chunks"))
    print("Returned chunks:", chunks_data.get("returned"))

    print("\n5. Generating story via RAG...")
    story_resp = requests.get(f"{BASE_URL}/story?topic=MS%20Dhoni&story_type=timeline", headers=headers)
    print("Story response status:", story_resp.status_code)
    story_data = story_resp.json()
    print("Story snippet:\n", story_data.get("story", "")[:300] + "...")
    print("Evaluation/Metrics:", story_data.get("evaluation"))

if __name__ == "__main__":
    try:
        test_api()
        print("\n✅ All API Integration Tests Passed Successfully!")
    except Exception as e:
        print("\n❌ API Integration Test Failed:", e)
