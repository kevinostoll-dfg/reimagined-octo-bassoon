import redis
import json
import os
import sys

# Configuration
DRAGONFLY_HOST = os.getenv("DRAGONFLY_HOST", "dragonfly.blacksmith.deerfieldgreen.com")
DRAGONFLY_PORT = int(os.getenv("DRAGONFLY_PORT", 6379))
DRAGONFLY_PASSWORD = os.getenv("DRAGONFLY_PASSWORD", "YXnzuzpbjgOZMEuJ")

SCHEMA_KEYS = [
    "memgraph:cipher:full",
    "memgraph:cipher:ea",
    "memgraph:cipher:fomc",
    "memgraph:cipher:sec-10k",
    "memgraph:cipher:sec-f4"
]

def main():
    print(f"🔌 Connecting to Dragonfly at {DRAGONFLY_HOST}:{DRAGONFLY_PORT}...")
    try:
        client = redis.Redis(
            host=DRAGONFLY_HOST,
            port=DRAGONFLY_PORT,
            password=DRAGONFLY_PASSWORD,
            decode_responses=True,
            socket_timeout=5
        )
        client.ping()
        print("✅ Connected!")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return

    full_dump = {}

    print("\n🔍 Fetching raw data for all keys...")
    for key in SCHEMA_KEYS:
        print(f"📥 GET {key}...", end=" ")
        try:
            data = client.get(key)
            if data:
                print("✅ Found")
                try:
                    # Try to parse as JSON just to ensure it's valid, but store structure
                    json_data = json.loads(data)
                    full_dump[key] = json_data
                except json.JSONDecodeError:
                    print("⚠️  (Not JSON)")
                    full_dump[key] = {"__raw_error__": "Could not decode JSON", "__raw_content__": data}
            else:
                print("⚠️  (Not Found)")
                full_dump[key] = None
        except Exception as e:
            print(f"❌ Error: {e}")
            full_dump[key] = {"__error__": str(e)}

    # Save full dump
    output_file = "dragonfly_full_dump.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(full_dump, f, indent=2)
    
    print(f"\n💾 Raw data dumped to `{output_file}`")
    
    # Also print to console for immediate visibility if small enough
    print("\n--- RAW CONTENT START ---")
    print(json.dumps(full_dump, indent=2))
    print("--- RAW CONTENT END ---")

if __name__ == "__main__":
    main()
