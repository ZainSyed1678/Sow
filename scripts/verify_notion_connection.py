"""Read-only verification tool for Notion integration and database permissions.

Safe to run: Never writes or alters any Notion content.
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ai_ml_notion_researcher.config import load_config
from ai_ml_notion_researcher.logging_utils import setup_logging
from ai_ml_notion_researcher.models import NotionDatabaseType
from ai_ml_notion_researcher.notion.client import NotionClient


def main() -> int:
    setup_logging(log_level="INFO")
    print("\n" + "=" * 60)
    print("AI/ML Notion Researcher - Notion Connection Verification")
    print("=" * 60)

    config = load_config()

    if not config.notion.api_key or not config.notion.api_key.get_secret_value().strip():
        print("[FAIL] NOTION_API_KEY is not set in your .env file.")
        print("Please create an integration at https://www.notion.so/my-integrations")
        return 1

    api_key = config.notion.api_key.get_secret_value()
    all_passed = True

    with NotionClient(api_key=api_key) as client:
        print(f"\nChecking 5 Notion databases using integration token (ends with ...{api_key[-4:]}):\n")

        for db_type in NotionDatabaseType:
            db_id = config.notion.database_ids.get(db_type, "")
            label = db_type.value.replace("_", " ").title()

            if not db_id:
                print(f"[-] {label:22} | Missing database ID in .env")
                all_passed = False
                continue

            is_valid, message, props = client.validate_database(db_id)
            if is_valid:
                prop_names = ", ".join(list(props.keys())[:4])
                if len(props) > 4:
                    prop_names += f" (+{len(props) - 4} more)"
                print(f"[OK] {label:22} | {message}")
                print(f"     Properties detected: {prop_names}")
            else:
                print(f"[FAIL] {label:20} | {message}")
                all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("All 5 Notion databases are verified and accessible!")
        print("=" * 60 + "\n")
        return 0
    else:
        print("One or more databases could not be verified.")
        print("Tips:")
        print("1. Ensure each database has been shared with your Notion integration:")
        print("   In Notion -> Open Database -> Click '...' -> Connections -> Add Integration.")
        print("2. Ensure the database ID is the 32-character hex string from the URL.")
        print("=" * 60 + "\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
