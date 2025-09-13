# debug_list_curated.py
import os
import boto3
from botocore.config import Config

BUCKET = "betfair-curated"  # change if your bucket name differs
PREFIXES = [
    "orderbook_snapshots_5s/",
    "market_definitions/",
    "results/",
]

def main():
    endpoint = os.environ.get("S3_ENDPOINT_URL") or os.environ.get("AWS_ENDPOINT_URL")
    access = os.environ.get("AWS_ACCESS_KEY_ID")
    secret = os.environ.get("AWS_SECRET_ACCESS_KEY")
    region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access,
        aws_secret_access_key=secret,
        region_name=region,
        config=Config(s3={"addressing_style": "path"})
    )

    # List buckets
    print("Buckets:")
    for b in s3.list_buckets().get("Buckets", []):
        print("  -", b["Name"])

    # Scan curated datasets
    for pref in PREFIXES:
        print(f"\n=== Listing: s3://{BUCKET}/{pref} ===")
        resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=pref, MaxKeys=2000)
        contents = resp.get("Contents", [])
        if not contents:
            print("  (empty)")
            continue

        # Show a few sample keys and any date partitions discovered
        shown = 0
        dates = set()
        for obj in contents:
            key = obj["Key"]
            if "/sport=" in key and "/date=" in key:
                # pick out date partition
                try:
                    part = key.split("/date=", 1)[1].split("/", 1)[0]
                    dates.add(part)
                except Exception:
                    pass
            if shown < 10:
                print("  -", key)
                shown += 1
        if dates:
            print("  dates found:", sorted(dates)[:20], ("... (+more)" if len(dates) > 20 else ""))

if __name__ == "__main__":
    main()
