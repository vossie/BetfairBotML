#!/usr/bin/env python3
import os
import boto3
from botocore.exceptions import ClientError

def main():
    endpoint = os.environ.get("S3_ENDPOINT_URL")
    access = os.environ.get("AWS_ACCESS_KEY_ID")
    secret = os.environ.get("AWS_SECRET_ACCESS_KEY")
    region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

    if not (endpoint and access and secret):
        print("❌ Missing required environment variables:")
        print("  S3_ENDPOINT_URL, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY")
        return

    print(f"🔌 Connecting to MinIO at {endpoint} with access key {access}")

    try:
        s3 = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access,
            aws_secret_access_key=secret,
            region_name=region,
        )

        # List buckets
        buckets = s3.list_buckets()
        print("✅ Buckets found:")
        for b in buckets.get("Buckets", []):
            print(f"   - {b['Name']}")

        # Try listing objects in your curated bucket if it exists
        test_bucket = "betfair-curated"
        try:
            objs = s3.list_objects_v2(Bucket=test_bucket, MaxKeys=5)
            print(f"\n📂 Objects in bucket '{test_bucket}':")
            if "Contents" in objs:
                for obj in objs["Contents"]:
                    print(f"   - {obj['Key']} ({obj['Size']} bytes)")
            else:
                print("   (empty)")
        except ClientError as e:
            print(f"⚠️ Could not list objects in '{test_bucket}': {e}")

    except Exception as e:
        print(f"❌ Failed to connect to MinIO: {e}")

if __name__ == "__main__":
    main()
