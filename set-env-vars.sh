export AWS_ENDPOINT_URL="http://ziqni-minio.bronzegate.local:9000"

# required creds
export AWS_ACCESS_KEY_ID="94xjm3OIExIud42SorG8"
export AWS_SECRET_ACCESS_KEY="vXK9NOzNTHrfmACZIXvYsmyXcS6gJqr2JAcTTme8"

# "bronzegate-local-1" use a normal AWS region name so it doesn't synthesize s3.<region>.amazonaws.com
export AWS_REGION="us-east-1"
unset AWS_DEFAULT_REGION   # <- IMPORTANT: this was causing the bogus host

# optional but recommended
export AWS_EC2_METADATA_DISABLED=true
export S3_USE_SSL=false