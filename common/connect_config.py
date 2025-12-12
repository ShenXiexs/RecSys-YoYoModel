import os

# upload
upload_oss_access_key_id = os.getenv("UPLOAD_OSS_ACCESS_KEY_ID", "")
upload_oss_access_key_secret = os.getenv("UPLOAD_OSS_ACCESS_KEY_SECRET", "")
upload_oss_endpoint = os.getenv("UPLOAD_OSS_ENDPOINT", "oss-cn-beijing-internal.aliyuncs.com")
upload_oss_bucket_name = os.getenv("UPLOAD_OSS_BUCKET_NAME", "")

# aliyun/downodps/odps_util/generate_body
your_default_project = os.getenv("ALIYUN_DEFAULT_PROJECT", "adx_dmp")
your_accesskey_id = os.getenv("ALIYUN_ACCESS_KEY_ID", "")
your_accesskey_secret = os.getenv("ALIYUN_ACCESS_KEY_SECRET", "")
tunnel_endpoint = os.getenv("ALIYUN_TUNNEL_ENDPOINT", "http://dt.cn-beijing.maxcompute.aliyun-inc.com")
your_end_point = os.getenv("ALIYUN_ENDPOINT", "http://service.cn-beijing.maxcompute.aliyun-inc.com/api")

# redis credential placeholders (override via environment variables)
redis_username = os.getenv("YOYO_REDIS_USERNAME", "yoyo")
redis_password = os.getenv("YOYO_REDIS_PASSWORD", "")
redis_host = os.getenv("YOYO_REDIS_HOST", "")
redis_port = int(os.getenv("YOYO_REDIS_PORT", "6379"))

redis_feature_username = os.getenv("YOYO_REDIS_FEATURE_USERNAME", "yoyo")
redis_feature_password = os.getenv("YOYO_REDIS_FEATURE_PASSWORD", "")
redis_feature_host = os.getenv("YOYO_REDIS_FEATURE_HOST", "")
redis_feature_port = int(os.getenv("YOYO_REDIS_FEATURE_PORT", "6379"))
