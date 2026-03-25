"""Convenience entrypoint for local deployment from source checkout."""

from gb_sqd.main import deploy_ext_sqd, deploy_trim_sqd


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "trim":
        print("Deploying GB-SQD TrimSQD workflow...")
        deploy_trim_sqd()
    else:
        print("Deploying GB-SQD ExtSQD workflow...")
        deploy_ext_sqd()

