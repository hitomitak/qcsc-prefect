# Workflow for observability demo on Miyabi
#
# Author: Naoki Kanazawa (knzwnao@jp.ibm.com)


import asyncio
import sys
from pathlib import Path

from src import riken_sqd_de
from src.flow_params import FlowParameters


def main():
    json_file = Path(sys.argv[1])
    params = FlowParameters.model_validate_json(json_file.read_text())
    
    coro = riken_sqd_de(parameters=params)    
    asyncio.run(coro)


if __name__ == "__main__":
    main()
