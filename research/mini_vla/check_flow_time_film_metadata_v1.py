"""Validate deployed websocket identity before the common rollout barrier."""
import argparse,json
from pathlib import Path
from openpi_client.websocket_client_policy import WebsocketClientPolicy
from .flow_time_film_metadata_v1 import validate_metadata

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--port',type=int,required=True);p.add_argument('--output',required=True);a=p.parse_args()
    plan=json.loads(Path(a.plan).read_text());client=WebsocketClientPolicy('127.0.0.1',a.port)
    metadata=client.get_server_metadata();validate_metadata(metadata,plan)
    Path(a.output).write_text(json.dumps(dict(status='PASS',metadata=metadata,new_policy_calls=0),indent=2)+'\n')
if __name__=='__main__':main()
