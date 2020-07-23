import psycopg2
import sqlalchemy
import sshtunnel

#%%
import paramiko
from sshtunnel import SSHTunnelForwarder

with SSHTunnelForwarder(
    (REMOTE_SERVER_IP, 443),
    ssh_username="patrickfmaus",
    ssh_private_key_password="secret",
    remote_bind_address=(PRIVATE_SERVER_IP, 22),
    local_bind_address=('0.0.0.0', 10022)
) as tunnel:
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect('127.0.0.1', 10022)



    client.close()

print('FINISH!')server.stop()