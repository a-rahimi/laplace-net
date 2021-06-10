#!/usr/bin/env python3

from typing import List, Sequence

import argparse
import os
import time

import boto3
import botocore.exceptions

client = boto3.client("ec2")
ec2 = boto3.resource("ec2")


def run_instances(
    num_instances: int,
    role: str,
    user: str = os.environ["USER"],
    instance_type: str = None,
    image_id: str = "ami-0409ee1400c11f1e7",
) -> List[ec2.Instance]:
    # This call is documented here:
    #    https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ec2.html#EC2.Client.run_instances
    response = client.run_instances(
        BlockDeviceMappings=[
            {
                "DeviceName": "/dev/sda1",
                "Ebs": {
                    "DeleteOnTermination": True,
                    "VolumeSize": 3000,
                    "VolumeType": "gp2",
                },
            },
        ],
        ImageId=image_id,
        InstanceType=instance_type
        or ("g4dn.xlarge" if role == "dev-instance" else "p3.2xlarge"),
        MaxCount=num_instances,
        MinCount=num_instances,
        Monitoring={"Enabled": True},
        SecurityGroupIds=["sg-0719266b3840a636a", "sg-5c8ce072"],
        UserData=user,
        EbsOptimized=True,
        InstanceInitiatedShutdownBehavior=(
            "stop" if role == "dev-instance" else "terminate"
        ),
        TagSpecifications=[
            {
                "ResourceType": "instance",
                "Tags": [
                    {"Key": "Name", "Value": user + " " + role},
                    {"Key": "WorkGroup", "Value": user},
                    {"Key": "Role", "Value": role},
                ],
            },
        ],
    )

    return [ec2.Instance(instance["InstanceId"]) for instance in response["Instances"]]


def update_dev_instance_ssh_config(ip_address: str):
    """Update ~/.ssh/config.d/dev-instance with information about your instance.

    This presumes your ~/.ssh/config contains a line like:

       Include ~/.ssh/config.d/*
    """
    ssh_config_dir = os.path.join(os.environ["HOME"], ".ssh/config.d")
    os.makedirs(ssh_config_dir, exist_ok=True)

    with open(os.path.join(ssh_config_dir, "dev-instance"), "w") as f:
        f.write(
            f"""
                    Host dev-instance
                      HostName {ip_address}
                      IdentityFile ~/.ssh/counting.pem
                      User ubuntu
                      # for jupyter notebook
                      LocalForward 8888 localhost:8888

                    Host dev-instance-tmux
                      HostName {ip_address}
                      IdentityFile ~/.ssh/counting.pem
                      User ubuntu
                      RemoteCommand tmux -CC new-session -A -s main
                      RequestTTY yes
                      ServerAliveInterval 100
                      ServerAliveCountMax 2
                      # for jupyter notebook
                      LocalForward 8888 localhost:8888
                  """
        )

    print(
        """
        Updated your ssh config. You can now 'ssh dev-instance' or 'ssh dev-instance-tmux' into
        your instance, assuming your ~/.ssh/config contains a line like

            Include ~/.ssh/config.d/*
        """
    )


def wait_for_instances(role: str, instances: Sequence[ec2.Instance]):
    """Wait until all the instances have public IP addresses."""

    while instances:
        instances_still_down = []

        for instance in instances:
            # Refresh all information about all the instances we tried to launch

            while True:
                try:
                    instance.load()
                    break
                except botocore.exceptions.ClientError:
                    # Wait a second for the instance id to propagate throughout the system
                    # Related issue: https://github.com/boto/boto3/issues/2556
                    time.sleep(1)

            # If any of them have an IP address, print it and remove it from the list of
            # instances to check on the next iteration.
            if instance.public_ip_address:
                print(instance.public_ip_address)
                if role == "dev-instance":
                    update_dev_instance_ssh_config(instance.public_ip_address)
            else:
                instances_still_down.append(instance)

        instances = instances_still_down
        time.sleep(1)


def main(args: argparse.Namespace):
    if args.the_argument == "dev-instance":
        num_instances = 1
        role = args.the_argument
    else:
        num_instances = int(args.the_argument)
        role = "worker"

    instances = run_instances(num_instances, role, instance_type=args.instance_type)

    wait_for_instances(role, instances)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Spawn an EC2 worker or dev-instance. Waits until the workers are up."
    )
    parser.add_argument(
        "--instance-type",
        help="Defaults to p3.2xlarge for dev-instances and p3.8xlarge for workers.",
    )
    parser.add_argument(
        "the_argument",
        metavar="dev-instance | N",
        type=str,
        help='Either the string "dev-instance" or the number of workers to spawn',
    )
    args = parser.parse_args()

    main(args)
