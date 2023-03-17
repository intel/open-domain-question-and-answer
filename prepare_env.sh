#!/bin/bash
# install required software packages
yum updtate
yum install -y yum-utils

# set up the repository
yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo

# install and start service
yum install -y docker-ce
systemctl start docker

# install docker compose
curl -SL https://github.com/docker/compose/releases/download/v2.6.1/docker-compose-linux-x86_64 -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose
ln -sf /usr/local/bin/docker-compose /usr/bin/docker-compose
# check the installation
echo `docker-compose version`
