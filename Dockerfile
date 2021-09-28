FROM supervisely/base-py-sdk:6.1.97

RUN apt-get update && apt-get install -y \
openssh-server \
python3-tk \
sudo

WORKDIR /work

COPY requirements.txt /work/
RUN pip3 install -r /work/requirements.txt

RUN mkdir -p /run/sshd

ARG home=/root
RUN mkdir $home/.ssh
COPY my_key.pub $home/.ssh/authorized_keys
RUN chown root:root $home/.ssh/authorized_keys && \
    chmod 600 $home/.ssh/authorized_keys

COPY sshd_daemon.sh /sshd_daemon.sh
RUN chmod 755 /sshd_daemon.sh
CMD ["/sshd_daemon.sh"]
ENTRYPOINT ["sh", "-c", "/sshd_daemon.sh"]
