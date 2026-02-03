#!/bin/bash
# ��ȡ���н�ʬ���̵�PID
# zombie_pids=$(ps -A -ostat,ppid,pid,cmd | grep -e '^[Zz]' | awk '{print $2}')
zombie_pids=$(ps aux|grep python | awk '{print $2}')
# ����Ƿ��ҵ���ʬ����
if [ -z "$zombie_pids" ]; then
    echo "No zombie process was found"
    exit 0
fi
# ѭ���������н�ʬ���̵�PID��������ɱ������
for pid in $zombie_pids; do
    echo "Try to kill the zombie process: PID $pid"
    kill -9 $pid 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "Successfully kill the zombie process: PID $pid"
    else
        echo "Failed to kill zombie process: PID $pid"
    fi
done