nohup python -m tvm.exec.rpc_tracker --host=0.0.0.0 --port=9190 >/dev/null 2>&1 & 

sleep 1

CUDA_VISIBLE_DEVICES=0 nohup python -m tvm.exec.rpc_server --tracker=127.0.0.1:9190 --key=3090  >/dev/null 2>&1 & 

sleep 1

CUDA_VISIBLE_DEVICES=1 nohup python -m tvm.exec.rpc_server --tracker=127.0.0.1:9190 --key=3090  >/dev/null 2>&1 & 

sleep 1

CUDA_VISIBLE_DEVICES=2 nohup python -m tvm.exec.rpc_server --tracker=127.0.0.1:9190 --key=3090  >/dev/null 2>&1 & 

sleep 1

CUDA_VISIBLE_DEVICES=3 nohup python -m tvm.exec.rpc_server --tracker=127.0.0.1:9190 --key=3090  >/dev/null 2>&1 & 