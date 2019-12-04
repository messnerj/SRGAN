#### Main script to train SRGAN ####
# Authors: Zainab Khan, Jonas Messner

# Packages
import os
import socket

import time
import argparse

parser = argparse.ArgumentParser(description='Parameters for training SRGAN.')
parser.add_argument('--upscale_factor', default=4, type=int,
                            help='how much to super resolve image by')
parser.add_argument('--port', default=45483, type=int,
                            help='which port to host')
args = parser.parse_args()

HOST = ''
PORT = args.port # Port to listen on (non-privileged ports are > 1023)

path = './results/pirm_valset_10_%d/' % args.upscale_factor
if not os.path.exists(path):
    os.makedirs(path)

i = 0
while True:
        print("starting evaluation",i)
        i = i+1
        print('PIRM Validation')
        # Rcv PIRM images
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
          s.bind((HOST, PORT))
          s.listen()
 
          for batch_idx in range(10):
            print("Waiting for connection at port",PORT,"from",HOST)
            conn, addr = s.accept()
            print('Connected by', addr)
            app = str(batch_idx+1)+'.png'
            fname = path + app
            tmp = open(fname, 'wb')
            while True:
                data = conn.recv(1024)
                if not data:
                    break
                tmp.write(data)     
            print("ended at",tmp.tell())
            tmp.close()
            print("rcved file")
            conn.close()
         
          # Execute PIRM validation
          os.system("matlab -nodisplay -nosplash -nodesktop -r \"run('evaluation/PIRM2018/evaluate_results_%d.m');exit;\"" \
                        % args.upscale_factor)

