#### Main script to train SRGAN ####
# Authors: Zainab Khan, Jonas Messner

# Packages
import os
import socket

import time

HOST = ''
PORT = 45483 # Port to listen on (non-privileged ports are > 1023)

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
            path = './results/pirm_valset_10/'
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
          os.system("matlab -nodisplay -nosplash -nodesktop -r \"run('evaluation/PIRM2018/evaluate_results.m');exit;\"")

