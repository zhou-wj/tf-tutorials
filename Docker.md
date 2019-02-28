
# Docker
You have to use the Docker to complete all your assignments on servers.

## Instructions
### Initialization(Only need to run once)
Before creating the container, you should build a directory under **~/students/** using your own name, e.g. **~/students/chenghao**.

And then,
```bash
nvidia-docker run -it -v /home/student/course/:/root/course/ -v /home/student/students/${YourName}/:/root/${YourName} --name=${YourID} ufoym/deepo:tensorflow-py36-cu90 bash
```
**${Yourname} **mean your name in pinyin, **${YourID} **means your student ID.

e.g.  `nvidia-docker run -it -v /home/student/course/:/root/course/ -v /home/student/students/chenghao/:/root/chenghao --name=1801213964 ufoym/deepo:tensorflow-py36-cu90 bash `

Then you can run your code in the container. **The code is in /root/course. **

To use opencv package, you need to install it in your container.

```
pip install opencv-python
apt update && apt install -y libsm6 libxext6
apt-get install libxrender1
```



##### Warnings
- **You must use your real name and real id. All containers that do not conform to the naming convention will be cleared!!**

  ​

### GPU Usage

When running your train script,  you should use environment variable **CUDA_VISIBLE_DEVICES** to specify which GPU your program is running on. 

```
CUDA_VISIBLE_DEVICES=0 python train.py
```

To monitor GPU usage, your can use

```
watch nvidia-smi
```



If your program is still running, but you want to temporarily exit the terminal...

### Quit

If your program is still running, but you want to temporarily exit the terminal...

`Ctrl+P+Q`  (Press the three buttons together)

**Attention**:

**Do not exit from your container when you program is still running!!**

### Attach

Attach to a started container

```bash
nvidia-docker attach ${YourId}
```

e.g.  `nvidia-docker attach 1801213964  `

And then press  `Enter`.

### Exit

You should stop the container after completing the homework.
```bash
exit
```
or  `Ctrl+D`. 

Once you exit from the container, all files will not be deleted, but all the programs still running will be killed.

If you want to attach to the container you exited, you should first **start** the container, and then **attach** to it.

### Start
You cannot attach to a stopped container, start it first
```bash
nvidia-docker start ${YourID}
```
e.g.  `nvidia-docker start 1801213964  `



