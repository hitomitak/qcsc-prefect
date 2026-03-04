# How to File Sync Between MDX and Miyabi-C

This guide explains how to set up a QCSC workflow execution environment on the [mdx](https://mdx.jp/) platform.
By following these steps, you will mount Miyabi shared storage on the MDX workflow server and prepare it for job submission.

## Prerequisites

Before you begin, ensure the following:

- You have an MDX workflow server account.
- SSH authentication to the workflow server is configured.
- SSH public key authentication and one-time password (OTP) are set up for the Miyabi login node.

## Instructions
> [!IMPORTANT]  
> Replace `gz00` and `z12345` with your actual group and account name.

### Step 1. Log in to the Workflow Client

Connect to the MDX workflow Client:

<img src="./images/icon-pc.png" alt="pc" width="50"/><br>
```bash
ssh -A z12345@mdx-workflow.example.org
```

### Step 2. Configure SSH Keepalive

Open your SSH config file of the MDX workflow server:

<img src="./images/icon-mdx.png" alt="mdx" width="50"/><br>
```bash
vi ~/.ssh/config
```

Add the following lines:

```ini
Host *
    ServerAliveInterval 60
```

> [!NOTE]
> This setting is necessary to avoid firewall inactivity timeout.

Set appropriate file permissions:

<img src="./images/icon-mdx.png" alt="mdx" width="50"/><br>
```bash
chmod 600 ~/.ssh/config
```

### Step 3. Create a Mount Point

Create a directory to mount Miyabi shared storage:

<img src="./images/icon-mdx.png" alt="mdx" width="50"/><br>
```bash
mkdir /work/gz00/z12345
```

### Step 4. Start a Persistent SSH Session

Launch a new `screen` session:

<img src="./images/icon-mdx.png" alt="mdx" width="50"/><br>
```bash
screen -S sshfs
```

### Step 5. Mount Miyabi Shared Storage

Mount the remote storage using `sshfs`:

<img src="./images/icon-mdx.png" alt="mdx" width="50"/><br>
```bash
sshfs miyabi-c.example.org:/work/gz00/z12345 /work/gz00/z12345
```

You will be prompted to authenticate with the Miyabi login node.
This requires:

- SSH public key authentication
- A six-digit one-time password (OTP) from your authenticator app

Example output:

```
The authenticity of host 'hpc-login.example.org (203.0.113.10)' can't be established.
ED25519 key fingerprint is SHA256:...
This key is not known by any other names
Are you sure you want to continue connecting (yes/no/[fingerprint])? yes
(z12345@miyabi-c.example.org) Verification code: 123456
```

Your Miyabi shared storage is now mounted and accessible from the MDX workflow server.

**Success check:** 

<img src="./images/icon-mdx.png" alt="mdx" width="50"/><br>
```bash
df -h | grep "/work/<group>/<account>"
ls -la /work/<group>/<account>
```

**Common `sshfs` failure and recovery** 

<img src="./images/icon-mdx.png" alt="mdx" width="50"/><br>
```bash
# If you see: "Transport endpoint is not connected"
fusermount -u /work/<group>/<account>

umount /work/<group>/<account>
# Then re-run sshfs
```

### Step 6. Detach the Screen Session

Once the mount is successful, detach the screen session:

Press `<ctrl> + a`, then:

```bash
d
```
## Verify Connection (Optional)

To test HPC job execution from the workflow server, follow these steps.

### Step 1. Create a Job Script

Create and move to a workflow directory:

<img src="./images/icon-mdx.png" alt="mdx" width="50"/><br>
```bash
mkdir /work/gz00/z12345/test-mdx && cd /work/gz00/z12345/test-mdx
```

Open a job script file:

<img src="./images/icon-mdx.png" alt="mdx" width="50"/><br>
```bash
vi test.pbs
```

Add the following lines:
> [!NOTE]  
> Replace `group_list=gz00` with your user group (project) on Miyabi.  

```sh
#!/bin/sh

#PBS -q debug-c
#PBS -l select=1
#PBS -l walltime=00:00:30
#PBS -W group_list=gz00

echo "Hello I'm $(hostname)"
```

### Step 2. Execute on Miyabi

Submit the job using `qsub`:

<img src="./images/icon-mdx.png" alt="mdx" width="50"/><br>
```bash
qsub test.pbs
```

Example output:
```text
871213.opbs
```

It may take some time to complete the job on a Miyabi compute node.

After successful execution, check the output files:
<img src="./images/icon-mdx.png" alt="mdx" width="50"/><br>
```bash
ls .
```

Example output:

```text
test.pbs  test.pbs.e871213  test.pbs.o871213
```

View the stdout file:

<img src="./images/icon-mdx.png" alt="mdx" width="50"/><br>
```bash
cat test.pbs.o871213
```

Example output:

```text
Hello I'm mc002
```

---
*END OF GUIDE*
