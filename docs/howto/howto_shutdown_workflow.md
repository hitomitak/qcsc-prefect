# How to shutdown the Workflow

## Step 1. Reconnect to the Workflow Server

<img src="./images/icon-pc.png" alt="pc" width="50"/><br>
```bash
ssh -A z12345@mdx-workflow.example.org
```

## Step 2. Exit the Screen Session

Exit screen session from the terminal:

<img src="./images/icon-mdx.png" alt="mdx" width="50"/><br>
```bash
screen -X -S sqd-workflow quit
```

Confirm the session is removed:

<img src="./images/icon-mdx.png" alt="mdx" width="50"/><br>
```bash
screen -ls
```

You should see:

```text
There are screens on:
        146047.sshfs    (Detached)
1 Sockets in /run/screen/S-z12345.
```
> [!NOTE]
> If you want to logout the Prefect portal, please delete the cookie for https://prefect-portal.example.org with your web browser. 