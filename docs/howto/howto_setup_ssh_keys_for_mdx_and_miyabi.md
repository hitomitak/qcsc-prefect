# Tutorial: SSH Connection Setup for MDX and Miyabi-C with Git Configuration

This tutorial explains how to set up SSH login to the MDX workflow server, configure Git for cloning from private repositories, and establish SSH connections to Miyabi-C.

## Overview

This tutorial covers the following configurations:

1. **SSH Login to MDX**: Connecting from your local PC to the MDX workflow server
2. **Git SSH Key Setup**: Cloning from private repositories to MDX
3. **SSH Key Setup for Miyabi-C**: Logging in to Miyabi-C from MDX

## Prerequisites

- MDX workflow server account (e.g., `z12345`)
- Miyabi-C account and group (e.g., `gz00/z12345`)
- GitHub account (with access to private repositories)
- OTP authentication app (for Miyabi-C login)

> [!IMPORTANT]
> Replace `z12345` with your actual account name and `gz00` with your actual group name throughout this guide.

---

## Part 1: SSH Login Setup for MDX

### Step 1.1: Generate SSH Key Pair on Local PC

Create a new SSH key pair on your local PC.

<img src="./images/icon-pc.png" alt="pc" width="50"/><br>
```bash
ssh-keygen -t ed25519 -C "your_email@example.com" -f ~/.ssh/id_ed25519_mdx
```

When prompted, set a passphrase (recommended).

Generated files:
- `~/.ssh/id_ed25519_mdx` (private key)
- `~/.ssh/id_ed25519_mdx.pub` (public key)

### Step 1.2: Register Public Key on MDX Server

Display the public key content:

<img src="./images/icon-pc.png" alt="pc" width="50"/><br>
```bash
cat ~/.ssh/id_ed25519_mdx.pub
```

Example output:
```text
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIAbCdEfGhIjKlMnOpQrStUvWxYz your_email@example.com
```

Send this public key to your MDX administrator to register it with your account.

### Step 1.3: Edit SSH Configuration File

Edit the `~/.ssh/config` file on your local PC:

<img src="./images/icon-pc.png" alt="pc" width="50"/><br>
```bash
vim ~/.ssh/config
```

Add the following configuration:

```text
Host mdx
    HostName qii-kawasaki-miyabi-cli.cspp.cc.u-tokyo.ac.jp
    User z12345
    IdentityFile ~/.ssh/id_ed25519_mdx
    ForwardAgent yes
```

### Step 1.4: Test MDX Connection

Once configured, connect to MDX:

<img src="./images/icon-pc.png" alt="pc" width="50"/><br>
```bash
ssh mdx
```

Or:

<img src="./images/icon-pc.png" alt="pc" width="50"/><br>
```bash
ssh -A z12345@qii-kawasaki-miyabi-cli.cspp.cc.u-tokyo.ac.jp
```

On first connection, you will be asked to verify the host key. Type `yes` to continue.

---

## Part 2: Git SSH Key Setup (for Private Repositories)

### Step 2.1: Generate Git SSH Key Pair on MDX

While logged into the MDX server, create a key for Git:

<img src="./images/icon-mdx.png" alt="mdx" width="50"/><br>
```bash
ssh-keygen -t ed25519 -C "your_email@example.com" -f ~/.ssh/id_ed25519_github
```

Generated files:
- `~/.ssh/id_ed25519_github` (private key)
- `~/.ssh/id_ed25519_github.pub` (public key)

### Step 2.2: Register Public Key on GitHub

Display the public key content:

<img src="./images/icon-mdx.png" alt="mdx" width="50"/><br>
```bash
cat ~/.ssh/id_ed25519_github.pub
```

Copy the output and register it on GitHub:

1. Log in to GitHub
2. Go to **Settings** → **SSH and GPG keys** → **New SSH key**
3. **Title**: `MDX Workflow Server`
4. **Key**: Paste the copied public key
5. Click **Add SSH key**

### Step 2.3: Edit SSH Configuration File (on MDX)

Edit `~/.ssh/config` on the MDX server:

<img src="./images/icon-mdx.png" alt="mdx" width="50"/><br>
```bash
vim ~/.ssh/config
```

Add the following configuration:

```text
Host github.com
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_ed25519_github
    IdentitiesOnly yes
```

### Step 2.4: Test GitHub Connection

Test the connection to GitHub:

<img src="./images/icon-mdx.png" alt="mdx" width="50"/><br>
```bash
ssh -T git@github.com
```

On first connection, you will be asked to verify the host key. Type `yes` to continue.

If successful, you will see a message like:

```text
Hi username! You've successfully authenticated, but GitHub does not provide shell access.
```

### Step 2.5: Clone Private Repository

You can now clone private repositories:

<img src="./images/icon-mdx.png" alt="mdx" width="50"/><br>
```bash
cd /work/gz00/z12345
git clone git@github.com:your-org/your-private-repo.git
```

---

## Part 3: SSH Key Setup for Miyabi-C

### Step 3.1: Generate SSH Key Pair for Miyabi-C on MDX

While logged into the MDX server, create a key for Miyabi-C:

<img src="./images/icon-mdx.png" alt="mdx" width="50"/><br>
```bash
ssh-keygen -t ed25519 -C "your_email@example.com" -f ~/.ssh/id_ed25519_miyabi
```

Generated files:
- `~/.ssh/id_ed25519_miyabi` (private key)
- `~/.ssh/id_ed25519_miyabi.pub` (public key)

### Step 3.2: Register Public Key on Miyabi-C

Display the public key content:

<img src="./images/icon-mdx.png" alt="mdx" width="50"/><br>
```bash
cat ~/.ssh/id_ed25519_miyabi.pub
```

Register this public key by Miyabi portal site.

### Step 3.3: Edit SSH Configuration File (on MDX)

Edit `~/.ssh/config` on the MDX server:

<img src="./images/icon-mdx.png" alt="mdx" width="50"/><br>
```bash
vim ~/.ssh/config
```

Add the following configuration:

```text
Host miyabi-c
    HostName miyabi-c.jcahpc.jp
    User z12345
    IdentityFile ~/.ssh/id_ed25519_miyabi
```

### Step 3.4: Test Miyabi-C Connection

Connect to Miyabi-C:

<img src="./images/icon-mdx.png" alt="mdx" width="50"/><br>
```bash
ssh miyabi-c
```

Or:

<img src="./images/icon-mdx.png" alt="mdx" width="50"/><br>
```bash
ssh z12345@miyabi-c.jcahpc.jp
```

On first connection, you will be prompted to enter an OTP (one-time password). Enter the code generated by your authentication app.

If successful, you will be connected to the Miyabi-C login node:

<img src="./images/icon-miyabi.png" alt="miyabi" width="50"/><br>
```bash
# Successfully logged in to Miyabi-C
```

---

## Complete SSH Configuration File Examples

### Local PC (`~/.ssh/config`)

```text
Host mdx
    HostName qii-kawasaki-miyabi-cli.cspp.cc.u-tokyo.ac.jp
    User z12345
    IdentityFile ~/.ssh/id_ed25519_mdx
    ForwardAgent yes
```

### MDX Server (`~/.ssh/config`)

```text
Host github.com
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_ed25519_github
    IdentitiesOnly yes

Host miyabi-c
    HostName miyabi-c.jcahpc.jp
    User z12345
    IdentityFile ~/.ssh/id_ed25519_miyabi
```

---

## Troubleshooting

### Connection Issues

Display detailed debug information when attempting to connect:

<img src="./images/icon-pc.png" alt="pc" width="50"/><br>
```bash
ssh -v mdx
```

Or:

<img src="./images/icon-mdx.png" alt="mdx" width="50"/><br>
```bash
ssh -v miyabi-c
```

### Check Key Permissions

Verify that SSH key permissions are correct:

```bash
chmod 700 ~/.ssh
chmod 600 ~/.ssh/id_ed25519_*
chmod 644 ~/.ssh/id_ed25519_*.pub
```

### GitHub Connection Errors

If you have issues connecting to GitHub:

<img src="./images/icon-mdx.png" alt="mdx" width="50"/><br>
```bash
ssh -vT git@github.com
```

Check the error messages and verify that your public key is correctly registered on GitHub.

---

## Security Best Practices

1. **Use Passphrases**: Always set a passphrase for your SSH keys
2. **Key Management**: Never share your private keys
3. **Regular Key Rotation**: Consider rotating keys periodically for security

---
*END OF GUIDE*