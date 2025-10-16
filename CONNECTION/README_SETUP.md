# ✅ Development Environment Setup Guide

This README provides step-by-step installation instructions for setting up essential tools for development:

- MySQL  
- Python 3.10  
- Jupyter Notebook  
- Visual Studio Code (VS Code)  

---

## 1. 🔧 Install MySQL

**Step 1:** Download MySQL Installer from:  
👉 [https://dev.mysql.com/downloads/installer/](https://dev.mysql.com/downloads/installer/)

**Step 2:** Choose the **"Windows (x86, 32-bit), MSI Installer"**.

**Step 3:** Run the installer and select:
- `Developer Default` (for full setup) OR  
- `Custom` (to install only MySQL Server & Workbench)

**Step 4:** Follow the installation steps:
- Install MySQL Server (default port: 3306)  
- Set a **root password**  
- Optionally create additional users  
- Finish configuration  

**Step 5:** Launch **MySQL Workbench** or use `mysql` from the terminal.

---

## 2. 🐍 Install Python 3.10

**Step 1:** Download from the official site:  
👉 [https://www.python.org/downloads/release/python-3100/](https://www.python.org/downloads/release/python-3100/)

**Step 2:** Select the Windows installer (64-bit or 32-bit).

**Step 3:** Run the installer and:
- ✅ Check **"Add Python 3.10 to PATH"**  
- Click **Install Now**

**Step 4:** Verify installation:
```bash
python --version
```

---

## 3. 📓 Install Jupyter Notebook

> Make sure Python and pip are installed before continuing.

**Step 1:** Open Command Prompt or Terminal.

**Step 2:** Install Jupyter:
```bash
pip install notebook
```

**Step 3:** Launch Jupyter:
```bash
jupyter notebook
```
This will open in your default web browser.

---

## 4. 🧑‍💻 Install Visual Studio Code (VS Code)

**Step 1:** Download from:  
👉 [https://code.visualstudio.com/](https://code.visualstudio.com/)

**Step 2:** Run the installer and check:
- ✅ "Add to PATH"  
- ✅ "Register Code as editor for supported file types"

**Step 3:** Launch VS Code.

**Step 4:** Install Python Extension:
- Open **Extensions (Ctrl+Shift+X)**  
- Search for **Python**  
- Install the one by **Microsoft**

---

## ✅ Verify Installations

Run these commands to confirm everything is installed:

```bash
python --version
pip --version
mysql --version
code --version
jupyter --version
```

---

### 🚀 You're Ready to Code!
