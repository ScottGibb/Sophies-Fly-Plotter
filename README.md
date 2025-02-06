# Using UV

TKinter (Need to Install first on WSL)
```bash
sudo apt update
sudo apt install python3-tk -y
```

```bash
uv venv --python 3.12.0
uv pip install -r requirements.txt
```

Test TKinter
```bash
python3 -c "import tkinter; tkinter._test()"
```

