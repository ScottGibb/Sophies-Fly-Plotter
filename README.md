# Using UV

TKinter (Need to Install first on WSL)

```bash
sudo apt update
sudo apt install python3-tk -y

TKinter On Windows

```powershell
$env:Path = "C:\Users\kylel\.local\bin;$env:Path"

```

```powershell
uv venv --python 3.12.0
.venv\Scripts\activate.bat
uv pip install -r requirements.txt
```


Test TKinter

```bash
python3 -c "import tkinter; tkinter._test()"
```
