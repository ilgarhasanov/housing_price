# Running Commands on Windows

Since `make` is not available by default on Windows, you have several options:

## Option 1: Use PowerShell Scripts (Recommended)

Use the `make.ps1` script that replicates all Makefile functionality:

```powershell
# Install dependencies
.\make.ps1 install

# Train the model
.\make.ps1 train

# Start the server
.\make.ps1 serve

# Run tests
.\make.ps1 test

# Build Docker image
.\make.ps1 docker-build

# Run Docker container
.\make.ps1 docker-run

# Show help
.\make.ps1 help
```

## Option 2: Use Individual PowerShell Scripts

You can also run individual scripts directly:

```powershell
.\install.ps1
.\train.ps1
.\serve.ps1
```

## Option 3: Install Make for Windows

If you prefer using `make` directly, you can install it:

### Using Chocolatey:
```powershell
choco install make
```

### Using Scoop:
```powershell
scoop install make
```

### Using winget:
```powershell
winget install GnuWin32.Make
```

### Using Git for Windows:
If you have Git for Windows installed, `make` might already be available in Git Bash.

## Option 4: Use WSL (Windows Subsystem for Linux)

If you have WSL installed, you can use `make` from within WSL:

```bash
wsl make install
wsl make train
wsl make serve
```
