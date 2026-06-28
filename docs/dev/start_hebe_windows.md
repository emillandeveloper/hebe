# Starting Hebe on Windows

Use the launch scripts in the project root to start Hebe without manually opening CMD and changing folders.

## One-click launch

Double-click:

```text
start-hebe.bat
```

The launcher will:

- find the project root from its own location
- check that Node.js and npm are available
- change into the `frontend` folder
- run `npm run electron:dev`
- keep the window open if startup fails

You can also run the PowerShell version:

```powershell
.\start-hebe.ps1
```

If PowerShell blocks script execution, use `start-hebe.bat`.

## Create a desktop shortcut

From PowerShell in the project root, run:

```powershell
.\create-desktop-shortcut.ps1
```

This creates:

```text
Hebe.lnk
```

on the Windows desktop. The shortcut points to `start-hebe.bat` and starts in the project root.

## Troubleshooting

If the launcher says Node.js or npm was not found, install Node.js or make sure it is available on `PATH`.

If the launcher cannot find `frontend\package.json`, make sure the scripts are still in the Hebe project root.

If `npm run electron:dev` fails, the terminal window will stay open with the error. The manual command still works:

```powershell
cd "C:\Users\Leo Nifelheim\Documents\Hebe\hebe-ui\frontend"
npm run electron:dev
```

## Notes

These scripts do not package Electron, build an exe, install a service, or enable startup on login. They only make the existing development command easier to start.
