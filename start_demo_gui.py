"""Button-based startup checklist for the CapNavi demo."""

from __future__ import annotations

import shutil
import subprocess
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk


ROOT = Path(__file__).resolve().parent
LAUNCHER = ROOT / "start_demo.ps1"
REQUIRED_FOLDER = Path(r"R:\Release_Internal_V1A")
SOURCE_FOLDER = Path(r"C:\Users\johncklau\Desktop\Release_Internal_V1A")

STEPS = [
    (
        "Coil safety",
        "Set the coil to a 10 Hz AC sine wave with a 12 A reading. "
        "Increase current slowly and decrease it slowly to zero when shutting off.",
    ),
    (
        "Capsule warm-up",
        "Warm up the capsule by taking photos continuously for at least 10 minutes.",
    ),
    (
        "Recorder and capsule",
        "Re-flash the recorder and restart the capsule.",
    ),
    (
        "FX3",
        "Previous FX3 windows will be closed and FX3 will be opened automatically. "
        "Set 1080x720, enable Image Information Frame, disable Show Image, and press Start.",
    ),
    (
        "Start algorithm on linux machine",
        "Mount the CapNaviExport SMB share and start the reconstruction algorithm "
        "on the Linux machine.",
    ),
    (
        "Robot safety",
        "Make sure the system is unobstructed, restart motor power, and restart "
        "the CAN bus USB connection.",
    ),
]


class StartupGui:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.step_index = 0
        self.process: subprocess.Popen[str] | None = None

        root.title("CapNavi Demo Startup")
        root.geometry("650x390")
        root.resizable(False, False)

        ttk.Label(
            root,
            text="CapNavi automatic demo setup",
            font=("Segoe UI", 16, "bold"),
        ).pack(pady=(18, 8))

        self.progress = ttk.Label(root)
        self.progress.pack()

        self.step_title = ttk.Label(root, font=("Segoe UI", 13, "bold"))
        self.step_title.pack(pady=(22, 8))

        self.step_text = tk.Text(root, height=6, width=70, wrap="word")
        self.step_text.configure(state="disabled")
        self.step_text.pack(padx=24)

        controls = ttk.Frame(root)
        controls.pack(pady=18)
        self.confirm_button = ttk.Button(
            controls,
            text="Confirm and continue",
            command=self.confirm_step,
        )
        self.confirm_button.grid(row=0, column=0, padx=6)
        self.cancel_button = ttk.Button(
            controls,
            text="Cancel",
            command=self.cancel,
        )
        self.cancel_button.grid(row=0, column=1, padx=6)

        self.status = ttk.Label(root, text="Waiting for confirmation.")
        self.status.pack()
        self.check_required_folder()

    def check_required_folder(self) -> None:
        self.confirm_button.configure(state="disabled")
        self.cancel_button.configure(state="disabled")
        self.progress.configure(text="Checking required folder...")
        self.step_title.configure(text="R disk")
        self.set_text(
            "Checking whether R:\\Release_Internal_V1A exists. "
            "If it is missing, it will be copied from the desktop source folder."
        )
        self.status.configure(text="Please wait...")
        self.root.update_idletasks()

        try:
            if not REQUIRED_FOLDER.is_dir():
                if not SOURCE_FOLDER.is_dir():
                    raise FileNotFoundError(
                        f"Source folder not found: {SOURCE_FOLDER}"
                    )
                shutil.copytree(
                    SOURCE_FOLDER,
                    REQUIRED_FOLDER,
                    dirs_exist_ok=True,
                )
            if not REQUIRED_FOLDER.is_dir():
                raise FileNotFoundError(
                    f"Required folder is still missing: {REQUIRED_FOLDER}"
                )
        except OSError as exc:
            self.confirm_button.configure(
                state="normal",
                text="Retry folder check",
                command=self.check_required_folder,
            )
            self.cancel_button.configure(state="normal")
            self.status.configure(text="Required folder check failed.")
            messagebox.showerror("R disk setup failed", str(exc))
            return

        self.step_index = 0
        self.confirm_button.configure(
            state="normal",
            text="Confirm and continue",
            command=self.confirm_step,
        )
        self.cancel_button.configure(state="normal")
        self.status.configure(text="Required folder is ready.")
        self.show_step()

    def show_step(self) -> None:
        if self.step_index >= len(STEPS):
            self.step_title.configure(text="Starting CapNavi...")
            self.progress.configure(text="All preparation steps confirmed.")
            self.set_text("Launching the existing startup workflow.")
            self.confirm_button.configure(state="disabled")
            self.status.configure(text="Please wait for the startup windows.")
            self.launch()
            return

        title, description = STEPS[self.step_index]
        self.progress.configure(
            text=f"Step {self.step_index + 1} of {len(STEPS)}"
        )
        self.step_title.configure(text=title)
        self.set_text(description)
        self.status.configure(text="Confirm this step when it is complete.")

    def set_text(self, text: str) -> None:
        self.step_text.configure(state="normal")
        self.step_text.delete("1.0", tk.END)
        self.step_text.insert("1.0", text)
        self.step_text.configure(state="disabled")

    def confirm_step(self) -> None:
        self.step_index += 1
        self.show_step()

    def launch(self) -> None:
        if not LAUNCHER.is_file():
            messagebox.showerror("Launcher not found", str(LAUNCHER))
            self.status.configure(text="Launcher file was not found.")
            return

        try:
            self.process = subprocess.Popen(
                [
                    "powershell.exe",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(LAUNCHER),
                    "-GuiConfirmed",
                ],
                cwd=str(ROOT),
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
        except OSError as exc:
            messagebox.showerror("Could not start launcher", str(exc))
            self.status.configure(text="Failed to start the launcher.")
            return

        self.status.configure(
            text="Launcher started. Other program windows will open separately."
        )
        self.root.after(1000, self.check_process)

    def check_process(self) -> None:
        if self.process is None:
            return
        return_code = self.process.poll()
        if return_code is None:
            self.root.after(1000, self.check_process)
        elif return_code == 0:
            self.status.configure(text="Launcher completed.")
        else:
            self.status.configure(
                text=f"Launcher stopped with exit code {return_code}."
            )

    def cancel(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
        self.root.destroy()


def main() -> None:
    window = tk.Tk()
    StartupGui(window)
    window.mainloop()


if __name__ == "__main__":
    main()
