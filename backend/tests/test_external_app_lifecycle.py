from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import psutil

from app.tools import base, windows_apps


class ExternalLauncherUnitTests(unittest.TestCase):
    @patch("app.tools.base.subprocess.run")
    @patch("app.tools.base.is_windows", return_value=True)
    def test_direct_executable_uses_windows_cim_broker(self, _windows, run):
        run.return_value=Mock(returncode=0,stdout='{"pid": 321}\n',stderr="")
        pid=base.spawn_detached(r"C:\Apps\Example\example.exe",["--profile","one"],cwd=r"C:\Apps\Example")
        self.assertEqual(pid,321)
        argv=run.call_args.args[0]
        self.assertEqual(argv[0],"powershell.exe")
        self.assertIn("-EncodedCommand",argv)
        payload=json.loads(base64.b64decode(run.call_args.kwargs["env"]["HEBE_EXTERNAL_LAUNCH"]).decode("utf-8"))
        self.assertEqual(payload,{"command_line":r"C:\Apps\Example\example.exe --profile one","cwd":r"C:\Apps\Example"})
        self.assertEqual(run.call_args.kwargs["creationflags"],0x08000000)
        self.assertTrue(run.call_args.kwargs["capture_output"])

    @patch("app.tools.base.subprocess.run")
    @patch("app.tools.base.is_windows", return_value=True)
    def test_registered_command_uses_same_external_boundary(self, _windows, run):
        run.return_value=Mock(returncode=0,stdout='{"pid": 654}',stderr="")
        self.assertEqual(base.run_cmd_windows("start chrome"),654)
        payload=json.loads(base64.b64decode(run.call_args.kwargs["env"]["HEBE_EXTERNAL_LAUNCH"]).decode("utf-8"))
        self.assertIn("cmd.exe",payload["command_line"].casefold())
        self.assertIn("start chrome",payload["command_line"])

    @patch("app.tools.base.subprocess.run")
    @patch("app.tools.base.is_windows", return_value=True)
    def test_windows_shortcut_uses_shell_inside_same_external_boundary(self, _windows, run):
        run.return_value=Mock(returncode=0,stdout='{"pid": 655}',stderr="")
        self.assertEqual(base.spawn_detached(r"C:\Apps\Example App.lnk"),655)
        payload=json.loads(base64.b64decode(run.call_args.kwargs["env"]["HEBE_EXTERNAL_LAUNCH"]).decode("utf-8"))
        self.assertIn("explorer.exe",payload["command_line"].casefold())
        self.assertIn(r"C:\Apps\Example App.lnk",payload["command_line"])

    @patch("app.tools.base.subprocess.run")
    def test_cim_failure_is_not_reported_as_launch_success(self, run):
        run.return_value=Mock(returncode=1,stdout="",stderr="Win32_Process.Create failed: 2")
        with self.assertRaisesRegex(OSError,"failed: 2"):
            base._external_launcher({"command_line":"missing.exe","cwd":""})

    @patch("app.tools.windows_apps.time.sleep")
    @patch("app.tools.windows_apps.register_app_usage")
    @patch("app.tools.windows_apps.spawn_detached", return_value=9001)
    @patch("app.tools.windows_apps.is_process_running", side_effect=[False,True])
    @patch("app.tools.windows_apps.os.path.exists", return_value=True)
    def test_open_app_preserves_validation_and_usage_learning(self, _exists, _running, spawn, usage, _sleep):
        app={"id":4,"name":"Example","command":r"C:\Apps\example.exe","process_name":"example.exe"}
        self.assertTrue(windows_apps.open_app(app,lambda _text:None))
        spawn.assert_called_once_with(r"C:\Apps\example.exe",cwd=r"C:\Apps")
        usage.assert_called_once_with(4)

    @patch("app.tools.windows_apps.register_app_usage")
    @patch("app.tools.windows_apps.try_focus_app_window", return_value=True)
    @patch("app.tools.windows_apps.is_process_running", return_value=True)
    @patch("app.tools.windows_apps.spawn_detached")
    def test_already_open_application_is_focused_without_relaunch(self, spawn, _running, focus, usage):
        app={"id":5,"name":"Example","command":r"C:\Apps\example.exe","process_name":"example.exe"}
        self.assertTrue(windows_apps.open_app(app,lambda _text:None))
        focus.assert_called_once_with(app);spawn.assert_not_called();usage.assert_called_once_with(5)

    @patch("app.tools.windows_apps.time.sleep")
    @patch("app.tools.windows_apps.learn_process_name_after_launch", return_value="newapp.exe")
    @patch("app.tools.windows_apps.register_app_usage")
    @patch("app.tools.windows_apps.run_cmd_windows", return_value=9002)
    @patch("app.tools.windows_apps.is_process_running", side_effect=[False,True])
    def test_unknown_process_name_still_uses_learning_flow(self, _running, launch, usage, learn, _sleep):
        app={"id":6,"name":"New App","command":"newapp"}
        self.assertTrue(windows_apps.open_app(app,lambda _text:None))
        launch.assert_called_once_with("newapp");usage.assert_called_once_with(6)
        learn.assert_called_once_with(6,expected_exe="newapp.exe")

    def test_electron_keeps_recursive_internal_tree_cleanup(self):
        source=(Path(__file__).resolve().parents[2]/"frontend"/"electron"/"main.cjs").read_text(encoding="utf-8")
        self.assertIn('const args = ["/PID", String(pid), "/T"]',source)
        self.assertIn('if (force) args.push("/F")',source)
        self.assertIn("await stopBackendGracefully()",source)


@unittest.skipUnless(os.name=="nt","Windows process-tree integration test")
class ExternalLauncherWindowsIntegrationTests(unittest.TestCase):
    def test_user_app_survives_recursive_backend_tree_kill(self):
        repo=Path(__file__).resolve().parents[2]
        env=dict(os.environ);env["PYTHONPATH"]=str(repo/"backend")
        target_code="import time; time.sleep(60)"
        worker_code=(
            "import sys,time; from app.tools.base import spawn_detached; "
            f"pid=spawn_detached(sys.executable,['-c',{target_code!r}]); "
            "print(pid,flush=True); time.sleep(60)"
        )
        worker=subprocess.Popen([sys.executable,"-c",worker_code],cwd=repo,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        target_pid=0
        try:
            line=worker.stdout.readline().strip()
            target_pid=int(line)
            self.assertTrue(psutil.pid_exists(target_pid))
            self.assertNotIn(worker.pid,[parent.pid for parent in psutil.Process(target_pid).parents()])
            subprocess.run(["taskkill","/PID",str(worker.pid),"/T","/F"],check=False,capture_output=True)
            worker.wait(timeout=10);worker.communicate(timeout=2);time.sleep(.5)
            self.assertTrue(psutil.pid_exists(target_pid),"external application was killed with Hebe's internal tree")
        finally:
            if worker.poll() is None:subprocess.run(["taskkill","/PID",str(worker.pid),"/T","/F"],check=False,capture_output=True)
            if worker.stdout and not worker.stdout.closed:worker.stdout.close()
            if worker.stderr and not worker.stderr.closed:worker.stderr.close()
            if target_pid and psutil.pid_exists(target_pid):subprocess.run(["taskkill","/PID",str(target_pid),"/T","/F"],check=False,capture_output=True)


if __name__=="__main__":unittest.main()
