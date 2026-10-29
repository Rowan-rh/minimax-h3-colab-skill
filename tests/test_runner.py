from __future__ import annotations

import ast
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import runner


FAKE_COLAB = r'''#!PYTHON_EXECUTABLE
import json, os, pathlib, shutil, sys
args = sys.argv[1:]
if args and args[0].startswith("--auth="):
    args = args[1:]
command = args[0]
args = args[1:]
root = pathlib.Path(os.environ["FAKE_COLAB_STATE_DIR"])
root.mkdir(parents=True, exist_ok=True)
log = root / "calls.jsonl"
def record(kind, **values):
    with log.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"command": kind, **values}) + "\n")
if command == "usage":
    print("Current balance: 100.00 compute units")
    print("Usage rate: 0.00/hr")
    print("Active assignments: 0")
elif command == "new":
    record("new", args=args)
    print("Session created")
elif command == "stop":
    record("stop", args=args)
    print("Session stopped")
elif command == "upload":
    remote = args[-1]
    source = pathlib.Path(args[-2])
    remote_path = root / "remote" / remote.lstrip("/")
    remote_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, remote_path)
    record("upload", remote=remote, source=source.name)
elif command == "exec":
    envs = [args[i + 1] for i, value in enumerate(args[:-1]) if value == "--env"]
    values = dict(item.split("=", 1) for item in envs)
    output = values["H3_OUTPUT_PATH"]
    record(
        "exec", output=output, mode=values["H3_INFERENCE_MODE"],
        refs=json.loads(values["H3_REFERENCE_IMAGES"]) if "H3_REFERENCE_IMAGES" in values else [],
        input_image=values.get("H3_INPUT_IMAGE"), prompt=values["H3_PROMPT_FILE"],
        width=values.get("H3_WIDTH"), height=values.get("H3_HEIGHT"),
    )
    if os.environ.get("FAKE_LONG_LOG"):
        print(os.environ["FAKE_LONG_LOG"])
        print("Prompt leak <Picture 1> should be filtered")
    if values.get("H3_OUTPUT_PREFIX") == os.environ.get("FAKE_FAIL_PREFIX"):
        print("simulated inference failure", file=sys.stderr)
        sys.exit(7)
    remote_path = root / "remote" / output.lstrip("/")
    remote_path.parent.mkdir(parents=True, exist_ok=True)
    remote_path.write_bytes(b"fake video bytes")
elif command == "download":
    remote = args[-2]
    target = pathlib.Path(args[-1])
    source = root / "remote" / remote.lstrip("/")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    record("download", remote=remote, target=target.name)
else:
    print("unexpected fake command: " + command, file=sys.stderr)
    sys.exit(2)
'''

FAKE_FFPROBE = r'''#!PYTHON_EXECUTABLE
print('{"streams":[{"codec_type":"video"},{"codec_type":"audio"}]}')
'''


class RunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.fake_bin = self.root / "bin"
        self.fake_bin.mkdir()
        for name, content in (("colab", FAKE_COLAB), ("ffprobe", FAKE_FFPROBE)):
            path = self.fake_bin / name
            path.write_text(content.replace("PYTHON_EXECUTABLE", sys.executable), encoding="utf-8")
            path.chmod(0o755)
        self.state_dir = self.root / "state"
        self.env_patch = patch.dict(os.environ, {
            "PATH": f"{self.fake_bin}:{Path(sys.executable).parent}:/usr/bin:/bin",
            "FAKE_COLAB_STATE_DIR": str(self.state_dir),
        })
        self.env_patch.start()

    def tearDown(self) -> None:
        self.env_patch.stop()
        self.temp.cleanup()

    def manifest(self, job_count: int = 2, prompt_file: bool = False) -> Path:
        jobs = []
        for index in range(job_count):
            image = self.root / f"reference-{index}.png"
            image.write_bytes(b"fake image")
            job = {
                "id": f"job-{index}",
                "title": f"clip {index}",
                "reference_images": [str(image)],
                "prompt": f"complete prompt <Picture 1> for clip {index}",
                "duration_seconds": 8,
                "seed": 100 + index,
                "output_name": f"clip-{index}",
            }
            if prompt_file:
                prompt = self.root / f"prompt-{index}.txt"
                prompt.write_text(job.pop("prompt"), encoding="utf-8")
                job["prompt_file"] = str(prompt)
            jobs.append(job)
        manifest = self.root / "manifest.json"
        manifest.write_text(json.dumps({"jobs": jobs}), encoding="utf-8")
        return manifest

    def calls(self) -> list[dict[str, object]]:
        path = self.state_dir / "calls.jsonl"
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    def test_usage_parser_handles_labelled_colab_cli_output(self) -> None:
        result = runner.parse_usage("Current balance: 1,234.50 compute units\nUsage rate: 12.00/hr\nActive assignments: 3\n")
        self.assertEqual(result["balance"], 1234.5)
        self.assertEqual(result["active_assignments"], 3)

    def test_usage_command_reads_colab_through_cli(self) -> None:
        result = runner.get_usage()
        self.assertEqual(result["balance"], 100.0)
        self.assertEqual(result["rate_per_hour"], 0.0)
        self.assertEqual(result["active_assignments"], 0)

    def test_guided_prompt_builds_shots_and_checks_picture_mapping(self) -> None:
        prompt = runner.compose_ref2va_prompt(
            subject_definitions="<Subject 1> is the presenter in <Picture 1>.",
            summary="A short introduction.",
            retention_analysis="<Subject 1> remains consistent across [Shot 1].",
            shots=[
                {"start_seconds": 0, "description": "Faces the camera.", "dialogue": "大家好！"},
                {"start_seconds": 4.2, "description": "Gestures toward a terminal card."},
            ],
            overall_soundscape="Quiet studio ambience.",
            non_diegetic_music="N/A",
            image_count=1,
            duration_seconds=8,
            detailed_description="Opening: a clean blue studio with a terminal card beside the presenter.",
        )
        self.assertIn("<d>[Chinese] 大家好！</d>", prompt)
        self.assertIn("[Shot 2] At 00:04.200", prompt)
        self.assertIn("Opening: a clean blue studio", prompt)
        with self.assertRaisesRegex(ValueError, "Picture"):
            runner.compose_ref2va_prompt(
                subject_definitions="<Subject 1> uses <Picture 2>.",
                summary="Summary", retention_analysis="Retention",
                shots=[{"start_seconds": 0, "description": "Faces camera."}],
                overall_soundscape="Room tone", non_diegetic_music="N/A",
                image_count=1, duration_seconds=8,
            )

    def test_batch_reuses_one_session_and_preserves_prompt_file_content(self) -> None:
        manifest = self.manifest(2, prompt_file=True)
        state_path = self.root / "progress.json"
        result = runner.run_batch(
            manifest, session=None, gpu="A100", high_mem=True, stop_on_complete=False,
            progress_path=state_path, output_dir=self.root / "outputs", exec_timeout=100,
        )
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["completed_count"], 2)
        self.assertEqual(result["session_status"], "stopped")
        self.assertTrue(all(Path(item["output"]).is_file() for item in result["results"]))
        calls = self.calls()
        self.assertEqual(sum(call["command"] == "new" for call in calls), 1)
        self.assertEqual(sum(call["command"] == "exec" for call in calls), 2)
        self.assertEqual(sum(call["command"] == "stop" for call in calls), 1)
        exec_calls = [call for call in calls if call["command"] == "exec"]
        self.assertTrue(all(call["mode"] == "reference" for call in exec_calls))
        self.assertTrue(all(call["width"] == "1344" and call["height"] == "768" for call in exec_calls))
        uploads = [call for call in calls if call["command"] == "upload"]
        self.assertEqual(len({call["remote"] for call in uploads}), 4)

    def test_failed_later_job_keeps_completed_video_and_stops_session(self) -> None:
        manifest = self.manifest(2)
        with patch.dict(os.environ, {"FAKE_FAIL_PREFIX": "MiniMax_H3_job-1"}):
            result = runner.run_batch(
                manifest, session=None, gpu="A100", high_mem=False, stop_on_complete=False,
                progress_path=None, output_dir=self.root / "outputs", exec_timeout=100,
            )
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["completed_count"], 1)
        self.assertEqual(result["failed_count"], 1)
        self.assertTrue(Path(result["results"][0]["output"]).is_file())
        self.assertEqual(sum(call["command"] == "stop" for call in self.calls()), 1)

    def test_shell_launcher_accepts_multiple_local_images_and_prompt_file(self) -> None:
        image_one = self.root / "ref-one.png"
        image_two = self.root / "ref-two.jpg"
        prompt = self.root / "prompt.txt"
        output = self.root / "from-shell.mp4"
        image_one.write_bytes(b"fake png bytes")
        image_two.write_bytes(b"fake jpeg bytes")
        prompt_content = "complete prompt with <Picture 1> and <Picture 2>\n"
        prompt.write_text(prompt_content, encoding="utf-8")
        project = Path(__file__).resolve().parents[1]
        subprocess.run([
            str(project / "run_colab_inference.sh"),
            "--image", str(image_one), "--image", str(image_two),
            "--prompt", str(prompt), "--output", str(output),
        ], env=os.environ.copy(), capture_output=True, text=True, check=True, timeout=30)
        self.assertEqual(output.read_bytes(), b"fake video bytes")
        calls = self.calls()
        self.assertEqual(sum(call["command"] == "new" for call in calls), 1)
        self.assertEqual(sum(call["command"] == "exec" for call in calls), 1)
        self.assertEqual(sum(call["command"] == "stop" for call in calls), 1)
        exec_call = next(call for call in calls if call["command"] == "exec")
        self.assertEqual(len(exec_call["refs"]), 2)
        uploaded_prompt = self.state_dir / "remote" / str(exec_call["prompt"]).lstrip("/")
        self.assertEqual(uploaded_prompt.read_text(encoding="utf-8"), prompt_content)

    def test_shell_launcher_passes_first_frame_mode_and_dimensions(self) -> None:
        image = self.root / "first-frame.png"
        prompt = self.root / "prompt.txt"
        image.write_bytes(b"fake png bytes")
        prompt.write_text("A concise motion prompt.\n", encoding="utf-8")
        project = Path(__file__).resolve().parents[1]
        subprocess.run([
            str(project / "run_colab_inference.sh"),
            "--image", str(image), "--prompt", str(prompt),
            "--mode", "first_frame", "--width", "512", "--height", "512",
        ], env=os.environ.copy(), capture_output=True, text=True, check=True, timeout=30)
        exec_call = next(call for call in self.calls() if call["command"] == "exec")
        reference_upload = next(
            call for call in self.calls()
            if call["command"] == "upload" and call["remote"].endswith("_reference_1.img")
        )
        self.assertEqual(exec_call["mode"], "first_frame")
        self.assertEqual(exec_call["input_image"], reference_upload["remote"])
        self.assertEqual((exec_call["width"], exec_call["height"]), ("512", "512"))

    def test_resolve_jobs_rejects_duplicate_job_ids_and_outputs(self) -> None:
        manifest = self.manifest(2)
        data = json.loads(manifest.read_text(encoding="utf-8"))
        data["jobs"][1]["id"] = data["jobs"][0]["id"]
        with self.assertRaisesRegex(ValueError, "duplicates job id"):
            runner.resolve_jobs(data, self.root / "output")
        data["jobs"][1]["id"] = "job-1"
        data["jobs"][1]["output_path"] = str(self.root / "same.mp4")
        data["jobs"][0]["output_path"] = str(self.root / "same.mp4")
        with self.assertRaisesRegex(ValueError, "same output path"):
            runner.resolve_jobs(data, self.root / "output")

    def test_resolve_jobs_rejects_picture_tags_outside_uploaded_references(self) -> None:
        manifest = self.manifest(1)
        data = json.loads(manifest.read_text(encoding="utf-8"))
        data["jobs"][0]["prompt"] += " <Picture 2>"
        manifest.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Picture"):
            runner.resolve_jobs(data, self.root / "output")

    def test_first_frame_mode_sets_single_image_and_output_dimensions(self) -> None:
        manifest = self.manifest(1)
        data = json.loads(manifest.read_text(encoding="utf-8"))
        data["jobs"][0].update({"inference_mode": "first_frame", "width": 512, "height": 512})
        manifest.write_text(json.dumps(data), encoding="utf-8")

        result = runner.run_batch(
            manifest, session=None, gpu="A100", high_mem=False, stop_on_complete=False,
            progress_path=None, output_dir=self.root / "outputs", exec_timeout=100,
        )

        self.assertEqual(result["status"], "completed")
        exec_call = next(call for call in self.calls() if call["command"] == "exec")
        self.assertEqual(exec_call["mode"], "first_frame")
        self.assertEqual(exec_call["refs"], [])
        self.assertEqual(exec_call["input_image"], "/content/h3_job-0_reference_1.img")
        self.assertEqual((exec_call["width"], exec_call["height"]), ("512", "512"))

    def test_resolve_jobs_validates_mode_reference_count_and_dimensions(self) -> None:
        manifest = self.manifest(2)
        data = json.loads(manifest.read_text(encoding="utf-8"))
        resolved = runner.resolve_jobs(data, self.root / "output")
        self.assertEqual(resolved[0]["inference_mode"], "reference")
        self.assertEqual((resolved[0]["width"], resolved[0]["height"]), (1344, 768))

        data["jobs"][0]["inference_mode"] = "first_frame"
        data["jobs"][0]["reference_images"].append(data["jobs"][0]["reference_images"][0])
        with self.assertRaisesRegex(ValueError, "exactly one"):
            runner.resolve_jobs(data, self.root / "output")

        data["jobs"][0]["reference_images"].pop()
        data["jobs"][0].update({"width": 1376, "height": 768})
        with self.assertRaisesRegex(ValueError, "may not exceed"):
            runner.resolve_jobs(data, self.root / "output")
        data["jobs"][0].update({"width": 513, "height": 512})
        with self.assertRaisesRegex(ValueError, "divisible by 32"):
            runner.resolve_jobs(data, self.root / "output")

    def test_notebook_code_compiles_and_applies_dimension_environment_and_limits(self) -> None:
        notebook_path = Path(__file__).resolve().parents[1] / "assets" / "MiniMax_H3_Turbo_Colab.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        code_sources = [
            (index, "".join(cell.get("source", [])))
            for index, cell in enumerate(notebook["cells"])
            if cell.get("cell_type") == "code"
        ]
        for index, source in code_sources:
            compile(source, f"{notebook_path}#cell-{index}", "exec")

        config = None
        for index, source in code_sources:
            tree = ast.parse(source)
            assigned = {
                target.id
                for node in ast.walk(tree)
                if isinstance(node, (ast.Assign, ast.AnnAssign))
                for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
                if isinstance(target, ast.Name)
            }
            assertion_names = {
                name.id
                for node in ast.walk(tree)
                if isinstance(node, ast.Assert)
                for name in ast.walk(node.test)
                if isinstance(name, ast.Name)
            }
            if {"WIDTH", "HEIGHT"} <= assigned and {"WIDTH", "HEIGHT"} <= assertion_names:
                config = (index, source)
                break
        self.assertIsNotNone(config, "Could not find the Notebook cell that assigns and validates output dimensions.")
        cell_index, config_source = config

        def execute_config(width: str, height: str) -> dict[str, object]:
            namespace: dict[str, object] = {"os": os, "INFERENCE_MODE": "first_frame"}
            with patch.dict(os.environ, {
                "H3_WIDTH": width,
                "H3_HEIGHT": height,
                "H3_DURATION_SECONDS": "5",
                "H3_SEED": "123",
            }), contextlib.redirect_stdout(io.StringIO()):
                exec(compile(config_source, f"{notebook_path}#cell-{cell_index}", "exec"), namespace)
            return namespace

        valid = execute_config("512", "512")
        self.assertEqual((valid["WIDTH"], valid["HEIGHT"]), (512, 512))
        for width, height in (("1376", "768"), ("513", "512")):
            with self.subTest(width=width, height=height), self.assertRaises(AssertionError):
                execute_config(width, height)

    def test_log_line_truncates_long_diagnostics_while_filtering_prompt_markers(self) -> None:
        manifest = self.manifest(1)
        state_path = self.root / "progress.json"
        long_Diagnostic = "AssertionError: Free disk GiB below threshold: " + ("x" * 350)
        with patch.dict(os.environ, {"FAKE_LONG_LOG": long_Diagnostic}):
            runner.run_batch(
                manifest, session=None, gpu="A100", high_mem=False, stop_on_complete=False,
                progress_path=state_path, output_dir=self.root / "outputs", exec_timeout=100,
            )
        progress = json.loads(state_path.read_text(encoding="utf-8"))
        truncated = [line for line in progress["log_tail"] if line.startswith("AssertionError:")]
        self.assertEqual(len(truncated), 1)
        self.assertEqual(len(truncated[0]), 320)
        self.assertTrue(truncated[0].endswith("..."))
        self.assertFalse(any("<Picture" in line for line in progress["log_tail"]))

    def test_install_script_defaults_to_dot_codex_skills_when_codex_home_unset(self) -> None:
        project = Path(__file__).resolve().parents[1]
        fake_home = self.root / "home"
        fake_home.mkdir()
        env = {k: v for k, v in os.environ.items() if k != "CODEX_HOME"}
        env["HOME"] = str(fake_home)
        subprocess.run(
            [str(project / "install.sh")],
            env=env, capture_output=True, text=True, check=True, timeout=30,
        )
        self.assertTrue((fake_home / ".codex" / "skills" / "minimax-h3-colab" / "SKILL.md").is_file())
        self.assertFalse((fake_home / "skills").exists())

    def test_resolve_jobs_composes_structured_ref2va_shots_manifest(self) -> None:
        image = self.root / "speaker.png"
        image.write_bytes(b"fake image")
        manifest_data = {
            "jobs": [
                {
                    "id": "structured-1",
                    "title": "structured dialogue clip",
                    "reference_images": [str(image)],
                    "duration_seconds": 10,
                    "subject_definitions": "<Subject 1> is the lawyer in <Picture 1>.",
                    "summary": "A courtroom questioning clip.",
                    "retention_analysis": "<Subject 1> maintains consistent glasses and suit.",
                    "shots": [
                        {
                            "start_seconds": 0,
                            "description": "Medium close-up of <Subject 1> speaking.",
                            "speaker": "<Subject 1>",
                            "dialogue": "十点零五分打给你的人是谁？",
                        },
                        {
                            "start_seconds": 5.0,
                            "description": "Tight close-up of <Subject 1>.",
                        },
                    ],
                    "overall_soundscape": "Quiet room tone with rain.",
                    "non_diegetic_music": "Subtle cello drone.",
                }
            ]
        }
        resolved = runner.resolve_jobs(manifest_data, self.root / "output")
        self.assertEqual(len(resolved), 1)
        self.assertIn("<d>[Chinese] 十点零五分打给你的人是谁？</d>", resolved[0]["prompt"])
        self.assertIn("[Shot 2] At 00:05.000", resolved[0]["prompt"])

    def test_call_colab_with_retry_recovers_from_transient_failure(self) -> None:
        attempts = 0

        def flaky_call(*args: object, **kwargs: object) -> str:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise runner.ColabCommandError("upload", 1, ["WebSocketConnectionClosedException"])
            return "uploaded"

        with patch.object(runner, "call_colab", side_effect=flaky_call):
            result = runner.call_colab_with_retry(
                ["upload", "--session", "s1", "a", "b"],
                label="upload",
                timeout=10,
                retry_delay=0.01,
            )
        self.assertEqual(result, "uploaded")
        self.assertEqual(attempts, 2)


if __name__ == "__main__":
    unittest.main()
