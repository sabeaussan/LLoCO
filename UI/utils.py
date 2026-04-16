import sys
import threading
import time
import random
from typing import List, Optional

from wcwidth import wcswidth


def show_logo():
	# TODO : add check working directory before loading the logo
	with open("UI/logo.txt", "r", encoding="utf-8") as file:
		logo = file.read()
	print(logo)


# ---------------------------------------------------------------------------
# Console formatting helpers
# ---------------------------------------------------------------------------

def hr(char: str = "─", n: int = 70) -> str:
    return char * n


def wlen(s: str) -> int:
    return wcswidth(s)


def box(title: str, lines: List[str]) -> str:
    width = max(wlen(title), *(wlen(x) for x in lines)) + 4
    top = "╭" + "─" * width + "╮"
    mid = [f"│  {title}{' ' * (width - 2 - wlen(title))}│"]
    for ln in lines:
        mid.append(f"│  {ln}{' ' * (width - 2 - wlen(ln))}│")
    bot = "╰" + "─" * width + "╯"
    return "\n".join([top] + mid + [bot])


def mode_str(all_flag, single_id, ids_csv, range_pair, start, limit) -> str:
    if all_flag:
        return "--all"
    if single_id is not None:
        return f"--id {single_id}"
    if ids_csv is not None:
        return f"--ids {ids_csv}"
    if range_pair is not None:
        return f"--range {range_pair[0]} {range_pair[1]}"
    s = "" if start is None else f"--start {start} "
    l = "" if limit is None else f"--limit {limit}"
    return (s + l).strip()


# ---------------------------------------------------------------------------
# Batch display functions
# ---------------------------------------------------------------------------

def print_batch_header(dataset_name, format_label, mode_label, problems_root, total, problem_timeout):
    print(
        box(
            "🚀 LLoCO Batch Runner",
            [
                f"Dataset: {dataset_name}",
                f"Source:  {format_label}",
                f"Mode:    {mode_label}",
                f"Root:    {problems_root}",
                f"Total:   {total} problem(s)",
                f"Timeout: {problem_timeout}s per problem",
            ],
        )
    )
    print()


def print_batch_dry_run(idx, total, problem_folder):
    print(f"[{idx}/{total}] 🧪 DRY_RUN {problem_folder}")


def print_batch_running(idx, total, problem_folder, problem_timeout):
    print(hr())
    print(f"[{idx}/{total}] ▶ Running {problem_folder}  (timeout {problem_timeout}s)")
    print(hr())


def print_batch_timeout(problem_folder, problem_timeout):
    print(
        f"\n⏱  TIMEOUT — {problem_folder} exceeded {problem_timeout}s, killing.",
        file=sys.stderr,
    )


def print_batch_result(status, ok, problem_folder, expected, objective, optim_path,
                       short_err=None, source_dir=None):
    if status == "OK" and ok is True:
        head = "✅ PASSED"
    elif status == "OK" and ok is False:
        head = "❌ FAILED"
    elif status == "OK" and ok is None:
        head = "⚠️ OK (no expected)"
    else:
        head = f"⚠️ {status}"

    src_label = f"  ({source_dir})" if source_dir else ""
    print(f"\n{head} — {problem_folder}{src_label}")
    print(f"   ├─ Expected:  {expected}")
    print(f"   ├─ Objective: {objective}")
    if short_err:
        print(f"   └─ Error:     {short_err}")
    else:
        print(f"   └─ Output:    {optim_path}")
    print()


def print_batch_summary(report_path, comparable, passed, timeouts, total):
    print(hr())
    print(f"✅ Report saved: {report_path}")
    print(f"📊 Comparable: {comparable} | Passed: {passed} | Timeouts: {timeouts} | Total: {total}")
      
class Spinner(threading.Thread):
	def __init__(self, description="Doing some work ...  "):
		super().__init__()
		self.spinner_active = False
		self.description = description

	def run(self):
		# ANSI escape codes for colors
		colors = [
			'\033[91m',  # Red
			'\033[92m',  # Green
			'\033[93m',  # Yellow
			'\033[94m',  # Blue
			'\033[95m',  # Purple
			'\033[96m',  # Cyan
		]
		# Reset color to default
		reset_color = '\033[0m'
		robot = "🤖 "
		spin_chars = "⠁,⠃,⠇,⠧,⠷,⠿,⠻,⠽,⠯,⠟".split(',')
		offset = 15
		while True:
			for i in range(len(spin_chars)):
				color = random.choice(colors)
				chars = " ".join([spin_chars[(i+j+2)%len(spin_chars)] for j in range(offset)])
				sys.stdout.write('\r'+ robot + self.description+ color + chars + reset_color)
				sys.stdout.flush()
				time.sleep(0.03)

			if not self.spinner_active:
				# Clear the previous spinner line completely
				line_length = len(robot + self.description + " " + chars)
				sys.stdout.write('\r' + ' ' * line_length)  # overwrite with spaces and return
				sys.stdout.flush()
				sys.stdout.write('\r' + robot + self.description + " Done !\n")  # print final message
				sys.stdout.flush()
				break


class SpinnerManager(object):

	def __init__(self, text_desc, active=True):
		self.text_desc = text_desc
		self.active = active

	def __enter__(self):
		if self.active:
			self.spinner = Spinner(self.text_desc)
			self.spinner.start()
			self.spinner.spinner_active = True

	def __exit__(self, type, value, traceback):
		if self.active:
			self.spinner.spinner_active = False
			self.spinner.join()