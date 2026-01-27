import argparse
import os
import llm_utils
import code_utils
import io_utils
import subprocess
from UI.utils import show_logo, SpinnerManager
import sys

PROBLEM_BASE_DIR = "problems"
PROMPT_DIR = "prompts"

def build_api_doc():
	import inspect
	from data import DataLoader

	def document_public_api(cls):
		"""
		Generate documentation for all public methods of a class.
		"""
		docs = ["# API documentation for DataLoader class\n"]
		for name, member in inspect.getmembers(cls, predicate=inspect.isfunction):
			if not name.startswith("_"):  # public only
				sig = inspect.signature(member)
				doc = inspect.getdoc(member)
				docs.append(f"def {name}{sig}\n\n\"\"\"{doc}\"\"\"\n")
		return "\n".join(docs)
	
	return document_public_api(DataLoader)

def run_solution(problem_path, code):
	import shutil

	output_file_path = os.path.join(problem_path, "solution.py")
	with open(output_file_path, "w", encoding="utf-8") as f:
		f.write(code)
	print()

	# Copy a libraries file from the current directory to the problem directory
	# TODO : messy, make the files a python package
	# TODO : check that result conatains no error or traceback -> self correct
	shutil.copy('optimization_utils.py', os.path.join(problem_path,"optimization_utils.py"))
	shutil.copy('utils.py', os.path.join(problem_path,"utils.py"))
	shutil.copy('log_utils.py', os.path.join(problem_path,"log_utils.py"))
	shutil.move('data.py', os.path.join(problem_path,"data.py"))

	# Run the solution
	results = subprocess.run(
		[sys.executable, "solution.py"],
		cwd=problem_path,
		stdout=subprocess.PIPE,
		stderr=subprocess.PIPE,
		universal_newlines=True  # Equivalent to text=True
	)
	return results

def run_baseline(problem_path, high_level_description):
	sys_prompt_path = os.path.join(PROMPT_DIR, "system_prompt_baseline.txt")
	code = llm_utils.ask_baseline(sys_prompt_path, high_level_description)
	output_file_path = os.path.join(problem_path, "baseline.py")
	with open(output_file_path, "w", encoding="utf-8") as f:
		f.write(code)
	results = subprocess.run(["python", "baseline.py"], cwd=problem_path)
	print(results)
	return

def get_high_level_description(problem_path):
	# Read problem description from the file
	if not os.path.exists(problem_path):
		raise FileNotFoundError(f"Problem directory {problem_path} does not exist.")
	
	problem_description_path = os.path.join(problem_path, "user_input.md")
	high_level_description = "# PROBLEM DESCRIPTION\n\n"
	with open(problem_description_path, "r", encoding="utf-8") as f:
		high_level_description += f.read()
	return high_level_description

def main(args):

	problem_dir = args.fname
	problem_path = os.path.join(PROBLEM_BASE_DIR, problem_dir)

	#--------------- UI ------------------
	if args.verbosity > 0:
		show_logo()

	#--------------- READ THE PROBLEM ------------------
	high_level_description = get_high_level_description(problem_path)

	if args.baseline:
		run_baseline(problem_path, high_level_description)

	#--------------- PROBLEM REFINEMENT ------------------
	with SpinnerManager("Analyzing the user inputs ...", active=args.verbosity > 0):
		csv_files_summary, has_csv_file = io_utils.get_csv_files_summary(problem_path)
		if args.interactive:
			refinement_questions = io_utils.refine_problem_description(
				os.path.join(PROMPT_DIR, "system_prompt_problem_framing.txt"),
				high_level_description + "\n\n" + csv_files_summary
			)

	if args.interactive:
		refinement = "# ADDITIONAL DETAILS\n\n"
		for q in refinement_questions["questions"]:
			answer = input(f"🤖 {q['question']} : ")
			refinement += f"Q: {q['question']}\nA: {answer}\n"
	else:
		refinement = ""

	#--------------- PROBLEM FORMALIZATION ------------------
	with SpinnerManager("Refining and formalizing the problem ...", active=args.verbosity > 0):
		sys_prompt_path = os.path.join(PROMPT_DIR, "system_prompt_problem_summary.txt")
		complete_description = llm_utils.summarize_problem_description(
			sys_prompt_path,
			high_level_description+ "\n\n" + csv_files_summary + "\n\n" + refinement
		)

	#--------------- DATA EXTRACTION ------------------
	# Process the input data
	if has_csv_file:
		with SpinnerManager("Now I need to extract and prepare the data...", active=args.verbosity > 0):
			sys_prompt_path = os.path.join(PROMPT_DIR, "system_prompt_dataloader.txt")
			input_files_description, has_csv_file = io_utils.convert_file_to_json(problem_path, complete_description)
			context = complete_description+ "\n\n" + csv_files_summary + "\n\n" + input_files_description

			# Create data.py
			code_data = llm_utils.data_processing(sys_prompt_path, context)
			with open("data.py", "w", encoding="utf-8") as f:
				f.write(code_data)
			
			# Generate API documentation for DataLoader
			api_doc = build_api_doc()
	else:
		code_data = ""

	#--------------- MODEL IMPLEMENTATION ------------------
	# Build system prompt for formalization
	# Initialize context with necessary basic imports
	# TODO : make the system write imports
	code_base = code_utils.define_imports()

	with SpinnerManager("Ok time to code the model !", active=args.verbosity > 0):
		sys_prompt_path = os.path.join(PROMPT_DIR, "system_prompt_code_.txt")
		context = complete_description 
		if has_csv_file:
			context += "\n\n" + csv_files_summary
			context += "\n\n" + input_files_description

		code_optimization = llm_utils.implement_optimization(sys_prompt_path, context, code_base, api_doc)

	#--------------- SOLUTION RENDERING ------------------
	with SpinnerManager("Almost there ! Just missing the final touch now ...", active=args.verbosity > 0):
		sys_prompt_path = os.path.join(PROMPT_DIR, "system_prompt_sol_print.txt")
		code_summary = code_utils.add_print_summary()
		solution_code = code_optimization + code_summary
		code_print = llm_utils.print_solution(sys_prompt_path, complete_description+"\n\n" + csv_files_summary, solution_code, api_doc)
		solution_code += "\n\n" + code_print
	optim_summary = run_solution(problem_path, solution_code)
	optim_summary_path = os.path.join(problem_path, "optim_summary.txt")
	with open(optim_summary_path, "w", encoding="utf-8") as f:
		f.write(optim_summary.stdout)
		if optim_summary.stderr:
			f.write("\n\n[stderr]\n")
			f.write(optim_summary.stderr)

	if args.verbosity > 0:
		print(optim_summary)

	# if not optim_summary.stderr=="":
	# 	raise ValueError(optim_summary.stderr)

	if args.verbosity > 1:
		sys_prompt_path = os.path.join(PROMPT_DIR, "system_prompt_write_report.txt")
		report = llm_utils.write_report(sys_prompt_path, complete_description, optim_summary.stdout)
		report_path = os.path.join(problem_path, "report.txt")
		with open(report_path, "w", encoding="utf-8") as f:
			f.write(report)
		print("\n🤖 Lets see what we got : \n\n")
		print(report)
		
	

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description="Run a an optimization to solve a problem.")
	parser.add_argument(
		"-f", "--fname", type=str, required=True, help="Text file containing the problem description."
	)
	parser.add_argument(
		"-v", "--verbosity", type=int, default=1, required=False, help="Verbosity level : only print code exec (0), print spinner (1) and full debug (2)."
	)
	parser.add_argument(
		"-i", "--interactive", action="store_true", help="If set, LLoCO will ask multiple questions to dismiss potential ambiguity."
	)
	parser.add_argument(
		"-b", "--baseline", action="store_true", help="Run baseline."
	)
	args = parser.parse_args()

	main(args)
	
	



	
	
	
	
	
	
	
